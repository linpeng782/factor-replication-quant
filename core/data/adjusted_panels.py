"""
后复权宽表面板构建（从 my-alpha-engine 抽取，路径改读 config）

核心逻辑：
  - 磁盘只存原始价格（不复权） + 复权因子（稀疏 cum_factor）
  - 后复权价格 = 原始价格 × cum_factor（按日期 ffill）
  - 后复权成交量 = 原始成交量 ÷ cum_factor

  ⚠️ 后复权锚定起点、cum_factor 向后累乘单调增：新除权只影响除权日及之后，
     历史后复权价不变 —— 故日频可安全 append + 重算最近窗口（无需全量重算）。

提供两类接口：
  - load_adjusted_for_stock : 单只股票后复权
  - load_adjusted_panels    : 全市场后复权宽表面板（多进程并行）
"""

from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
from loguru import logger

import config


# ==================== 基础读取 ====================

def list_available_stocks():
    """列出本地已下载的所有股票代码（按 RAW_OHLCV_DIR 下的 parquet）"""
    if not config.RAW_OHLCV_DIR.exists():
        return []
    return [f.stem for f in sorted(config.RAW_OHLCV_DIR.glob("*.parquet"))]


def load_raw_ohlcv(stock):
    """读取单只股票原始 OHLCV（不复权），不存在返回 None"""
    path = config.RAW_OHLCV_DIR / f"{stock}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_ex_factors(stock):
    """读取单只股票复权因子（稀疏表），无除权记录返回 None"""
    path = config.EX_FACTORS_DIR / f"{stock}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ==================== 单股票复权 ====================

def adjust_ohlcv(raw, ex):
    """对单只股票做后复权：价格 × cum_factor（ffill），成交量 ÷ cum_factor。"""
    if ex is None or ex.empty:
        cum_factor = pd.Series(1.0, index=raw.index, name="ex_cum_factor")
    else:
        cum_factor = ex["ex_cum_factor"].reindex(raw.index, method="ffill").fillna(1.0)

    adj = raw.copy()
    for col in config.PRICE_FIELDS:
        if col in adj.columns:
            adj[col] = adj[col] * cum_factor
    for col in config.VOLUME_FIELDS:
        if col in adj.columns:
            adj[col] = adj[col] / cum_factor
    # total_turnover 不变（= 原始价×原始量 = 后复权价×后复权量）
    return adj


def load_adjusted_for_stock(stock, start=None, end=None):
    """加载单只股票后复权数据（含日期切片），文件不存在返回 None。"""
    raw = load_raw_ohlcv(stock)
    if raw is None:
        return None
    adj = adjust_ohlcv(raw, load_ex_factors(stock))
    if start is not None:
        adj = adj.loc[adj.index >= pd.Timestamp(start)]
    if end is not None:
        adj = adj.loc[adj.index <= pd.Timestamp(end)]
    return adj


# ==================== 批量并行加载（因子计算用） ====================

def _worker_load_adjusted(stock, raw_dir, ex_dir, start, end, fields):
    """子进程工作函数（独立函数以便 pickle）。"""
    raw_path = Path(raw_dir) / f"{stock}.parquet"
    if not raw_path.exists():
        return stock, None
    raw = pd.read_parquet(raw_path)

    ex_path = Path(ex_dir) / f"{stock}.parquet"
    if ex_path.exists():
        ex = pd.read_parquet(ex_path)
        cum_factor = ex["ex_cum_factor"].reindex(raw.index, method="ffill").fillna(1.0)
    else:
        cum_factor = pd.Series(1.0, index=raw.index)

    cols = [c for c in fields if c in raw.columns]
    adj = raw[cols].copy()
    price_fields_in_raw = [
        c for c in ["open", "high", "low", "close", "limit_up", "limit_down"] if c in cols
    ]
    volume_fields_in_raw = [c for c in ["volume"] if c in cols]
    for col in price_fields_in_raw:
        adj[col] = adj[col] * cum_factor
    for col in volume_fields_in_raw:
        adj[col] = adj[col] / cum_factor

    if start is not None:
        adj = adj.loc[adj.index >= pd.Timestamp(start)]
    if end is not None:
        adj = adj.loc[adj.index <= pd.Timestamp(end)]
    return stock, adj


def load_adjusted_panels(
    stocks=None,
    start=None,
    end=None,
    fields=("open", "high", "low", "close", "volume", "total_turnover"),
    n_workers=None,
):
    """并行加载全市场后复权 panel，返回 dict[field -> DataFrame(date × stock)]。

    虚拟字段 'vwap'：不在磁盘，由 total_turnover / volume 计算（后复权口径）。
    """
    if stocks is None:
        stocks = list_available_stocks()
    if n_workers is None:
        n_workers = config.COMPUTE_WORKERS

    fields = list(fields)
    need_vwap = "vwap" in fields
    disk_fields = [f for f in fields if f != "vwap"]
    if need_vwap:
        for dep in ("total_turnover", "volume"):
            if dep not in disk_fields:
                disk_fields.append(dep)

    logger.info(f"并行加载 {len(stocks)} 只股票后复权数据，进程数={n_workers}")
    worker_fn = partial(
        _worker_load_adjusted,
        raw_dir=str(config.RAW_OHLCV_DIR),
        ex_dir=str(config.EX_FACTORS_DIR),
        start=start, end=end, fields=disk_fields,
    )

    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for stock, df in executor.map(worker_fn, stocks, chunksize=20):
            if df is not None and not df.empty:
                results[stock] = df
    logger.info(f"成功加载 {len(results)} / {len(stocks)} 只股票")

    panels = {}
    for field in disk_fields:
        series_dict = {s: df[field] for s, df in results.items() if field in df.columns}
        if series_dict:
            panels[field] = pd.DataFrame(series_dict).sort_index()

    if need_vwap and "total_turnover" in panels and "volume" in panels:
        panels["vwap"] = panels["total_turnover"] / panels["volume"].replace(0, np.nan)
        logger.info(f"已计算 vwap 字段，shape={panels['vwap'].shape}")

    return {f: panels[f] for f in fields if f in panels}
