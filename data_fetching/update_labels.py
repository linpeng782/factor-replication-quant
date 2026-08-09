"""
日频增量更新：vwap_panel + forward_return labels（VWAP 口径，dquant 后端）
==========================================================================
设计同构 alpha158/daily_update.py：
  last = 现有 vwap_panel 最大日期；T = raw OHLCV 最新日期；need = (last, T]
  W    = max(horizons) + 1 + buffer → 回读窗口 = [last - W, T]（覆盖最早新日的 warmup）
  ① 并行读 raw OHLCV 窗口段 → 读时复权 → 算 vwap → pivot 临时宽表
  ② vwap_panel：切 (last, T] 新行 → concat(列并集自动纳新股) → dedup(keep last) → 原子写
  ③ 逐 horizon：从新 vwap_panel 重算 forward_return → 切 (last, T] 新行 → append 到 labels

零 API：只读已落盘的 raw_ohlcv / ex_factors（data_fetching 日更负责拉取）。
首次须先有基线：若无 vwap_panel，先跑 build_labels.py（同后端 env）。

用法：python data_fetching/update_labels.py
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from alpha_shared.evaluation.returns import build_forward_returns

FACTOR_DTYPE = np.float32
HORIZONS = (1, 2, 5, 10, 20)
W = max(HORIZONS) + 1          # forward_return 需要 T+1+N，回读窗口至少覆盖
BUFFER = 10                    # warmup 安全余量（交易日）
N_READ_WORKERS = 64
RAW_DIR = config.RAW_OHLCV_DIR
EX_DIR = config.EX_FACTORS_DIR


# ==================== 源读取（读时复权）====================
def _load_one(stock: str, raw_dir: str, ex_dir: str, cutoff: str) -> pd.DataFrame | None:
    raw_path = Path(raw_dir) / f"{stock}.parquet"
    if not raw_path.exists():
        return None
    raw = pd.read_parquet(raw_path)
    raw.index = pd.to_datetime(raw.index)
    raw = raw[raw.index >= pd.Timestamp(cutoff)]
    if raw.empty:
        return None
    ex_path = Path(ex_dir) / f"{stock}.parquet"
    if ex_path.exists():
        ex = pd.read_parquet(ex_path)
        cum = ex["ex_cum_factor"].dropna().reindex(raw.index, method="ffill").fillna(1.0)
    else:
        cum = pd.Series(1.0, index=raw.index)
    adj = raw[["total_turnover", "volume"]].copy()
    adj["volume"] = adj["volume"] / cum
    adj["vwap"] = adj["total_turnover"] / adj["volume"].replace(0, np.nan)
    adj["stock_code"] = stock
    return adj.reset_index().rename(columns={"date": "datetime"})


def load_vwap_panel(stocks: list[str], cutoff: str) -> pd.DataFrame:
    """并行读 raw OHLCV 窗口段 → 复权 → 算 vwap → pivot 宽表。"""
    worker = partial(_load_one, raw_dir=str(RAW_DIR), ex_dir=str(EX_DIR), cutoff=cutoff)
    dfs = []
    with ProcessPoolExecutor(max_workers=N_READ_WORKERS) as ex:
        for fut in as_completed([ex.submit(worker, s) for s in stocks]):
            r = fut.result()
            if r is not None:
                dfs.append(r)
    if not dfs:
        return pd.DataFrame()
    long = pd.concat(dfs, ignore_index=True)
    wide = long.pivot(index="datetime", columns="stock_code", values="vwap")
    return wide.sort_index().sort_index(axis=1)


# ==================== 基线探查 ====================
def baseline_last_date() -> pd.Timestamp | None:
    """现有 vwap_panel 的最大日期。无基线返回 None。"""
    if not config.VWAP_PANEL_PATH.exists():
        return None
    df = pd.read_parquet(config.VWAP_PANEL_PATH, columns=[])
    return pd.to_datetime(df.index).max()


def list_stocks() -> list[str]:
    return sorted(p.stem for p in RAW_DIR.glob("*.parquet"))


# ==================== 原子写 ====================
def _atomic_write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    tmp.replace(path)


# ==================== 主流程 ====================
def update():
    logger.info("=" * 72)
    logger.info("vwap_panel + labels 增量日更开始（dquant 后端）")
    logger.info("=" * 72)

    last = baseline_last_date()
    if last is None:
        logger.error(f"基线不存在: {config.VWAP_PANEL_PATH}，请先跑 build_labels.py 全量构建")
        return

    # 源数据最新日期
    stocks = list_stocks()
    sample = pd.read_parquet(RAW_DIR / f"{stocks[0]}.parquet", columns=[])
    src_end = pd.to_datetime(sample.index).max()
    logger.info(f"  基线末日={last.date()} | 源最新={src_end.date()} | 股票数={len(stocks)}")

    if src_end <= last:
        logger.info("  源无新数据，跳过")
        return

    # 回读窗口：last - (W + buffer) 个交易日 ~ src_end
    cutoff = last - pd.Timedelta(days=(W + BUFFER) * 2)  # 日历日近似（交易日×1.5）
    logger.info(f"  回读窗口 cutoff={cutoff.date()} ~ {src_end.date()}")

    t0 = time.time()
    logger.info("[1/3] 并行加载 vwap 面板窗口段...")
    new_vwap = load_vwap_panel(stocks, cutoff=str(cutoff.date()))
    logger.info(f"    新 vwap shape={new_vwap.shape}, 耗时 {time.time()-t0:.1f}s")

    # ========== vwap_panel append ==========
    t0 = time.time()
    logger.info("[2/3] vwap_panel 增量 append...")
    old_vwap = pd.read_parquet(config.VWAP_PANEL_PATH)
    old_vwap.index = pd.to_datetime(old_vwap.index)
    new_rows = new_vwap.loc[new_vwap.index > last]
    if new_rows.empty:
        logger.info("  无新行，跳过")
        return
    combined = pd.concat([old_vwap, new_rows])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index().sort_index(axis=1)
    combined = combined.astype(FACTOR_DTYPE)
    _atomic_write(combined, config.VWAP_PANEL_PATH)
    logger.info(
        f"    vwap_panel: {old_vwap.shape} -> {combined.shape} "
        f"(+{len(new_rows)} 天), 耗时 {time.time()-t0:.1f}s"
    )

    # ========== forward_return append ==========
    t0 = time.time()
    logger.info(f"[3/3] forward_return 增量 append (horizons={HORIZONS})...")
    # 从合并后的完整 vwap 面板重算 forward_return（pct_change + shift 是向量化的，全量算很快）
    returns_dict = build_forward_returns(combined.astype(np.float64), horizons=HORIZONS)

    config.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for n, r in returns_dict.items():
        path = config.LABELS_DIR / f"forward_return_{n}d.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            old.index = pd.to_datetime(old.index)
            new_r = r.loc[r.index > last]
            merged = pd.concat([old, new_r])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index().sort_index(axis=1)
        else:
            merged = r.sort_index().sort_index(axis=1)
        merged = merged.astype(FACTOR_DTYPE)
        _atomic_write(merged, path)
        logger.info(f"  [{n}d] {merged.shape} (+{len(merged.loc[merged.index > last])} 天) -> {path.name}")

    logger.info(f"    forward_return 耗时 {time.time()-t0:.1f}s")
    logger.info("=" * 72)
    logger.info("vwap_panel + labels 增量日更完成")
    logger.info("=" * 72)


if __name__ == "__main__":
    update()
