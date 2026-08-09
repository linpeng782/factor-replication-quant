"""
alpha158 日频因子日更增量 append（生产）
==========================================
设计见 docs/alpha158_incremental_design.md；增量内核已由
alpha158/smoke_replay.py 真实数据对账验证（全 158 因子 rel<1e-6 + 新股吻合）。

算法（§6）：
  last = 现有 WIDE 面板的最大日期；T = 源数据(raw_ohlcv)最新日期；need = (last, T]
  W    = max(WINDOWS) = 60 → 回读窗口 = [last - (W+buffer)交易日, T]（覆盖最早新日的 warmup；
         单日日更 / 多日 catch-up 同一路径，窗口随 gap 自适应）
  ① glob 全市场按股 OHLCV → 读时复权 → pivot 临时宽表
  ② Alpha158Panel.compute_all → 158 宽表
  ③ 逐因子：读旧 WIDE → 切 (last,T] 新行 → concat(列并集自动纳新股) → dedup(keep last) → 原子写

零 API：只读已落盘的 raw_ohlcv/ex_factors（数据线负责拉取）。
基线目录后端感知 = config.ALPHA158_RAW_BASE（dquant→alpha158-dquant，rq→alpha158）；
首次须先有基线：若无对应面板，先跑 alpha158/build.py --full（同后端 env）。
"""
from __future__ import annotations

import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from alpha158.engine import Alpha158Panel
from alpha158.engine.groups import factor_group

RAW_DIR = config.RAW_OHLCV_DIR
EX_DIR = config.EX_FACTORS_DIR
ALPHA158_BASE = config.ALPHA158_RAW_BASE     # 后端感知：dquant→alpha158-dquant，rq→alpha158
WINDOWS = [5, 10, 20, 30, 60]
W = max(WINDOWS)
BUFFER = 10                     # warmup 安全余量（交易日）
N_READ_WORKERS = 32
N_WRITE_WORKERS = 16
FIELDS = ["open", "high", "low", "close", "volume", "vwap"]


# ==================== 源读取（读时复权：价×ffill(cum)，量÷cum）====================
def _load_one(stock: str, raw_dir: str, ex_dir: str, cutoff: str) -> pd.DataFrame | None:
    raw_path = Path(raw_dir) / f"{stock}.parquet"
    if not raw_path.exists():
        return None
    raw = pd.read_parquet(raw_path)
    raw.index = pd.to_datetime(raw.index)
    raw = raw[raw.index >= pd.Timestamp(cutoff)]      # 只读窗口尾段
    if raw.empty:
        return None
    ex_path = Path(ex_dir) / f"{stock}.parquet"
    if ex_path.exists():
        ex = pd.read_parquet(ex_path)
        cum = ex["ex_cum_factor"].dropna().reindex(raw.index, method="ffill").fillna(1.0)
    else:
        cum = pd.Series(1.0, index=raw.index)
    adj = raw[["open", "high", "low", "close", "volume", "total_turnover"]].copy()
    for c in ["open", "high", "low", "close"]:
        adj[c] = adj[c] * cum
    adj["volume"] = adj["volume"] / cum
    adj["vwap"] = adj["total_turnover"] / adj["volume"].replace(0, np.nan)
    adj["stock_code"] = stock
    return adj.reset_index().rename(columns={"date": "datetime"})


def load_source_panels(stocks: list[str], cutoff: str) -> dict[str, pd.DataFrame]:
    worker = partial(_load_one, raw_dir=str(RAW_DIR), ex_dir=str(EX_DIR), cutoff=cutoff)
    dfs = []
    with ProcessPoolExecutor(max_workers=N_READ_WORKERS) as ex:
        for fut in as_completed([ex.submit(worker, s) for s in stocks]):
            r = fut.result()
            if r is not None:
                dfs.append(r)
    long = pd.concat(dfs, ignore_index=True)
    panels = {}
    for f in FIELDS:
        w = long.pivot(index="datetime", columns="stock_code", values=f)
        panels[f] = w.sort_index().sort_index(axis=1)
    return panels


# ==================== 基线探查 ====================
def baseline_last_date() -> pd.Timestamp | None:
    """现有 WIDE 面板的最大日期（取各因子 max 的最小值，保守对齐）。无基线返回 None。"""
    fs = list(ALPHA158_BASE.glob("*/*.parquet"))
    if not fs:
        return None
    rng = np.random.default_rng(0)
    samp = list(rng.choice(fs, min(20, len(fs)), replace=False))
    maxes = [pd.to_datetime(pd.read_parquet(f, columns=[]).index).max() for f in samp]
    return min(maxes)


# ==================== 单因子 append（原子写）====================
def _append_one(name: str, new_wide: pd.DataFrame, last: pd.Timestamp) -> str:
    group = factor_group(name)
    path = ALPHA158_BASE / group / f"{name}.parquet"
    new_rows = new_wide.loc[new_wide.index > last]
    if new_rows.empty:
        return "empty"
    if path.exists():
        old = pd.read_parquet(path)
        old.index = pd.to_datetime(old.index)
        combined = pd.concat([old, new_rows])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:                                  # 该因子尚无基线 → 直接落新行（覆盖度随后续日更补齐）
        combined = new_rows.sort_index()
    combined = combined.sort_index(axis=1).astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp)
    os.replace(tmp, path)                  # 原子写
    return "ok"


def main():
    t0 = time.time()

    last = baseline_last_date()
    if last is None:
        logger.error(f"无基线面板 {ALPHA158_BASE}/；请先跑 alpha158/build.py --full 建全量基线")
        sys.exit(1)
    logger.info(f"基线末日 last = {last.date()}")

    # 回读窗口起点：last 往前 (W+BUFFER) 交易日（用 1.6× 日历换算保守覆盖）
    cutoff = (last - pd.Timedelta(days=math.ceil((W + BUFFER) * 1.6))).date().isoformat()
    logger.info(f"回读窗口起点 cutoff = {cutoff}（W={W} + buffer={BUFFER} 交易日 warmup）")

    stocks = [f.stem for f in sorted(RAW_DIR.glob("*.parquet"))]
    logger.info(f"读源面板：{len(stocks)} 只股票，窗口 [{cutoff}, 最新]...")
    panels = load_source_panels(stocks, cutoff)
    T = panels["close"].index.max()
    n_new = int((panels["close"].index > last).sum())
    logger.info(f"  源末日 T = {T.date()} | need=(last, T] = {n_new} 个新交易日")
    if n_new == 0:
        logger.success(f"已是最新（基线已到 {last.date()}），无需更新。耗时 {time.time()-t0:.0f}s")
        return

    logger.info("计算 158 因子...")
    factors = Alpha158Panel(panels).compute_all(windows=WINDOWS)

    logger.info(f"逐因子 append + 原子写（{N_WRITE_WORKERS} 线程）...")
    stats = {"ok": 0, "empty": 0}
    with ThreadPoolExecutor(max_workers=N_WRITE_WORKERS) as ex:
        futs = {ex.submit(_append_one, n, w, last): n for n, w in factors.items()}
        for fut in as_completed(futs):
            stats[fut.result()] = stats.get(fut.result(), 0) + 1

    logger.success(
        f"✅ alpha158 日更完成：{stats['ok']} 因子 append {last.date()} → {T.date()}"
        f"（{n_new} 新日）| empty={stats.get('empty',0)} | 耗时 {time.time()-t0:.0f}s"
    )


if __name__ == "__main__":
    main()
