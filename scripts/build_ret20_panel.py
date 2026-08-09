"""
预计算 RetN 后复权收益面板
============================================================
从 stock-ohlcv（原始不复权）× stock-ex-factors（ex_cum_factor）→
后复权 close → N 日滚动收益率宽表。

输出（date × order_book_id, float32），值 = close_t_post / close_{t-N}_post - 1：
  --window 20（默认）→ factors/helpers/ret20_panel.parquet  APM 截面回归去动量用
  --window 1          → factors/helpers/ret1_panel.parquet   改进动量因子（wgt_return 系）收益源

用法:
  PYTHONPATH=. python scripts/build_ret20_panel.py
  PYTHONPATH=. python scripts/build_ret20_panel.py --window 1 --workers 64
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config import EX_FACTORS_DIR, RAW_OHLCV_DIR, RET1_PANEL_PATH, RET20_PANEL_PATH

# 窗口 → 输出面板路径（每个窗口一份独立 helpers 面板）
WINDOW_TO_PATH = {1: RET1_PANEL_PATH, 20: RET20_PANEL_PATH}
DTYPE  = "float32"


def _post_close_series(stock: str) -> tuple[str, pd.Series | None]:
    """读单股原始 close + ex_cum_factor → 后复权 close 序列（date 索引）。"""
    ohlcv_path = RAW_OHLCV_DIR / f"{stock}.parquet"
    if not ohlcv_path.exists():
        return stock, None
    raw = pd.read_parquet(ohlcv_path, columns=["close"])
    raw.index = pd.to_datetime(raw.index)
    close_raw = raw["close"]

    ex_path = EX_FACTORS_DIR / f"{stock}.parquet"
    if ex_path.exists():
        ex = pd.read_parquet(ex_path, columns=["ex_cum_factor"])
        ex.index = pd.to_datetime(ex.index)
        cf = ex["ex_cum_factor"].dropna().reindex(close_raw.index, method="ffill").fillna(1.0)
    else:
        cf = pd.Series(1.0, index=close_raw.index)

    close_post = (close_raw * cf).astype("float64")
    return stock, close_post


def build(workers: int = 32, window: int = 20) -> None:
    if window not in WINDOW_TO_PATH:
        raise ValueError(f"window={window} 无对应输出路径；可选 {sorted(WINDOW_TO_PATH)}")
    out_path = WINDOW_TO_PATH[window]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stocks = sorted(p.stem for p in RAW_OHLCV_DIR.glob("*.parquet"))
    logger.info(f"构建 Ret{window} 面板: {len(stocks)} 只股票, workers={workers}")

    all_series: dict[str, pd.Series] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_post_close_series, s): s for s in stocks}
        for i, fut in enumerate(as_completed(futs), 1):
            stock, s = fut.result()
            if s is not None:
                all_series[stock] = s
            if i % 1000 == 0:
                logger.info(f"  进度: {i}/{len(stocks)}")

    logger.info(f"合并 {len(all_series)} 只股票 close 序列...")
    wide_close = pd.DataFrame(all_series).sort_index()  # date × stock

    # N 日滚动收益 = close_t / close_{t-N} - 1
    ret = (wide_close / wide_close.shift(window) - 1.0).astype(DTYPE)
    ret.index.name = "date"
    ret.columns.name = "order_book_id"

    ret.to_parquet(out_path)
    size_mb = out_path.stat().st_size / 1024**2
    logger.success(
        f"Ret{window} 面板已保存: {out_path}\n"
        f"  shape={ret.shape} | range={ret.index.min().date()}~{ret.index.max().date()} "
        f"| NaN={ret.isna().values.mean():.1%} | {size_mb:.1f}MB"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="预计算 RetN 后复权收益面板")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--window", type=int, default=20, help="滚动窗口（1 或 20）")
    a = ap.parse_args()
    build(workers=a.workers, window=a.window)
