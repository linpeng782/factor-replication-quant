"""
预计算 Ret20 后复权面板（APM 截面回归去动量用）
============================================================
从 stock-ohlcv（原始不复权）× stock-ex-factors（ex_cum_factor）→
后复权 close → 20 日滚动收益率宽表。

输出: factors/helpers/ret20_panel.parquet（date × order_book_id, float32）
  值 = close_t_post / close_{t-20}_post - 1

用法:
  PYTHONPATH=. python scripts/build_ret20_panel.py
  PYTHONPATH=. python scripts/build_ret20_panel.py --workers 64  # 并行加速
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from core.config import EX_FACTORS_DIR, RAW_OHLCV_DIR, RET20_PANEL_PATH

WINDOW = 20   # 20 日滚动收益
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
        cf = ex["ex_cum_factor"].reindex(close_raw.index, method="ffill").fillna(1.0)
    else:
        cf = pd.Series(1.0, index=close_raw.index)

    close_post = (close_raw * cf).astype("float64")
    return stock, close_post


def build(workers: int = 32) -> None:
    out_path = RET20_PANEL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stocks = sorted(p.stem for p in RAW_OHLCV_DIR.glob("*.parquet"))
    logger.info(f"构建 Ret20 面板: {len(stocks)} 只股票, workers={workers}")

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

    # 20 日滚动收益 = close_t / close_{t-20} - 1
    ret20 = (wide_close / wide_close.shift(WINDOW) - 1.0).astype(DTYPE)
    ret20.index.name = "date"
    ret20.columns.name = "order_book_id"

    ret20.to_parquet(out_path)
    size_mb = out_path.stat().st_size / 1024**2
    logger.success(
        f"Ret20 面板已保存: {out_path}\n"
        f"  shape={ret20.shape} | range={ret20.index.min().date()}~{ret20.index.max().date()} "
        f"| NaN={ret20.isna().values.mean():.1%} | {size_mb:.1f}MB"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="预计算 Ret20 后复权面板")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    build(workers=a.workers)
