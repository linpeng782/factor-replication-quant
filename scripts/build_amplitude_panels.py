"""
预计算日振幅面板（长端动量因子的切割指标）
============================================================
从 stock-ohlcv（原始不复权）× stock-ex-factors（ex_cum_factor）→ 后复权 high/low/close
→ 两种振幅定义的宽表面板（date × order_book_id, float32）：

  AMP_PANEL_PATH     amp    = (high − low) / close_{t-1}   长端动量 2.0 口径（研报表2 步骤3）
  AMP_HL_PANEL_PATH  amp_hl = high / low − 1               长端动量 1.0 口径（研报表1 步骤2）

复权说明：振幅是比率，同日 high/low 同乘 cum_factor 后 amp_hl 不变；而 amp 的分母是
前一日收盘，跨除权日必须用**后复权**价才不失真 → 统一用后复权价计算。

用法:
  PYTHONPATH=. python scripts/build_amplitude_panels.py
  PYTHONPATH=. python scripts/build_amplitude_panels.py --workers 64
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from loguru import logger

from config import AMP_HL_PANEL_PATH, AMP_PANEL_PATH, EX_FACTORS_DIR, RAW_OHLCV_DIR

DTYPE = "float32"


def _amp_series(stock: str) -> tuple[str, pd.DataFrame | None]:
    """读单股 high/low/close + ex_cum_factor → 两种振幅序列（date 索引）。"""
    ohlcv_path = RAW_OHLCV_DIR / f"{stock}.parquet"
    if not ohlcv_path.exists():
        return stock, None
    raw = pd.read_parquet(ohlcv_path, columns=["high", "low", "close"])
    raw.index = pd.to_datetime(raw.index)

    ex_path = EX_FACTORS_DIR / f"{stock}.parquet"
    if ex_path.exists():
        ex = pd.read_parquet(ex_path, columns=["ex_cum_factor"])
        ex.index = pd.to_datetime(ex.index)
        cf = ex["ex_cum_factor"].dropna().reindex(raw.index, method="ffill").fillna(1.0)
    else:
        cf = pd.Series(1.0, index=raw.index)

    post = raw.mul(cf, axis=0).astype("float64")
    out = pd.DataFrame({
        "amp": (post["high"] - post["low"]) / post["close"].shift(1),
        "amp_hl": post["high"] / post["low"] - 1.0,
    })
    return stock, out


def build(workers: int = 32) -> None:
    stocks = sorted(p.stem for p in RAW_OHLCV_DIR.glob("*.parquet"))
    logger.info(f"构建振幅面板: {len(stocks)} 只股票, workers={workers}")

    amp: dict[str, pd.Series] = {}
    amp_hl: dict[str, pd.Series] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_amp_series, s): s for s in stocks}
        for i, fut in enumerate(as_completed(futs), 1):
            stock, df = fut.result()
            if df is not None:
                amp[stock] = df["amp"]
                amp_hl[stock] = df["amp_hl"]
            if i % 1000 == 0:
                logger.info(f"  进度: {i}/{len(stocks)}")

    for name, series_map, out_path in (
        ("amp（(H−L)/前收）", amp, AMP_PANEL_PATH),
        ("amp_hl（H/L−1）", amp_hl, AMP_HL_PANEL_PATH),
    ):
        panel = pd.DataFrame(series_map).sort_index().astype(DTYPE)
        panel.index.name = "date"
        panel.columns.name = "order_book_id"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(out_path)
        logger.success(
            f"{name} 面板已保存: {out_path}\n"
            f"  shape={panel.shape} | range={panel.index.min().date()}~{panel.index.max().date()} "
            f"| NaN={panel.isna().values.mean():.1%} | {out_path.stat().st_size / 1024**2:.1f}MB"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="预计算日振幅面板（两种定义）")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    build(workers=a.workers)
