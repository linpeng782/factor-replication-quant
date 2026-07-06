"""
全量构建：Forward Return Labels（VWAP 口径，dquant 后端）
============================================================
从 my-alpha-engine/alpha_engine/labels/build.py 移植，数据源改为 dquant raw OHLCV。

职责：
  - 加载后复权 vwap 面板（vwap = total_turnover / volume，已对齐后复权口径）
  - 把 vwap 宽表副产到 config.VWAP_PANEL_PATH（PIT canonical 源；评估 fallback 也读它）
  - 用 alpha_shared.evaluation.returns.build_forward_returns 计算多 horizon 未来收益
  - 每个 horizon 一个宽表 parquet，覆盖保存到 config.LABELS_DIR

收益定义（VWAP 口径，避免未来函数；更接近实盘 TWAP/VWAP 算法单可执行价）：
    return_N[T] = (vwap[T+1+N] - vwap[T+1]) / vwap[T+1]
    T 日信号 → T+1 vwap 买 → T+1+N vwap 卖

产物：
    market-data/labels/vwap_panel.parquet              （后复权 vwap 宽表，评估 fallback 用）
    market-data/labels/forward_return_{N}d.parquet     （float32 宽表）

用法：python data_fetching/build_labels.py            # 全量：2005-01-01 ~ 最新交易日
      python data_fetching/build_labels.py --start 20200101 --end 20261231
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.data.adjusted_panels import load_adjusted_panels
from alpha_shared.evaluation.returns import build_forward_returns

FACTOR_DTYPE = np.float32  # 与 my-alpha-engine config.FACTOR_DTYPE 一致


def build_labels(start: str, end: str, horizons=(1, 2, 5, 10, 20)):
    """全量构建 forward return labels（VWAP 口径）。

    参数：
        start, end : 时间区间（YYYY-MM-DD）
        horizons   : 持有期列表（天）

    产物：
        VWAP_PANEL_PATH / vwap_panel.parquet （float32 宽表）
        LABELS_DIR / forward_return_{N}d.parquet （float32 宽表）

    注意：
        为保证 T+1+N <= end，实际上最后 N+1 日的 label 会是 NaN。
        这是正确行为（避免未来函数），训练时需要丢弃这些尾部行。
    """
    logger.info("=" * 72)
    logger.info(f"全量 Forward Return Label 构建开始（VWAP 口径，dquant 后端）：{start} ~ {end}")
    logger.info(f"  horizons = {horizons}")
    logger.info("=" * 72)

    # ========== 1. 加载后复权 vwap 面板 ==========
    t0 = time.time()
    logger.info("[1/3] 加载后复权 vwap 面板（虚拟字段，自动由 total_turnover / volume 计算）")
    panels = load_adjusted_panels(start=start, end=end, fields=("vwap",))
    vwap_p = panels["vwap"]
    t_load = time.time() - t0
    logger.info(
        f"    vwap shape={vwap_p.shape}, "
        f"时间={vwap_p.index.min().date()} ~ {vwap_p.index.max().date()}, "
        f"非空比例={vwap_p.notna().values.mean():.2%}"
    )
    logger.info(f"    加载耗时 {t_load:.1f}s")

    # ========== 1.5. 副产 vwap_panel.parquet ==========
    config.VWAP_PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    vwap_p.astype(FACTOR_DTYPE).to_parquet(config.VWAP_PANEL_PATH)
    logger.info(f"    -> vwap 宽表副产: {config.VWAP_PANEL_PATH}")

    # ========== 2. 批量计算 forward_return ==========
    t0 = time.time()
    logger.info(f"[2/3] 计算 forward_return (horizons={horizons})")
    returns_dict = build_forward_returns(vwap_p, horizons=horizons)
    t_compute = time.time() - t0
    logger.info(f"    计算耗时 {t_compute:.1f}s")

    # ========== 3. 逐 horizon 覆盖保存 ==========
    t0 = time.time()
    logger.info("[3/3] 逐 horizon 保存到 labels/")
    config.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for n, r in returns_dict.items():
        path = config.LABELS_DIR / f"forward_return_{n}d.parquet"
        r_out = r.astype(FACTOR_DTYPE)
        r_out.to_parquet(path)
        logger.info(
            f"  [{n}d] shape={r_out.shape}, "
            f"非空比例={r_out.notna().values.mean():.2%} "
            f"-> {path.name}"
        )
    t_save = time.time() - t0
    logger.info(f"    保存耗时 {t_save:.1f}s")

    logger.info("=" * 72)
    logger.info(
        f"Label 构建完成: 加载 {t_load:.1f}s + 计算 {t_compute:.1f}s + 保存 {t_save:.1f}s"
    )
    logger.info(f"  产物目录: {config.LABELS_DIR}")
    logger.info("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全量构建 forward return labels")
    parser.add_argument("--start", default="2005-01-01", help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认最新交易日）")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    args = parser.parse_args()

    # end 默认取 raw OHLCV 最新日期
    if args.end is None:
        from data_fetching.dquant_source import latest_trading_date
        args.end = str(latest_trading_date())  # 返回 'YYYY-MM-DD' 字符串

    build_labels(start=args.start, end=args.end, horizons=tuple(args.horizons))
