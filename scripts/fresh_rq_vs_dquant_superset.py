"""
fresh-vs-fresh 快验：现场重算 rq superset vs dquant 缓存
============================================================
背景：对账发现 rq 基线 superset 缓存陈旧（历史回填日残留错误 0 值），
拿它当基线对比 dquant 不公平。本脚本剔除陈旧缓存，做纯净对照：

  fresh_rq = 用当前 reducer 从 minute/raw(rq) + rq ex_cum_factor 现场重算的 superset
  dquant   = intermediate-cache-dquant/ 已有缓存（dquant raw + jy adjfactor 新算）

逐 superset 列在「样本股 × 窗口」上比较，量化「换数据源 + 换复权口径」的真实影响：
  - 成交量类列(peak_count/volume_sum/interval/eruption_count...)：复权无关 → 预期完全相等
  - 价格类列(daily_close/vwap/ridge_return...)：rq vs jy 复权差 → 预期 ~1e-6 噪声

用法（必须在 rq 后端下跑，即不要 export MINUTE_DATA_BACKEND=dquant）：
  python scripts/fresh_rq_vs_dquant_superset.py --n 50 --start 2024-01-01 --end 2026-06-12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from core.minute_data import load_adjusted_minute_window  # noqa: E402
from core.operators.minute_intraday_aggregate import (  # noqa: E402
    _SUPERSET_COLUMNS,
    _compute_one_stock,
)

DQ_CACHE = config._DATA_ROOT / "intermediate-cache-dquant" / "prv_v3__hc09d46528f"  # noqa: SLF001

# 成交量/计数类列：不受复权影响，预期与 dquant 完全相等
VOLUME_COLS = {
    "peak_count", "ridge_count", "valley_count",
    "peak_volume_sum", "ridge_volume_sum", "valley_volume_sum",
    "peak_turnover_sum", "ridge_turnover_sum", "valley_turnover_sum",
    "peak_interval_n", "peak_interval_m1", "peak_interval_m2",
    "peak_interval_m3", "peak_interval_m4",
    "ridge_interval_n", "ridge_interval_m1", "ridge_interval_m2",
    "ridge_interval_m3", "ridge_interval_m4",
    "eruption_count", "eruption_turnover_sum", "eruption_turnover_sumsq",
    "eruption_next_turnover_sum", "eruption_next_turnover_sumsq", "eruption_xy_sum",
    "peakridge_minute_corr_pooled", "daily_volume", "daily_turnover",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="抽样股票数")
    ap.add_argument("--start", default="2024-01-01", help="比较窗口起始")
    ap.add_argument("--end", default="2026-06-12", help="比较窗口结束")
    ap.add_argument("--lead-days", type=int, default=60, help="reducer 预热 lead-in 自然日")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if config.MINUTE_RAW_DIR.name != "raw" or "dquant" in str(config.MINUTE_RAW_DIR):
        logger.error(f"当前 MINUTE_RAW_DIR={config.MINUTE_RAW_DIR} 不是 rq 后端！请勿设 MINUTE_DATA_BACKEND=dquant")
        sys.exit(1)
    logger.info(f"rq raw     : {config.MINUTE_RAW_DIR}")
    logger.info(f"rq adjfac  : {config.MINUTE_EX_FACTORS_DIR}")
    logger.info(f"dquant 缓存: {DQ_CACHE}")

    # 抽样股票（含 002623 这类已知边界股）
    all_stocks = sorted(p.stem for p in DQ_CACHE.glob("*.parquet"))
    rng = np.random.default_rng(a.seed)
    samp = list(rng.choice(all_stocks, size=min(a.n, len(all_stocks)), replace=False))
    for must in ["002623.XSHE"]:
        if must in all_stocks and must not in samp:
            samp.append(must)
    logger.info(f"抽样 {len(samp)} 股 | 窗口 {a.start}~{a.end} | lead-in {a.lead_days}d")

    load_start = (pd.Timestamp(a.start) - pd.Timedelta(days=a.lead_days)).strftime("%Y-%m-%d")
    win_s, win_e = pd.Timestamp(a.start), pd.Timestamp(a.end)

    # 现场重算 fresh_rq superset（逐股）
    logger.info(f"读取 rq raw [{load_start} ~ {a.end}] 并现场重算 superset ...")
    raw_all = load_adjusted_minute_window(load_start, a.end, stocks=samp)
    if raw_all.empty:
        logger.error("rq raw 窗口为空"); sys.exit(1)
    logger.info(f"raw 行数={len(raw_all):,}，开始逐股 reduce ...")

    # 累计每列统计
    agg = {c: {"valued": 0, "diff": 0, "max_abs": 0.0, "max_rel": 0.0, "rels": []} for c in _SUPERSET_COLUMNS}
    for ob, g in raw_all.groupby("order_book_id", sort=False):
        fr = _compute_one_stock(ob, g, std_window=20, std_threshold=1.0)
        if fr.empty:
            continue
        fr["date"] = pd.to_datetime(fr["date"])
        fr = fr[(fr["date"] >= win_s) & (fr["date"] <= win_e)].set_index("date")
        dqp = DQ_CACHE / f"{ob}.parquet"
        if not dqp.exists():
            continue
        dq = pd.read_parquet(dqp)
        dq["date"] = pd.to_datetime(dq["date"])
        dq = dq[(dq["date"] >= win_s) & (dq["date"] <= win_e)].set_index("date")
        ci = fr.index.intersection(dq.index)
        if len(ci) == 0:
            continue
        for c in _SUPERSET_COLUMNS:
            A = fr.loc[ci, c].to_numpy(float)
            B = dq.loc[ci, c].to_numpy(float)
            bv = ~np.isnan(A) & ~np.isnan(B)
            if not bv.any():
                continue
            d = np.abs(A[bv] - B[bv])
            rel = d / np.maximum(np.abs(B[bv]), 1e-12)
            s = agg[c]
            s["valued"] += int(bv.sum())
            s["diff"] += int((d > 0).sum())
            s["max_abs"] = max(s["max_abs"], float(d.max()))
            s["max_rel"] = max(s["max_rel"], float(rel.max()))
            if (d > 0).any():
                s["rels"].append(rel[rel > 0])

    logger.info("-" * 110)
    logger.info(f"{'superset 列':<32}{'类别':>6}{'有值':>12}{'差异单元':>10}{'差异占比':>10}{'maxAbs':>11}{'maxRel':>11}")
    logger.info("-" * 110)
    vol_bad = price_bad = 0
    for c in _SUPERSET_COLUMNS:
        s = agg[c]
        if s["valued"] == 0:
            continue
        cls = "量" if c in VOLUME_COLS else "价"
        frac = s["diff"] / s["valued"]
        # 判据：量类列要求完全相等(diff=0)；价类列 maxRel<1e-3
        if cls == "量":
            ok = s["diff"] == 0
            if not ok:
                vol_bad += 1
        else:
            ok = s["max_rel"] < 1e-3
            if not ok:
                price_bad += 1
        flag = "✅" if ok else "❌"
        logger.info(
            f"{c:<32}{cls:>6}{s['valued']:>12,}{s['diff']:>10,}{frac:>10.2e}"
            f"{s['max_abs']:>11.2e}{s['max_rel']:>11.2e}  {flag}"
        )
    logger.info("-" * 110)
    logger.info(f"量类列异常(应完全相等却有差): {vol_bad} | 价类列异常(maxRel≥1e-3): {price_bad}")
    if vol_bad == 0 and price_bad == 0:
        logger.success("✅ fresh-rq 与 dquant：量类列逐值相等，价类列仅 <1e-3 复权噪声 → 迁移等价")
    else:
        logger.warning("存在异常列，见上方明细")


if __name__ == "__main__":
    main()
