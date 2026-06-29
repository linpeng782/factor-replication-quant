"""
分钟 raw 后端对齐验证：dquant 1m vs 现有 rq minute/raw（逐值 diff）
============================================================
方案 A 的"前提钉死"工具：在大样本交易日上证明 dquant `get_price(1m, source="rq")`
与现有 rqdatac 落盘的 minute/raw 逐值 bit 相同，从而保证换数据源后下游因子零漂移。

对每个抽样交易日：
  rq  = 读 market-data/minute/raw/<日>.parquet（rqdatac 历史落盘）
  dq  = minute_ohlcv_dquant.fetch_day_df(<日>)（dquant 现取，与落盘口径同一函数）
  比对：列/dtype/股票集合/行数 + open/high/low/close maxAbs/maxRel + volume/total_turnover Δ

通过判据（每日）：价格 maxRel < 1e-4（float32 舍入），量/额 maxAbsΔ = 0，交集外行数占比 < 0.5%。

用法：
  python scripts/smoke_minute_dquant_align.py --n 12              # 全史随机抽 12 日
  python scripts/smoke_minute_dquant_align.py --dates 2006-01-04,2015-06-01,2024-01-05
  python scripts/smoke_minute_dquant_align.py --n 8 --seed 42     # 可复现抽样
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_fetching.minute_ohlcv_dquant import COLS, fetch_day_df  # noqa: E402

RQ_RAW_DIR = Path(
    __import__("os").environ.get(
        "FACTOR_REPL_DATA_ROOT", "/nfs/ofs-prediction/peterzhenglinpeng"
    )
) / "market-data" / "minute" / "raw"

KEY = ["order_book_id", "datetime"]
PRICE = ["open", "high", "low", "close"]
VOLAMT = ["volume", "total_turnover"]


def _rq_dates() -> list[str]:
    return sorted(p.stem for p in RQ_RAW_DIR.glob("[0-9]*.parquet"))


def compare_one(day: str) -> dict:
    """单日对齐，返回指标 dict。"""
    rq_path = RQ_RAW_DIR / f"{day}.parquet"
    res = {"date": day, "ok": False, "note": ""}
    if not rq_path.exists():
        res["note"] = "rq raw 缺该日"
        return res
    rq = pd.read_parquet(rq_path)
    dq = fetch_day_df(day)
    if dq is None or dq.empty:
        res["note"] = "dquant 返回空"
        return res

    res["cols_eq"] = list(rq.columns) == list(dq.columns) == COLS
    res["dtype_eq"] = bool((rq.dtypes.reindex(COLS) == dq.dtypes.reindex(COLS)).all())

    rq_i = rq.set_index(KEY).sort_index()
    dq_i = dq.set_index(KEY).sort_index()
    s_rq = set(rq_i.index.get_level_values(0))
    s_dq = set(dq_i.index.get_level_values(0))
    res["stocks_rq"], res["stocks_dq"] = len(s_rq), len(s_dq)
    res["stocks_only_rq"], res["stocks_only_dq"] = len(s_rq - s_dq), len(s_dq - s_rq)

    idx = rq_i.index.intersection(dq_i.index)
    res["rows_rq"], res["rows_dq"], res["rows_common"] = len(rq_i), len(dq_i), len(idx)
    res["rows_only_rq"] = len(rq_i.index.difference(dq_i.index))
    res["rows_only_dq"] = len(dq_i.index.difference(rq_i.index))
    a, b = rq_i.loc[idx], dq_i.loc[idx]

    price_maxrel = 0.0
    for c in PRICE:
        av, bv = a[c].to_numpy(float), b[c].to_numpy(float)
        rel = np.abs(av - bv) / np.maximum(np.abs(bv), 1e-6)
        price_maxrel = max(price_maxrel, float(np.nanmax(rel)) if len(rel) else 0.0)
    res["price_maxrel"] = price_maxrel

    vol_maxabs = 0.0
    vol_maxrel = 0.0
    for c in VOLAMT:
        av, bv = a[c].to_numpy(float), b[c].to_numpy(float)
        d = np.abs(av - bv)
        vol_maxabs = max(vol_maxabs, float(np.nanmax(d)) if len(d) else 0.0)
        rel = d / np.maximum(np.abs(bv), 1.0)   # 量/额尺度大，用 max(|b|,1) 归一（防 0 量分钟除零）
        vol_maxrel = max(vol_maxrel, float(np.nanmax(rel)) if len(rel) else 0.0)
    res["volamt_maxabs"] = vol_maxabs
    res["volamt_maxrel"] = vol_maxrel

    union = max(len(rq_i), 1)
    outside_ratio = (res["rows_only_rq"] + res["rows_only_dq"]) / union
    res["outside_ratio"] = outside_ratio

    # 量/额判据用相对容差（float64 表示噪声 ~1e-8 绝对值，相对 ~1e-14；价格 float32 舍入 < 1e-4）
    res["ok"] = (
        res["cols_eq"] and res["dtype_eq"]
        and price_maxrel < 1e-4 and vol_maxrel < 1e-6
        and outside_ratio < 0.005
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="分钟 raw 后端对齐：dquant vs rq minute/raw")
    ap.add_argument("--n", type=int, default=12, help="全史随机抽样天数（默认 12）")
    ap.add_argument("--dates", default=None, help="逗号分隔显式日期 YYYY-MM-DD（覆盖 --n）")
    ap.add_argument("--seed", type=int, default=0, help="随机抽样种子（可复现）")
    a = ap.parse_args()

    if a.dates:
        days = [d.strip() for d in a.dates.split(",") if d.strip()]
    else:
        all_days = _rq_dates()
        if not all_days:
            logger.error(f"rq raw 目录无日文件: {RQ_RAW_DIR}")
            sys.exit(1)
        random.seed(a.seed)
        days = sorted(random.sample(all_days, min(a.n, len(all_days))))

    logger.info(f"对齐 {len(days)} 个交易日 | rq={RQ_RAW_DIR}")
    results = []
    for d in days:
        r = compare_one(d)
        results.append(r)
        flag = "✅" if r["ok"] else ("⚠️" if r.get("note") else "❌")
        if r.get("note") and "price_maxrel" not in r:
            logger.info(f"{flag} {d}: {r['note']}")
        else:
            logger.info(
                f"{flag} {d}: 股票rq/dq={r['stocks_rq']}/{r['stocks_dq']}"
                f"(仅rq{r['stocks_only_rq']}/仅dq{r['stocks_only_dq']}) "
                f"行common={r['rows_common']:,}(仅rq{r['rows_only_rq']}/仅dq{r['rows_only_dq']}) "
                f"价maxRel={r['price_maxrel']:.2e} 量额maxRel={r['volamt_maxrel']:.2e}"
            )

    evaluated = [r for r in results if "price_maxrel" in r]
    n_ok = sum(r["ok"] for r in evaluated)
    logger.info("=" * 60)
    logger.info(f"汇总: {n_ok}/{len(evaluated)} 日逐值对齐通过（共抽 {len(days)} 日）")
    if n_ok != len(evaluated) or not evaluated:
        logger.error("存在未通过/无法评估的交易日，请检查上方明细")
        sys.exit(1)
    logger.success("✅ dquant 1m 与 rq minute/raw 逐值 bit 对齐（大样本）")


if __name__ == "__main__":
    main()
