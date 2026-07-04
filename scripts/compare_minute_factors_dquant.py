"""
paper_27 分钟因子对账：dquant 后端 vs rq 基线（factors/raw-dquant vs factors/raw）
============================================================
方案 A 收尾验收：逐因子比较 dquant 后端产出的因子面板与 rq 基线面板，
量化"换数据源 + 换复权口径(jy)"对最终因子值的影响（预期 ~float32 噪声）。

对每个 <group> 下的因子（默认 kysec/paper_27_microstructure）：
  rq = factors/raw/<ns>/<f>.parquet     （rqdatac 源 + rq ex_cum_factor）
  dq = factors/raw-dquant/<ns>/<f>.parquet（dquant 源 + jy adjfactor）
  在共同日 × 共同股上比较：NaN 位置一致性 + 有值单元 maxAbs/maxRel/中位Rel。

用法：
  python scripts/compare_minute_factors_dquant.py
  python scripts/compare_minute_factors_dquant.py --namespace kysec/paper_27_microstructure
  python scripts/compare_minute_factors_dquant.py --start 2016-01-01 --end 2026-06-12
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

# config.RAW_FACTOR_BASE 受 MINUTE_DATA_BACKEND 影响；这里显式取两个固定基座，避免歧义。
_FACTORS = config._FACTORS  # noqa: SLF001
RQ_BASE = _FACTORS / "raw"
DQ_BASE = _FACTORS / "raw-dquant"

PASS_PRICE_REL = 1e-4   # 价格/比率类因子的相对容差判据（float32 + 复权口径差）


def compare_factor(ns: str, name: str, start, end) -> dict:
    rq_p = RQ_BASE / ns / f"{name}.parquet"
    dq_p = DQ_BASE / ns / f"{name}.parquet"
    res = {"factor": name, "ok": False, "note": ""}
    if not rq_p.exists():
        res["note"] = "rq 基线缺"; return res
    if not dq_p.exists():
        res["note"] = "dquant 缺"; return res
    rq = pd.read_parquet(rq_p); dq = pd.read_parquet(dq_p)
    rq.index = pd.to_datetime(rq.index); dq.index = pd.to_datetime(dq.index)

    cd = rq.index.intersection(dq.index)
    if start is not None:
        cd = cd[(cd >= pd.Timestamp(start)) & (cd <= pd.Timestamp(end))]
    cc = rq.columns.intersection(dq.columns)
    if len(cd) == 0 or len(cc) == 0:
        res["note"] = "无共同日/股"; return res

    A = rq.loc[cd, cc].to_numpy(float)
    B = dq.loc[cd, cc].to_numpy(float)
    res["days"], res["stocks"] = len(cd), len(cc)
    res["nan_mismatch"] = int((np.isnan(A) ^ np.isnan(B)).sum())
    bv = ~np.isnan(A) & ~np.isnan(B)
    res["valued"] = int(bv.sum())
    if res["valued"] == 0:
        res["note"] = "无共同有值单元"; return res
    d = np.abs(A[bv] - B[bv])
    rel = d / np.maximum(np.abs(B[bv]), 1e-9)
    res["max_abs"] = float(d.max())
    res["max_rel"] = float(rel.max())
    res["med_rel"] = float(np.median(rel))
    res["exact"] = bool(np.array_equal(A[bv], B[bv]))
    # NaN 位置占比（相对共同单元）；判据：位置一致 + 有值单元 maxRel 在容差内
    nan_ratio = res["nan_mismatch"] / max(A.size, 1)
    res["ok"] = (nan_ratio < 1e-3) and (res["max_rel"] < PASS_PRICE_REL)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="paper_27 分钟因子 dquant vs rq 对账")
    ap.add_argument("--namespace", default="kysec/paper_27_microstructure")
    ap.add_argument("--start", default=None, help="对比起始日 YYYY-MM-DD（默认全 overlap）")
    ap.add_argument("--end", default="2026-06-12")
    a = ap.parse_args()

    dq_dir = DQ_BASE / a.namespace
    if not dq_dir.exists():
        logger.error(f"dquant 因子目录不存在: {dq_dir}（先跑 MINUTE_DATA_BACKEND=dquant 产因子）")
        sys.exit(1)
    names = sorted(p.stem for p in dq_dir.glob("*.parquet"))
    logger.info(f"对账 {len(names)} 个因子 | ns={a.namespace} | 窗口 {a.start or 'full'}~{a.end}")
    logger.info(f"  rq 基线 : {RQ_BASE / a.namespace}")
    logger.info(f"  dquant  : {dq_dir}")
    logger.info("-" * 96)
    logger.info(f"{'因子':<34}{'天':>5}{'股':>6}{'有值':>11}{'NaN不一致':>10}{'maxRel':>10}{'中位Rel':>10}  判定")

    rows = []
    for n in names:
        r = compare_factor(a.namespace, n, a.start, a.end)
        rows.append(r)
        if "max_rel" not in r:
            logger.info(f"{n:<34}{'':>5}{'':>6}{'':>11}{'':>10}{'':>10}{'':>10}  ⚠️ {r['note']}")
        else:
            flag = "✅" if r["ok"] else "❌"
            logger.info(
                f"{n:<34}{r['days']:>5}{r['stocks']:>6}{r['valued']:>11,}"
                f"{r['nan_mismatch']:>10,}{r['max_rel']:>10.2e}{r['med_rel']:>10.2e}  {flag}"
            )

    evaluated = [r for r in rows if "max_rel" in r]
    n_ok = sum(r["ok"] for r in evaluated)
    logger.info("-" * 96)
    logger.info(f"汇总: {n_ok}/{len(evaluated)} 因子在容差内对齐（共 {len(names)} 因子）")
    if evaluated:
        worst = max(evaluated, key=lambda r: r["max_rel"])
        logger.info(f"最大偏差因子: {worst['factor']} maxRel={worst['max_rel']:.2e}")
    if n_ok != len(evaluated) or not evaluated:
        logger.warning("存在未通过/无法评估因子，请检查上方明细")
        sys.exit(1)
    logger.success("✅ paper_27 全部因子 dquant 后端与 rq 基线对齐（≤ float32/复权口径噪声）")


if __name__ == "__main__":
    main()
