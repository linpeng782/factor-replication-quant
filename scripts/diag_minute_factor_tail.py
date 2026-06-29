"""
paper_27 分钟因子 dquant vs rq 长尾诊断
============================================================
对账主脚本(compare_minute_factors_dquant.py)用连续型相对容差判 ❌，
但 paper_27 多为「峰/脊阈值检测」离散因子：JY vs RQ 复权口径的微小差
偶尔翻转某分钟的峰/脊判定 → 个别单元差异巨大(maxRel 很大)，
而绝大多数单元 bit 相同(median rel=0)。

本脚本量化「到底有多少单元真的不同」，回答：差异是长尾噪声还是系统性偏差。
对每个因子统计：
  - 总有值单元 / NaN 位置不一致单元
  - |Δ|>0 的单元数及占比(完全相同 vs 有任何差异)
  - 按相对误差分桶(>1e-6 / >1e-4 / >1e-2 / >1e-1)的单元数
  - 差异单元涉及的「日数」「股数」(是否集中)

用法:
  python scripts/diag_minute_factor_tail.py
  python scripts/diag_minute_factor_tail.py --factor peak_minute_count
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config  # noqa: E402

_FACTORS = config._FACTORS  # noqa: SLF001
RQ_BASE = _FACTORS / "raw"
DQ_BASE = _FACTORS / "raw-dquant"


def diag_factor(ns: str, name: str) -> dict:
    rq = pd.read_parquet(RQ_BASE / ns / f"{name}.parquet")
    dq = pd.read_parquet(DQ_BASE / ns / f"{name}.parquet")
    rq.index = pd.to_datetime(rq.index)
    dq.index = pd.to_datetime(dq.index)
    cd = rq.index.intersection(dq.index)
    cc = rq.columns.intersection(dq.columns)
    A = rq.loc[cd, cc].to_numpy(float)
    B = dq.loc[cd, cc].to_numpy(float)

    nan_mis = np.isnan(A) ^ np.isnan(B)
    bv = ~np.isnan(A) & ~np.isnan(B)
    d = np.abs(A[bv] - B[bv])
    rel = d / np.maximum(np.abs(B[bv]), 1e-12)

    # 有值单元里「完全相同」与「有任何差异」
    n_valued = int(bv.sum())
    n_diff = int((d > 0).sum())
    # 差异单元在原矩阵中的行列位置(用于统计涉及的日/股)
    diff_mask_full = np.zeros_like(A, dtype=bool)
    diff_mask_full[bv] = d > 0
    diff_mask_full |= nan_mis
    diff_rows = np.where(diff_mask_full.any(axis=1))[0]
    diff_cols = np.where(diff_mask_full.any(axis=0))[0]

    return {
        "factor": name,
        "valued": n_valued,
        "nan_mismatch": int(nan_mis.sum()),
        "exact_cells": n_valued - n_diff,
        "diff_cells": n_diff,
        "diff_frac": n_diff / max(n_valued, 1),
        "gt_1em6": int((rel > 1e-6).sum()),
        "gt_1em4": int((rel > 1e-4).sum()),
        "gt_1em2": int((rel > 1e-2).sum()),
        "gt_1em1": int((rel > 1e-1).sum()),
        "diff_days": len(diff_rows),
        "diff_stocks": len(diff_cols),
        "total_days": len(cd),
        "total_stocks": len(cc),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default="kysec/paper_27_microstructure")
    ap.add_argument("--factor", default=None, help="只看单个因子；默认全部")
    a = ap.parse_args()

    dq_dir = DQ_BASE / a.namespace
    names = ([a.factor] if a.factor else sorted(p.stem for p in dq_dir.glob("*.parquet")))
    logger.info(f"长尾诊断 {len(names)} 因子 | ns={a.namespace}")
    logger.info("-" * 118)
    logger.info(
        f"{'因子':<34}{'有值':>12}{'差异单元':>10}{'差异占比':>10}"
        f"{'>1e-6':>8}{'>1e-4':>8}{'>1e-2':>8}{'>1e-1':>8}{'涉及日':>7}{'涉及股':>7}"
    )
    for n in names:
        r = diag_factor(a.namespace, n)
        logger.info(
            f"{n:<34}{r['valued']:>12,}{r['diff_cells']:>10,}{r['diff_frac']:>10.2e}"
            f"{r['gt_1em6']:>8,}{r['gt_1em4']:>8,}{r['gt_1em2']:>8,}{r['gt_1em1']:>8,}"
            f"{r['diff_days']:>7}{r['diff_stocks']:>7}"
        )
    logger.info("-" * 118)


if __name__ == "__main__":
    main()
