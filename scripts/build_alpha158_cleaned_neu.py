"""
Alpha158 clean→neu 批量生产 + cleaned vs neu 指标对比（fork 并行）
============================================================
读 factors/raw/alpha158/<group>/ → 清洗(MAD+zscore+mask) → 行业市值中性化
  → 落 factors/{cleaned,neu}/alpha158/<group>/；对 cleaned/neu 两版各算 IC/ICIR/单调/多空Sharpe。

并行：父进程加载共享数据(masks/行业/市值/labels)一次 → fork 出 N 个 worker（COW 只读共享、不复制内存）
      → 每 worker 处理若干因子。BLAS 线程限 1 防超额订阅。16GB Mac 默认 8 worker 安全。

注：masks 当前滞后(combo→5-27, new_stock→4-30)，仅影响 5 月尾段(无 label、评估窗外)；masks 更新后重跑即可。

用法：python scripts/build_alpha158_cleaned_neu.py               # 全部 158，8 worker
      python scripts/build_alpha158_cleaned_neu.py --workers 6
      python scripts/build_alpha158_cleaned_neu.py --limit 8     # 前 8 个（计时）
"""
from __future__ import annotations

import os
# BLAS 单线程：必须在 numpy 之前设；防 8 进程 × 多线程 BLAS 把机器拖死
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import config
from alpha_shared.cleaning.mask_loader import load_filter_masks
from alpha_shared.cleaning.preprocess import prepare_factor
from alpha_shared.neu import neutralize
from alpha_shared.evaluation.ic import compute_ic_series, compute_ic_report
from alpha_shared.evaluation.layered import layered_backtest

START, END = "2016-01-01", "2025-12-31"
IC_HORIZONS = (5, 10, 20)
PRIMARY = 5
LAYER_G, LAYER_N = 5, 5
RAW_BASE = config.RAW_FACTOR_BASE / "alpha158"
MAD_N = 3.0

_SHARED: dict = {}   # 父进程填充 → fork 后 worker 经 COW 继承（只读）


def _metrics(factor: pd.DataFrame, fr: dict) -> dict:
    direction = -1 if float(compute_ic_series(factor, fr[PRIMARY]).dropna().mean()) < 0 else 1
    fa = factor if direction == 1 else -factor
    ic_sum, _ = compute_ic_report(fa, {h: fr[h] for h in IC_HORIZONS})
    lay = layered_backtest(fa, fr[1], n=LAYER_N, g=LAYER_G)
    out = {"dir": direction, "mono": lay.get("monotonicity", np.nan)}
    for h in IC_HORIZONS:
        out[f"ic{h}"] = ic_sum.loc[f"{h}d", "ic_mean"]
        out[f"icir{h}"] = ic_sum.loc[f"{h}d", "icir"]
    ls = lay["summary"]
    out["ls_sharpe"] = ls.loc["LongShort", "sharpe"] if "LongShort" in ls.index else np.nan
    return out


def _process_one(path_str: str) -> dict:
    """worker：单因子 clean→neu→写盘→两版指标。用 _SHARED（fork 继承）。"""
    logger.remove()  # worker 静默，避免 8 进程日志交错
    p = Path(path_str)
    group, name = p.parts[-2], p.stem
    try:
        raw = pd.read_parquet(p); raw.index = pd.to_datetime(raw.index)
        s, e = raw.index.min(), raw.index.max()
        cleaned = prepare_factor(factor=raw, pre_mask=_SHARED["pre"].loc[s:e],
                                 post_mask=_SHARED["post"].loc[s:e], mad_n=MAD_N)
        neu = neutralize(cleaned, _SHARED["industry"], _SHARED["size"], restandardize=True)
        for base, df in ((config.CLEANED_FACTOR_BASE, cleaned), (config.NEU_FACTOR_BASE, neu)):
            d = base / "alpha158" / group; d.mkdir(parents=True, exist_ok=True)
            df.to_parquet(d / f"{name}.parquet")
        fr = _SHARED["fr_eval"]
        c = _metrics(cleaned.loc[START:END], fr)
        n = _metrics(neu.loc[START:END], fr)
        return {"group": group, "factor": name, "dir": n["dir"], "ok": True,
                "ic5_c": c["ic5"], "ic5_n": n["ic5"], "icir5_c": c["icir5"], "icir5_n": n["icir5"],
                "mono_c": c["mono"], "mono_n": n["mono"],
                "ls_sharpe_c": c["ls_sharpe"], "ls_sharpe_n": n["ls_sharpe"]}
    except Exception as ex:
        return {"group": group, "factor": name, "ok": False, "err": str(ex)[:80]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    raw_files = sorted(RAW_BASE.glob("*/*.parquet"))
    if args.limit:
        raw_files = raw_files[: args.limit]
    if not raw_files:
        logger.error(f"{RAW_BASE} 下无 raw 因子"); sys.exit(1)
    logger.info(f"clean→neu 并行 | {len(raw_files)} 因子 | {args.workers} workers | 评估窗 {START}~{END}")

    ref = pd.read_parquet(raw_files[0]); ref.index = pd.to_datetime(ref.index)
    cols = ref.columns
    logger.info("父进程加载共享数据（masks/行业/市值/labels）一次...")
    pre, post = load_filter_masks(combo_mask_path=config.COMBO_MASK_PATH,
                                  new_stock_mask_path=config.NEW_STOCK_MASK_PATH,
                                  reindex_columns=cols)
    industry = pd.read_parquet(config.INDUSTRY_PANEL_ZX_PATH); industry.index = pd.to_datetime(industry.index)
    size = pd.read_parquet(config.MARKET_CAP_PANEL_PATH); size.index = pd.to_datetime(size.index)
    eval_idx = ref.loc[START:END].index
    fr_eval = {}
    for h in IC_HORIZONS + (1,):
        d = pd.read_parquet(config.LABELS_DIR / f"forward_return_{h}d.parquet"); d.index = pd.to_datetime(d.index)
        fr_eval[h] = d.reindex(index=eval_idx, columns=cols)
    _SHARED.update(pre=pre, post=post, industry=industry, size=size, fr_eval=fr_eval)

    t0 = time.time()
    rows = []
    ctx = multiprocessing.get_context("fork")   # COW 共享 _SHARED
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        for i, row in enumerate(ex.map(_process_one, [str(p) for p in raw_files]), 1):
            rows.append(row)
            tag = "✅" if row.get("ok") else "❌"
            if not row.get("ok"):
                logger.warning(f"[{i}/{len(raw_files)}] {tag} {row['group']}/{row['factor']}: {row.get('err')}")
            elif i % 20 == 0 or i == len(raw_files):
                logger.info(f"[{i}/{len(raw_files)}] {row['group']}/{row['factor']} "
                            f"ic5 {row['ic5_c']:+.4f}→{row['ic5_n']:+.4f} ({(time.time()-t0)/i:.1f}s/个均摊)")

    df = pd.DataFrame([r for r in rows if r.get("ok")])
    out_csv = config.INVENTORY_ROOT / "comparison" / f"alpha158_cleaned_vs_neu_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["ok"], errors="ignore").to_csv(out_csv, float_format="%.4f", index=False)

    def agg(cc, cn, label):
        a = df[cc].abs() if "ic" in cc else df[cc]
        b = df[cn].abs() if "ic" in cn else df[cn]
        logger.info(f"  {label:12s} cleaned 均值={a.mean():+.4f}  neu 均值={b.mean():+.4f}  "
                    f"neu更优 {(b.values > a.values).sum()}/{len(df)}")
    logger.info("=" * 64)
    logger.info(f"Alpha158 cleaned vs neu 总体对比（{len(df)} 因子, {START}~{END}, 总耗时 {time.time()-t0:.0f}s）")
    agg("ic5_c", "ic5_n", "|IC5|"); agg("icir5_c", "icir5_n", "ICIR5")
    agg("mono_c", "mono_n", "单调性"); agg("ls_sharpe_c", "ls_sharpe_n", "多空Sharpe")
    logger.info(f"明细 → {out_csv}")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
