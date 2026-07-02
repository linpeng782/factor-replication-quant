"""
Alpha158 评估出图 pass：直读已落盘 cleaned/neu → 评估 + 2×2 报告图 → output/alpha158/<group>/<factor>/
============================================================
复用 core.evaluation._analyze_and_plot（IC/ICIR/分层/单调 + 出图），不重算清洗/中性化。
每因子 2 张 PNG：evaluation_<range>__cleaned.png / __neu.png。fork 并行，matplotlib Agg。

用法：python alpha158/plot.py [--workers 4] [--limit N]
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")  # 无界面后端，fork 安全
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import config
from core.evaluation import _analyze_and_plot

START, END = "2016-01-01", "2025-12-31"
IC_HORIZONS = (5, 10, 20)
PRIMARY, LAYER_N, LAYER_G = 5, 5, 5
NEU_BASE = config.NEU_FACTOR_BASE   # 全部来源（cxl + alpha158 + ...）
_SHARED: dict = {}


def _plot_one(neu_path_str: str) -> dict:
    logger.remove()
    p = Path(neu_path_str)
    src, group, name = p.parts[-3], p.parts[-2], p.stem
    try:
        cleaned = pd.read_parquet(config.CLEANED_FACTOR_BASE / src / group / f"{name}.parquet")
        cleaned.index = pd.to_datetime(cleaned.index)
        neu = pd.read_parquet(p); neu.index = pd.to_datetime(neu.index)
        cev, nev = cleaned.loc[START:END], neu.loc[START:END]
        report_dir = config.OUTPUT_DIR / src / group / name
        report_dir.mkdir(parents=True, exist_ok=True)
        fr = _SHARED["fr"]; r1 = _SHARED["r1"]
        kw = dict(factor_name=name, forward_returns=fr, return_1d=r1,
                  ic_horizons=IC_HORIZONS, primary_ic_horizon=PRIMARY,
                  layer_rebalance=LAYER_N, layer_groups=LAYER_G, report_dir=report_dir,
                  start_date=START, end_date=END, plot=True)
        _analyze_and_plot(cev, variant="cleaned", **kw)
        _analyze_and_plot(nev, variant="neu", **kw)
        return {"name": name, "ok": True}
    except Exception as ex:
        return {"name": name, "ok": False, "err": str(ex)[:90]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    files = sorted(NEU_BASE.glob("*/*/*.parquet"))   # <source>/<group>/<factor>.parquet
    if args.limit:
        files = files[: args.limit]
    if not files:
        logger.error(f"{NEU_BASE} 下无 neu 因子，先跑 build_alpha158_cleaned_neu.py"); sys.exit(1)
    logger.info(f"出图 | {len(files)} 因子 × 2 版 = {len(files)*2} 张 | {args.workers} workers | {START}~{END}")

    ref = pd.read_parquet(files[0]); ref.index = pd.to_datetime(ref.index)
    eval_idx = ref.loc[START:END].index; cols = ref.columns
    logger.info("父进程加载 labels（一次）...")
    fr = {}
    for h in IC_HORIZONS:
        d = pd.read_parquet(config.LABELS_DIR / f"forward_return_{h}d.parquet"); d.index = pd.to_datetime(d.index)
        fr[h] = d.reindex(index=eval_idx, columns=cols)
    r1 = pd.read_parquet(config.LABELS_DIR / "forward_return_1d.parquet"); r1.index = pd.to_datetime(r1.index)
    _SHARED.update(fr=fr, r1=r1.reindex(index=eval_idx, columns=cols))

    t0 = time.time(); ok = bad = 0
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        for i, r in enumerate(ex.map(_plot_one, [str(f) for f in files]), 1):
            ok += r["ok"]; bad += (not r["ok"])
            if not r["ok"]:
                logger.warning(f"[{i}/{len(files)}] ❌ {r['name']}: {r.get('err')}")
            elif i % 20 == 0 or i == len(files):
                logger.info(f"[{i}/{len(files)}] {r['name']} 累计 {time.time()-t0:.0f}s")
    logger.success(f"出图完成：成功 {ok}，失败 {bad} → {config.OUTPUT_DIR}/alpha158/  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
