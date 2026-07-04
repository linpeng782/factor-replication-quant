"""
Factor Inventory（本地原生，fork 并行）：扫所有已产出因子，出 cleaned vs neu 指标总账 + neu IC 序列矩阵。

数据源：factors/{cleaned,neu}/<pub>/<group>/<factor>.parquet（直读盘算指标，不重跑清洗/中性化）。
labels 一次性加载（父进程），fork 出 N 个 worker 经 COW 共享、不复制。BLAS 线程限 1。

产出 <DATA_ROOT>/factor-inventory/<时间戳>/（代码与数据分离）：
  inventory.csv / inventory.parquet   每因子一行，cleaned(_c)/neu(_n) 指标并列
  ic_series_neu_5d.parquet            日 × 因子 IC 矩阵（已 direction 校正）—— 供相关/VIF/ICIR加权复用
  ic_series_neu_20d.parquet
  run_meta.json
并维护 factor-inventory/latest 软链。

用法：python scripts/build_factor_inventory.py [--workers 4]
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from alpha_shared.evaluation.ic import compute_ic_series, compute_ic_report
from alpha_shared.evaluation.layered import layered_backtest

# ── 参数 ──
START, END = "2016-01-01", "2025-12-31"
IC_HORIZONS = (5, 10, 20)
PRIMARY = 5
LAYER_GROUPS, LAYER_REBALANCE = 5, 5
IC_SERIES_HORIZONS = (5, 20)            # 落盘的 neu IC 序列矩阵 horizon
INVENTORY_ROOT = config.INVENTORY_ROOT  # <DATA_ROOT>/factor-inventory/

_FR: dict = {}   # 父进程加载 → fork 后 worker 经 COW 继承（只读）


def _metrics(factor: pd.DataFrame, fr: dict) -> tuple[dict, dict]:
    """单变体指标 + 日 IC 序列（已 direction 校正）。返回 (out, ic_series_dict)。"""
    direction = -1 if float(compute_ic_series(factor, fr[PRIMARY]).dropna().mean()) < 0 else 1
    fa = factor if direction == 1 else -factor
    ic_sum, ic_ser = compute_ic_report(fa, {h: fr[h] for h in IC_HORIZONS})
    lay = layered_backtest(fa, fr[1], n=LAYER_REBALANCE, g=LAYER_GROUPS)
    out = {"dir": direction, "mono": lay.get("monotonicity", np.nan)}
    for h in IC_HORIZONS:
        out[f"ic{h}"] = ic_sum.loc[f"{h}d", "ic_mean"]
        out[f"icir{h}"] = ic_sum.loc[f"{h}d", "icir"]
    ls = lay["summary"]
    out["ls_sharpe"] = ls.loc["LongShort", "sharpe"] if "LongShort" in ls.index else np.nan
    return out, ic_ser


def _process_one(neu_path_str: str) -> dict:
    """worker：单因子 cleaned/neu 指标 + neu 日 IC 序列。用 _FR（fork 继承）。"""
    logger.remove()
    p = Path(neu_path_str)
    pub, group, fname = p.parts[-3], p.parts[-2], p.stem
    cln_p = config.CLEANED_FACTOR_BASE / pub / group / p.name
    try:
        neu = pd.read_parquet(p); neu.index = pd.to_datetime(neu.index); neu = neu.loc[START:END]
        cln = pd.read_parquet(cln_p); cln.index = pd.to_datetime(cln.index); cln = cln.loc[START:END]
        frn = {h: _FR[h].reindex(index=neu.index, columns=neu.columns) for h in _FR}
        frc = {h: _FR[h].reindex(index=cln.index, columns=cln.columns) for h in _FR}
        (c, _), (n, n_ser) = _metrics(cln, frc), _metrics(neu, frn)
        row = {"publisher": pub, "group": group, "factor": fname,
               "dir": n["dir"], "mono_c": c["mono"], "mono_n": n["mono"],
               "ls_sharpe_c": c["ls_sharpe"], "ls_sharpe_n": n["ls_sharpe"]}
        for h in IC_HORIZONS:
            row[f"ic{h}_c"], row[f"ic{h}_n"] = c[f"ic{h}"], n[f"ic{h}"]
            row[f"icir{h}_c"], row[f"icir{h}_n"] = c[f"icir{h}"], n[f"icir{h}"]
        return {"ok": True, "pub": pub, "fname": fname, "row": row,
                "ic": {h: n_ser[h] for h in IC_SERIES_HORIZONS}, "ic5": n["ic5"], "ic5c": c["ic5"]}
    except Exception as e:
        return {"ok": False, "pub": pub, "fname": fname,
                "row": {"publisher": pub, "group": group, "factor": fname, "error": str(e)[:120]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = INVENTORY_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    factors = sorted(config.NEU_FACTOR_BASE.glob("*/*/*.parquet"))  # <pub>/<group>/<factor>.parquet
    logger.info(f"Factor Inventory {ts} | {START}~{END} | {len(factors)} 因子 | {args.workers} workers")
    if not factors:
        logger.error(f"{config.NEU_FACTOR_BASE} 下无因子，先产出"); sys.exit(1)

    logger.info("父进程加载 labels（一次，fork 共享）...")
    for h in IC_HORIZONS + (1,):
        d = pd.read_parquet(config.LABELS_DIR / f"forward_return_{h}d.parquet"); d.index = pd.to_datetime(d.index)
        _FR[h] = d

    t0 = time.time()
    ctx = multiprocessing.get_context("fork")
    rows, ic_ser_store, n_done = [], {h: {} for h in IC_SERIES_HORIZONS}, 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        for i, r in enumerate(ex.map(_process_one, [str(p) for p in factors]), 1):
            rows.append(r["row"])
            if r["ok"]:
                n_done += 1
                key = r["fname"] if r["fname"] not in ic_ser_store[IC_SERIES_HORIZONS[0]] else f"{r['pub']}__{r['fname']}"
                for h in IC_SERIES_HORIZONS:
                    ic_ser_store[h][key] = r["ic"][h]
                if i % 20 == 0 or i == len(factors):
                    logger.info(f"[{i}/{len(factors)}] {r['pub']}/{r['fname']} "
                                f"ic5 {r['ic5c']:+.4f}→{r['ic5']:+.4f} ({(time.time()-t0)/i:.1f}s/个均摊)")
            else:
                logger.warning(f"[{i}/{len(factors)}] {r['pub']}/{r['fname']} 失败: {r['row'].get('error')}")

    inv = pd.DataFrame(rows).sort_values(["publisher", "group", "factor"]).reset_index(drop=True)
    inv.to_parquet(out_dir / "inventory.parquet")
    inv.to_csv(out_dir / "inventory.csv", float_format="%.4f", index=False)

    for h in IC_SERIES_HORIZONS:
        mat = pd.DataFrame(ic_ser_store[h]).sort_index()
        mat.to_parquet(out_dir / f"ic_series_neu_{h}d.parquet")
        logger.info(f"  IC序列矩阵 neu {h}d: {mat.shape} → ic_series_neu_{h}d.parquet")

    (out_dir / "run_meta.json").write_text(json.dumps({
        "timestamp": ts, "window": [START, END], "ic_horizons": list(IC_HORIZONS),
        "n_factors": len(factors), "n_ok": n_done, "workers": args.workers,
        "eval_seconds": round(time.time() - t0, 1),
        "source": "factors/{cleaned,neu}/<pub>/<group>/ 直读",
        "ic_series_neu": {f"{h}d": f"ic_series_neu_{h}d.parquet" for h in IC_SERIES_HORIZONS},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    latest = INVENTORY_ROOT / "latest"
    if latest.is_symlink() or latest.exists(): latest.unlink()
    latest.symlink_to(ts, target_is_directory=True)

    logger.info(f"\n✅ inventory: {inv.shape} → {out_dir}  (latest → {ts}) 总耗时 {time.time()-t0:.0f}s")
    show = [c for c in ["publisher", "group", "factor", "ic5_n", "icir5_n", "icir20_n",
                        "mono_n", "ls_sharpe_n"] if c in inv.columns]
    print(inv[show].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
