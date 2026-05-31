"""
Factor Inventory（本地原生）：扫所有已产出因子，出 cleaned vs neu 指标总账。

数据源：factors/{cleaned,neu}/<publisher>/<group>/<factor>.parquet（已落盘，直接读盘算指标，
        不重跑清洗/中性化、不载 mask → 快）。labels 一次性加载共享。

产出 factor_inventory/<时间戳>/：
  inventory.csv / inventory.parquet   每因子一行，按 <publisher>/<group> 分组排序，
                                      cleaned 与 neu 指标并列（_c / _n 后缀）
  run_meta.json
并维护 factor_inventory/latest 软链。

用法：python scripts/build_factor_inventory.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import config
from alpha_shared.evaluation.ic import compute_ic_series, compute_ic_report
from alpha_shared.evaluation.layered import layered_backtest

# ── 参数 ──
START, END = "2016-01-01", "2025-12-31"
IC_HORIZONS = (5, 10, 20)
PRIMARY = 5
LAYER_GROUPS, LAYER_REBALANCE = 5, 5
INVENTORY_ROOT = PROJECT_ROOT / "factor_inventory"


def _load_labels_once() -> dict:
    """一次性把 forward_return horizons + 1d 读进内存（全量）；各因子按需内存 reindex。"""
    fr = {}
    for h in IC_HORIZONS + (1,):
        df = pd.read_parquet(config.LABELS_DIR / f"forward_return_{h}d.parquet")
        df.index = pd.to_datetime(df.index)
        fr[h] = df
    return fr


def _metrics(factor: pd.DataFrame, fr: dict) -> dict:
    """单变体指标：direction + 各 horizon IC/ICIR + 单调性 + 多空 Sharpe。"""
    direction = -1 if float(compute_ic_series(factor, fr[PRIMARY]).dropna().mean()) < 0 else 1
    fa = factor if direction == 1 else -factor
    ic_sum, _ = compute_ic_report(fa, {h: fr[h] for h in IC_HORIZONS})
    lay = layered_backtest(fa, fr[1], n=LAYER_REBALANCE, g=LAYER_GROUPS)
    out = {"dir": direction, "mono": lay.get("monotonicity", np.nan)}
    for h in IC_HORIZONS:
        out[f"ic{h}"] = ic_sum.loc[f"{h}d", "ic_mean"]
        out[f"icir{h}"] = ic_sum.loc[f"{h}d", "icir"]
    ls = lay["summary"]
    out["ls_sharpe"] = ls.loc["LongShort", "sharpe"] if "LongShort" in ls.index else np.nan
    return out


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = INVENTORY_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    neu_root = config.NEU_FACTOR_BASE
    factors = sorted(neu_root.glob("*/*/*.parquet"))  # <pub>/<group>/<factor>.parquet
    logger.info(f"Factor Inventory {ts} | {START}~{END} | 扫到 {len(factors)} 个已产出因子")
    if not factors:
        logger.error(f"{neu_root} 下无因子，先 run.py 产出"); sys.exit(1)

    logger.info("加载 labels 到内存（一次）...")
    fr = _load_labels_once()
    rows = []
    t0 = time.time()
    for i, neu_p in enumerate(factors, 1):
        pub, group, fname = neu_p.parts[-3], neu_p.parts[-2], neu_p.stem
        ns = f"{pub}/{group}"
        cln_p = config.CLEANED_FACTOR_BASE / pub / group / neu_p.name
        try:
            neu = pd.read_parquet(neu_p); neu.index = pd.to_datetime(neu.index)
            neu = neu.loc[START:END]
            cln = pd.read_parquet(cln_p); cln.index = pd.to_datetime(cln.index)
            cln = cln.loc[START:END]
            # 内存 reindex（labels 已全量在内存，不再读盘）
            frn = {h: fr[h].reindex(index=neu.index, columns=neu.columns) for h in fr}
            frc = {h: fr[h].reindex(index=cln.index, columns=cln.columns) for h in fr}
            c, n = _metrics(cln, frc), _metrics(neu, frn)
            row = {"publisher": pub, "group": group, "factor": fname,
                   "dir": n["dir"], "mono_c": c["mono"], "mono_n": n["mono"],
                   "ls_sharpe_c": c["ls_sharpe"], "ls_sharpe_n": n["ls_sharpe"]}
            for h in IC_HORIZONS:
                row[f"ic{h}_c"], row[f"ic{h}_n"] = c[f"ic{h}"], n[f"ic{h}"]
                row[f"icir{h}_c"], row[f"icir{h}_n"] = c[f"icir{h}"], n[f"icir{h}"]
            rows.append(row)
            logger.info(f"[{i}/{len(factors)}] {ns}/{fname}  ic5 {c['ic5']:+.4f}→{n['ic5']:+.4f}")
        except Exception as e:
            logger.warning(f"[{i}/{len(factors)}] {ns}/{fname} 失败: {e}")
            rows.append({"publisher": pub, "group": group, "factor": fname, "error": str(e)})

    inv = pd.DataFrame(rows).sort_values(["publisher", "group", "factor"]).reset_index(drop=True)
    inv.to_parquet(out_dir / "inventory.parquet")
    inv.to_csv(out_dir / "inventory.csv", float_format="%.4f", index=False)
    (out_dir / "run_meta.json").write_text(json.dumps({
        "timestamp": ts, "window": [START, END], "ic_horizons": list(IC_HORIZONS),
        "n_factors": len(factors), "eval_seconds": round(time.time() - t0, 1),
        "source": "factors/{cleaned,neu}/<pub>/<group>/ 直读",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    latest = INVENTORY_ROOT / "latest"
    if latest.is_symlink() or latest.exists(): latest.unlink()
    latest.symlink_to(ts, target_is_directory=True)

    logger.info(f"\n✅ inventory: {inv.shape} → {out_dir}/inventory.csv  (latest → {ts})")
    show = [c for c in ["publisher", "group", "factor", "ic5_c", "ic5_n", "icir5_c", "icir5_n",
                        "mono_n", "ls_sharpe_c", "ls_sharpe_n"] if c in inv.columns]
    print(inv[show].to_string(index=False))


if __name__ == "__main__":
    main()
