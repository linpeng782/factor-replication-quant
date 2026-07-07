"""
多 seed 集成 —— 5 个 LGBM（不同 seed）预测面板 rank 平均 → 导信号
============================================================
动机：单模型早停位置对标签微噪声敏感（rq vs dquant labels 实测 1e-6 标签差
→ top-500 每日换血 ~58 只），5-seed rank 平均把这类噪声互相抵消，
top-k 选股更稳定（预期还小幅改善 IC / 回撤）。

用法：改下方 MEMBERS / OUT_NAME → python -m ml_core.ensemble_seeds
产物：ML_PREDICTIONS_DIR/<OUT_NAME>/pred_panel_live.parquet + signals/
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

import config
from ml_core.signals import export_panel

# ── 手动参数（按需修改）──
MEMBERS = [
    "lgbm_a158_p27_shap_dqlabels",     # seed 42（已有，Stage-1/2 全 42）
    "lgbm_a158_p27_shap_dq_s43",
    "lgbm_a158_p27_shap_dq_s44",
    "lgbm_a158_p27_shap_dq_s45",
    "lgbm_a158_p27_shap_dq_s46",
]
OUT_NAME = "ensemble_seed5_dq"
TOP_N = 500


def main() -> None:
    panels, dates, cols = {}, None, None
    for m in MEMBERS:
        p = pd.read_parquet(config.ML_PREDICTIONS_DIR / m / "pred_panel_live.parquet")
        logger.info(f"[seed-ens] {m}: {p.shape} {p.index.min().date()}~{p.index.max().date()}")
        panels[m] = p
        dates = p.index if dates is None else dates.intersection(p.index)
        cols = p.columns if cols is None else cols.intersection(p.columns)

    # 逐日截面 rank(pct) 消除量纲 → 等权平均（任一成员 NaN 则该格 NaN，即取预测池交集）
    ranks = [p.loc[dates, cols].rank(axis=1, pct=True) for p in panels.values()]
    ens = sum(ranks) / len(ranks)
    logger.info(f"[seed-ens] 对齐后 {ens.shape} | {dates.min().date()}~{dates.max().date()} "
                f"| 非空 {ens.notna().values.sum():,}")

    out_dir = config.ML_PREDICTIONS_DIR / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    ens.to_parquet(out_dir / "pred_panel_live.parquet")
    export_panel(ens, out_dir / "signals", top_n=TOP_N, rebuild=True)
    logger.success(f"[seed-ens] {len(MEMBERS)} 成员集成 → {out_dir}")


if __name__ == "__main__":
    main()
