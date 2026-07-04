"""
集成两个模型（固定训练 A + 滚动训练 B）的预测面板，导信号供回测。
方案：rank 平均（消除量纲差异）+ 几种权重组合。
"""
import pandas as pd, numpy as np
from pathlib import Path
from loguru import logger
import config
from ml_core.signals import export_panel

A_PATH = config.ML_PREDICTIONS_DIR / "lgbm_a158_p27_top64" / "pred_panel_live.parquet"
B_PATH = config.ML_PREDICTIONS_DIR / "lgbm_rolling_concat_2019_2025" / "pred_panel_live.parquet"

# 权重方案: (w_A, w_B) — 对截面 rank 加权平均
SCHEMES = {
    "ensemble_50_50": (0.5, 0.5),
    "ensemble_60_40": (0.6, 0.4),
    "ensemble_70_30": (0.7, 0.3),
    "ensemble_40_60": (0.4, 0.6),
}

def main():
    A = pd.read_parquet(A_PATH)
    B = pd.read_parquet(B_PATH)
    common_dates = A.index.intersection(B.index)
    common_cols = A.columns.intersection(B.columns)
    A = A.loc[common_dates, common_cols]
    B = B.loc[common_dates, common_cols]
    logger.info(f"对齐: {A.shape} | {common_dates.min().date()}~{common_dates.max().date()}")

    # 逐日截面 rank（0~1），消除两模型量纲差异
    A_rank = A.rank(axis=1, pct=True)
    B_rank = B.rank(axis=1, pct=True)

    for name, (wA, wB) in SCHEMES.items():
        ensemble = wA * A_rank + wB * B_rank
        out_dir = config.ML_PREDICTIONS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        ensemble.to_parquet(out_dir / "pred_panel_live.parquet")
        sig_dir = out_dir / "signals"
        export_panel(ensemble, sig_dir, top_n=500, rebuild=True)
        logger.success(f"{name} (A={wA}/B={wB}) → {out_dir}")

if __name__ == "__main__":
    main()
