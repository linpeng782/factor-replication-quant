"""
SHAP 可视化：对已训练 run 的最终模型画 beeswarm(蜂群) + bar(条形) 图
============================================================
用最终模型(model.txt, 仅入选因子) + 训练段 scaler，在【样本外 test 区间】采样，
算 TreeSHAP，输出两张图：
  - beeswarm：每个因子每个样本一个点，横轴=SHAP值(对预测的拉动)，颜色=因子值高低
  - bar：mean(|SHAP|) 全局重要性条形

用法：
    python -m ml.shap_plot --run-id shap_top64
    python -m ml.shap_plot --run-id shap_top64 --n-sample 20000 --date-step 20
产物：ml/shap_<run_id>_beeswarm.png、ml/shap_<run_id>_bar.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from loguru import logger

matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402

from core import config  # noqa: E402
from ml.dataset import SPLIT, discover_features, load_can_buy_mask  # noqa: E402
from ml.predict_live import _load_scaler  # noqa: E402

REPO_ML = Path(__file__).resolve().parent  # 图落 repo 内 ml/


def build_sample(sel: list[str], n_sample: int, date_step: int) -> pd.DataFrame:
    """在 test 区间按 date_step 抽日 → 读入选因子 → can_buy_mask 取值 → 再随机抽 n_sample 行（原始未标准化）。"""
    pathmap = discover_features()
    lo, hi = SPLIT["test"]
    pre = load_can_buy_mask()
    dates = pre.index[(pre.index >= pd.Timestamp(lo)) & (pre.index <= pd.Timestamp(hi))][::date_step]
    pre = pre.loc[dates]; stocks = pre.columns
    pm = pre.fillna(False).to_numpy(dtype=bool)
    rr, cc = np.where(pm)
    mat = np.full((len(rr), len(sel)), np.nan, dtype=np.float32)
    for j, name in enumerate(sel):
        df = pd.read_parquet(pathmap[name]); df.index = pd.to_datetime(df.index)
        arr = df.reindex(index=dates, columns=stocks).to_numpy(dtype=np.float32, copy=True)  # 可写副本：避免 pyarrow 只读视图
        arr[~np.isfinite(arr)] = np.nan
        mat[:, j] = arr[rr, cc]
    X = pd.DataFrame(mat, columns=sel)
    if n_sample and len(X) > n_sample:
        X = X.sample(n_sample, random_state=0)
    logger.info(f"[shap-plot] 采样 {len(X):,} 行 × {len(sel)} 因子（test 区间，每 {date_step} 日抽 1）")
    return X


def main() -> None:
    ap = argparse.ArgumentParser(description="SHAP beeswarm + bar 可视化")
    ap.add_argument("--run-id", default="shap_top64")
    ap.add_argument("--n-sample", type=int, default=20000)
    ap.add_argument("--date-step", type=int, default=20)
    ap.add_argument("--max-display", type=int, default=20, help="图里显示前 N 个因子")
    args = ap.parse_args()

    model_dir = config.ML_MODELS_DIR / args.run_id
    sel: list[str] = json.loads((model_dir / "selected_features.json").read_text())["features"]
    model = lgb.Booster(model_file=str(model_dir / "model.txt"))

    X_raw = build_sample(sel, args.n_sample, args.date_step)
    sx = _load_scaler(model_dir, sel)
    Xz = sx.transform(X_raw)  # 与训练同尺子标准化

    sv = shap.TreeExplainer(model).shap_values(Xz)
    logger.info(f"[shap-plot] shap_values: {np.asarray(sv).shape}")

    # beeswarm：横轴 SHAP 值，颜色=因子值（红高蓝低）。展示因子值如何影响预测方向
    plt.figure()
    shap.summary_plot(sv, Xz, max_display=args.max_display, show=False)
    plt.title(f"SHAP beeswarm — {args.run_id} (test, {len(Xz):,} samples)")
    plt.tight_layout()
    bee = REPO_ML / f"shap_{args.run_id}_beeswarm.png"
    plt.savefig(bee, dpi=130, bbox_inches="tight"); plt.close()

    # bar：mean(|SHAP|) 全局重要性
    plt.figure()
    shap.summary_plot(sv, Xz, plot_type="bar", max_display=args.max_display, show=False)
    plt.title(f"SHAP mean(|value|) — {args.run_id}")
    plt.tight_layout()
    bar = REPO_ML / f"shap_{args.run_id}_bar.png"
    plt.savefig(bar, dpi=130, bbox_inches="tight"); plt.close()

    logger.success(f"[shap-plot] 已保存：\n  {bee}\n  {bar}")


if __name__ == "__main__":
    main()
