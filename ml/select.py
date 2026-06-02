"""
Stage 1 — 因子筛选（样本内，只用 train+valid，test 全程不可见）
============================================================
当前：GBDT 自带 feature_importance（gain）排序 → 选 top-64（先跑通）。
后续：SHAP TreeExplainer + mean(|SHAP|) 选一遍，对比两者选出的因子差异。
国金之十三§3：GBDT 自带特征选择能力；我们仍筛选是为降冗余、降过拟合、提升可解释。
**不做 MMR；一次性筛选。** 产出 selected_features.json。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ml.train import train_gbdt

TOP_K = 64


def select_by_gbdt_importance(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_valid: pd.DataFrame, y_valid: pd.Series,
    top_k: int = TOP_K,
    importance_type: str = "gain",
    params: dict | None = None,
):
    """训 LightGBM(train + valid 早停) → feature_importance(gain) 排序 → 取 top_k。

    返回 (selected: list[str], scores: pd.Series 按重要性降序, booster)。仅用 train+valid。
    """
    booster, best_it, _ = train_gbdt(X_train, y_train, X_valid, y_valid, params=params)
    imp = booster.feature_importance(importance_type=importance_type, iteration=best_it)
    scores = pd.Series(imp, index=list(X_train.columns)).sort_values(ascending=False)
    k = min(top_k, len(scores))
    selected = scores.index[:k].tolist()
    logger.info(f"[select-gbdt] {len(scores)} 因子 → 取 top-{k}；最高 {scores.index[0]}={scores.iloc[0]:.1f}")
    return selected, scores, booster


def select_by_shap(
    X_train, y_train, X_valid, y_valid,
    top_k: int = TOP_K, n_sample: int = 100_000, params: dict | None = None,
):
    """（后续）SHAP TreeExplainer：训 LightGBM → 采样 → mean(|SHAP|) 排序 → top_k。对照 gbdt-importance。"""
    import shap  # 延迟导入
    booster, best_it, _ = train_gbdt(X_train, y_train, X_valid, y_valid, params=params)
    Xall = pd.concat([X_train, X_valid])
    Xs = Xall.sample(min(n_sample, len(Xall)), random_state=0)
    sv = shap.TreeExplainer(booster).shap_values(Xs)
    scores = pd.Series(np.abs(sv).mean(axis=0), index=list(X_train.columns)).sort_values(ascending=False)
    k = min(top_k, len(scores))
    selected = scores.index[:k].tolist()
    logger.info(f"[select-shap] {len(scores)} 因子 → 取 top-{k}（采样 {len(Xs):,}）")
    return selected, scores, booster


def save_selection(selected: list[str], scores: pd.Series, method: str, out_dir: Path) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "selected_features.json"
    path.write_text(json.dumps({
        "method": method, "top_k": len(selected), "features": selected,
        "scores": {k: float(v) for k, v in scores.items()},
    }, ensure_ascii=False, indent=2))
    return path
