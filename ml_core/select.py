"""
ml_core.select —— Stage-1 因子筛选（SHAP / GBDT 重要性），从 ml.select 上提
============================================================
两阶段 LGBM 的第一阶段：在 train+valid 上训一棵 GBDT，按 mean(|SHAP|) 或
feature_importance(gain) 排序取 top-k。**仅用 train+valid，test 全程不可见**。

数值口径与 ml.select 一致：
  - Stage-1 GBDT 用 LGBMAdapter（参数同 ml.train.DEFAULT_PARAMS：seed=42 deterministic num_threads=64）。
  - SHAP 采样 random_state=0；pandas.DataFrame.sample 的抽样位置只依赖「行数 + seed」，
    与索引标签无关，故用 numpy 入参重建 DataFrame 不改变被抽中的样本（与 ml.select 等价）。

接口签名统一为 selector(X_train, y_train, X_valid, y_valid, feature_names) → (selected, scores)，
可直接注入 ml_core.pipeline.run_train 的两阶段编排（用 functools.partial 固定 top_k 等）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from ml_core.model import LGBMAdapter

TOP_K = 64


def select_by_gbdt_importance(
    X_train, y_train, X_valid, y_valid, feature_names,
    top_k: int = TOP_K, importance_type: str = "gain", params: dict | None = None,
) -> tuple[list[str], pd.Series]:
    """训 GBDT(train+valid 早停) → feature_importance(gain) 排序 → 取 top-k。仅用 train+valid。"""
    adapter = LGBMAdapter(params).fit(X_train, y_train, X_valid, y_valid, tag="select")
    imp = adapter.feature_importance(importance_type=importance_type)
    scores = pd.Series(imp, index=list(feature_names)).sort_values(ascending=False)
    k = min(top_k, len(scores))
    selected = scores.index[:k].tolist()
    logger.info(f"[select-gbdt] {len(scores)} 因子 → 取 top-{k}；最高 {scores.index[0]}={scores.iloc[0]:.1f}")
    return selected, scores


def select_by_shap(
    X_train, y_train, X_valid, y_valid, feature_names,
    top_k: int = TOP_K, n_sample: int = 100_000, params: dict | None = None,
) -> tuple[list[str], pd.Series]:
    """训 GBDT → 采样 → SHAP TreeExplainer → mean(|SHAP|) 排序 → top-k（对照 ml.select.select_by_shap）。"""
    import shap  # 延迟导入
    adapter = LGBMAdapter(params).fit(X_train, y_train, X_valid, y_valid, tag="select-shap")
    Xall = pd.DataFrame(np.concatenate([X_train, X_valid], axis=0), columns=list(feature_names))
    Xs = Xall.sample(min(n_sample, len(Xall)), random_state=0)
    sv = shap.TreeExplainer(adapter.booster).shap_values(Xs)
    scores = pd.Series(np.abs(sv).mean(axis=0), index=list(feature_names)).sort_values(ascending=False)
    k = min(top_k, len(scores))
    selected = scores.index[:k].tolist()
    logger.info(f"[select-shap] {len(scores)} 因子 → 取 top-{k}（采样 {len(Xs):,}）")
    return selected, scores
