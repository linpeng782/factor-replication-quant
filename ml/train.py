"""
Stage 2 — LightGBM (GBDT / MSE) 训练 + 早停
============================================================
当前设定（docs/ml_pipeline_plan.md §5）：
  objective=regression(MSE/L2)；boosting=gbdt(先跑通)；valid 早停；单种子；全 A。
valid 仅用于早停，绝不喂梯度、绝不 fit 尺子（防泄露）。LightGBM 原生处理 NaN。
"""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from loguru import logger

DEFAULT_PARAMS = dict(
    objective="regression",
    metric="l2",
    boosting_type="gbdt",      # 先跑通；后续可切 "dart"
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l1=0.0,
    lambda_l2=0.0,
    num_threads=0,
    seed=1,                    # 先单种子
    verbose=-1,
)
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 80


def train_gbdt(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_valid: pd.DataFrame, y_valid: pd.Series,
    params: dict | None = None,
    num_boost_round: int = NUM_BOOST_ROUND,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    verbose_eval: int = 0,
):
    """训练单个 LightGBM 回归模型；valid 早停。返回 (booster, best_iteration, eval_hist)。"""
    p = {**DEFAULT_PARAMS, **(params or {})}
    dtr = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dva = lgb.Dataset(X_valid, label=y_valid, reference=dtr, free_raw_data=False)
    hist: dict = {}
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.record_evaluation(hist),
    ]
    if verbose_eval:
        callbacks.append(lgb.log_evaluation(verbose_eval))
    booster = lgb.train(
        p, dtr, num_boost_round=num_boost_round,
        valid_sets=[dtr, dva], valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    logger.info(f"[train] best_iteration={booster.best_iteration} "
                f"valid_l2={hist['valid']['l2'][booster.best_iteration-1]:.6f}")
    return booster, booster.best_iteration, hist
