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
    tag: str = "train",
    n_log_points: int = 15,
):
    """训练单个 LightGBM 回归模型；valid 早停。返回 (booster, best_iteration, eval_hist)。

    训练过程的 train/valid loss 下降曲线会写入日志（约 n_log_points 个采样点 + best）。
    tag 用于区分不同阶段的训练（如 'select' / 'final'）。
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    dtr = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dva = lgb.Dataset(X_valid, label=y_valid, reference=dtr, free_raw_data=False)
    hist: dict = {}
    booster = lgb.train(
        p, dtr, num_boost_round=num_boost_round,
        valid_sets=[dtr, dva], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                   lgb.record_evaluation(hist)],
    )
    # 把 train/valid loss 下降曲线写进日志（record_evaluation 已逐轮捕获）
    tr, va = hist["train"]["l2"], hist["valid"]["l2"]
    n = len(va); best = booster.best_iteration
    step = max(1, n // n_log_points)
    logger.info(f"[{tag}] 训练损失曲线（共 {n} 轮，metric=l2）：")
    rounds = sorted(set(list(range(0, n, step)) + [n - 1]))
    for i in rounds:
        mark = "  <- best" if (i + 1) == best else ""
        logger.info(f"    round {i+1:>4}: train_l2={tr[i]:.6f}  valid_l2={va[i]:.6f}{mark}")
    logger.info(f"[{tag}] best_iteration={best}  valid_l2={va[best-1]:.6f}  "
                f"(早停于第 {n} 轮，valid 连续 {early_stopping_rounds} 轮无改善)")
    return booster, best, hist
