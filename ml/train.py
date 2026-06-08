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
    num_threads=64,
    seed=42,
    deterministic=True,        # 多线程浮点累加顺序固定 → 同 seed 同数据 bit 级复现
    force_row_wise=True,       # deterministic 要求显式指定 row/col wise
    verbose=-1,
)
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 200


def _make_live_logger(tag: str, period: int):
    """LightGBM callback：训练中每 period 轮（及第 1 轮）实时把 train/valid loss 写日志。
    放在 early_stopping 之后执行，确保拿到当轮已算好的 evaluation_result_list。"""
    def _cb(env) -> None:
        it = env.iteration + 1  # env.iteration 0-indexed
        if it != 1 and it % period != 0:
            return
        res = {f"{dn}_{en}": v for dn, en, v, *_ in env.evaluation_result_list}
        logger.info(f"[{tag}]   round {it:>4}: "
                    f"train_l2={res.get('train_l2', float('nan')):.6f}  "
                    f"valid_l2={res.get('valid_l2', float('nan')):.6f}")
    _cb.order = 20  # 在 early_stopping(order=30 默认更小) 后跑；数值越大越靠后
    return _cb


def train_gbdt(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_valid: pd.DataFrame, y_valid: pd.Series,
    params: dict | None = None,
    num_boost_round: int = NUM_BOOST_ROUND,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    tag: str = "train",
    log_every: int = 25,
):
    """训练单个 LightGBM 回归模型；valid 早停。返回 (booster, best_iteration, eval_hist)。

    训练过程中每 log_every 轮**实时**把 train/valid loss 写日志（可 tail -f 边训边看），
    不再等训练结束才一次性打印。tag 用于区分不同阶段（如 'select' / 'final'）。
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    dtr = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dva = lgb.Dataset(X_valid, label=y_valid, reference=dtr, free_raw_data=False)
    hist: dict = {}
    logger.info(f"[{tag}] 开始训练（最多 {num_boost_round} 轮，每 {log_every} 轮记录一次损失，metric=l2）：")
    booster = lgb.train(
        p, dtr, num_boost_round=num_boost_round,
        valid_sets=[dtr, dva], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                   lgb.record_evaluation(hist),
                   _make_live_logger(tag, log_every)],
    )
    va = hist["valid"]["l2"]; n = len(va); best = booster.best_iteration
    logger.info(f"[{tag}] best_iteration={best}  valid_l2={va[best-1]:.6f}  "
                f"(早停于第 {n} 轮，valid 连续 {early_stopping_rounds} 轮无改善)")
    return booster, best, hist
