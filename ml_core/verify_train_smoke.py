"""
训练编排冒烟：run_train 跑通 LGBM(回归) + MLP(二分类) 两条路径（date_sample 加速）。
仅验证"端到端可训 + test IC 方向合理"，非复现历史数字。

运行：ALPHA158_DATA_BACKEND=dquant python -m ml_core.verify_train_smoke
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from ml_core.features import HasFactorPolicy
from ml_core.labels import BinaryMedian, ExcessReturn
from ml_core.metrics import daily_rank_ic
from ml_core.model import LGBMAdapter, MLPAdapter
from ml_core.pipeline import PipelineConfig, run_train
from ml_core.scaling import DailyCrossSectionMAD, WholeSetRobustZ

DS = 40  # date_sample：每 40 个交易日取 1，纯加速


def _test_ic(res) -> dict:
    te = res["seg_row"]["test"]
    pred = res["adapter"].predict(res["Xz"][te])
    return daily_rank_ic(pred, res["y"][te], res["fm"].dates[te])


def main() -> None:
    # ── LGBM：超额回归 + 全集 RobustZ + NONE ──
    logger.info("=== LGBM 训练冒烟（alpha158-dquant，date_sample=40）===")
    cfg = PipelineConfig(sources=["alpha158-dquant"], has_factor_policy=HasFactorPolicy.NONE, horizon=20)
    res = run_train(cfg, ExcessReturn(), WholeSetRobustZ(), LGBMAdapter(), date_sample=DS)
    ic = _test_ic(res)
    logger.success(f"  LGBM test rank-IC={ic['ic_mean']:+.4f} ICIR={ic['icir']:+.2f} 天数={ic['n_days']}")
    assert ic["ic_mean"] > 0, "❌ LGBM test IC 非正，疑管线异常"

    # ── MLP：二分类 + 逐日截面 MAD + ALL ──
    logger.info("=== MLP 训练冒烟（alpha158-dquant，date_sample=40，max_epochs=3）===")
    cfg2 = PipelineConfig(sources=["alpha158-dquant"], has_factor_policy=HasFactorPolicy.ALL, horizon=20)
    mlp = MLPAdapter(n_features=158, max_epochs=3, patience=3)
    res2 = run_train(cfg2, BinaryMedian(), DailyCrossSectionMAD(), mlp, date_sample=DS)
    ic2 = _test_ic(res2)
    logger.success(f"  MLP test rank-IC={ic2['ic_mean']:+.4f} ICIR={ic2['icir']:+.2f} 天数={ic2['n_days']}")
    logger.success("  ✅ 两条训练路径端到端跑通")


if __name__ == "__main__":
    main()
