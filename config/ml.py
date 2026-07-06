"""
[第 3 层｜ML 训练产物] ml/ 数据线产出（模型 / 预测 / 信号 / 日志）。

代码线（repo 顶层 ml_core/）与数据线（此处，数据根下）分离。
★ 这些常量只被 ML 训练线（ml_core/）消费；因子生产线不碰。
  依赖方向单向：ML 读因子产物（factors_output.py 的 RAW/NEU_FACTOR_BASE）+ 评估输入（market_data.py 的 mask/labels），
  反之因子线不依赖这里。
"""

from .base import _DATA_ROOT, _REPO_ROOT

__all__ = [
    "ML_ROOT",
    "ML_MODELS_DIR",
    "ML_PREDICTIONS_DIR",
    "ML_DATASETS_DIR",
    "ML_SIGNALS_DIR",
    "ML_LOGS_DIR",
]

ML_ROOT = _DATA_ROOT / "ml"
ML_MODELS_DIR = ML_ROOT / "models"            # lgbm 模型 + 超参 + RobustZScore 尺子
ML_PREDICTIONS_DIR = ML_ROOT / "predictions"  # ŷ 面板 + test 评估(IC)
ML_DATASETS_DIR = ML_ROOT / "datasets"        # (可选) train/valid/test 矩阵，便于复跑
ML_SIGNALS_DIR = ML_ROOT / "signals"          # 回测可读信号：每日排序选股名单 txt（export_signal.py 产出）
# 日志落在【repo 内 ml/logs/】（方便查看，受 .gitignore 排除），非数据根
ML_LOGS_DIR = _REPO_ROOT / "ml" / "logs"      # 每次 run 的训练日志
