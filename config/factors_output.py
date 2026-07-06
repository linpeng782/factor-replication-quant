"""
[第 2 层｜因子产物] 因子产线的输出：raw→cleaned→neu 三阶段 + alpha158 + 评估输出。

数据流：原料（factors_input.py）→ 算子图算因子 → 产物（本文件）。

后端归属：
  三阶段产物 + OUTPUT_DIR   随 MINUTE_BACKEND（生产隔离轴，整体重定向到 -dquant 并行目录）
  alpha158 产物              随 ALPHA158_BACKEND（消费轴）
"""

from .base import (
    _FACTORS,
    _REPO_ROOT,
    ALPHA158_BACKEND,
    _IS_MINUTE_DQUANT,
)

__all__ = [
    "RAW_FACTOR_BASE",
    "CLEANED_FACTOR_BASE",
    "NEU_FACTOR_BASE",
    "ALPHA158_RAW_BASE",
    "RET20_PANEL_PATH",
    "OUTPUT_DIR",
]

# ============================================================
# 因子三阶段产物 <stage>/<source>/<group>/ + 评估输出
# ============================================================
# 按阶段(raw/cleaned/neu) × 来源(cxl/kysec/founder/...) × 分组分桶；
# namespace=<source>/<group> 由 spec 路径推导（见 spec_resolver.resolve_namespace）。
# ⚠️【生产隔离轴】MINUTE=dquant 时整体重定向到并行 -dquant 目录。
RAW_FACTOR_BASE = _FACTORS / ("raw-dquant" if _IS_MINUTE_DQUANT else "raw")
CLEANED_FACTOR_BASE = _FACTORS / ("cleaned-dquant" if _IS_MINUTE_DQUANT else "cleaned")
NEU_FACTOR_BASE = _FACTORS / ("neu-dquant" if _IS_MINUTE_DQUANT else "neu")

# alpha158 raw 产物（消费轴，随 ALPHA158_BACKEND）：
#   dquant → factors/raw-dquant/alpha158-dquant/（与分钟轴产物同住 raw-dquant/，物理隔离 rq 基线）
#   rq     → factors/raw/alpha158/
ALPHA158_RAW_BASE = (
    _FACTORS / "raw-dquant" / "alpha158-dquant" if ALPHA158_BACKEND == "dquant"
    else _FACTORS / "raw" / "alpha158"
)
# 辅助面板（跨因子共享，不属三阶段产物，存 helpers/ 下）
# Ret20：20日后复权收益面板（宽表），APM 截面回归去动量用（scripts/build_ret20_panel.py 产出）
RET20_PANEL_PATH = _FACTORS / "helpers" / "ret20_panel.parquet"

# 项目内输出目录（相对项目代码，不受 _DATA_ROOT 影响）：报告、图片等评估输出。
# ⚠️【生产隔离轴】MINUTE=dquant 时改用 output-dquant/
OUTPUT_DIR = _REPO_ROOT / ("output-dquant" if _IS_MINUTE_DQUANT else "output")
