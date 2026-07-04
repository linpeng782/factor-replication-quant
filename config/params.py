"""
[横切] 复权字段 / 计算参数 / 默认 fetch 与评估区间。

与具体数据后端无关的纯参数常量，全线共享。
"""

import os

__all__ = [
    "PRICE_FIELDS",
    "VOLUME_FIELDS",
    "COMPUTE_WORKERS",
    "DEFAULT_START_DATE",
    "DEFAULT_END_DATE",
    "DEFAULT_EVAL_START_DATE",
    "DEFAULT_EVAL_END_DATE",
]

# ==================== 复权字段 + 计算参数 ====================
# 复权时价格字段 ×cum_factor、成交量字段 ÷cum_factor
PRICE_FIELDS = ["open", "high", "low", "close", "limit_up", "limit_down"]
VOLUME_FIELDS = ["volume"]
# 因子批量计算并行进程数（本机核数自适应；脚本可覆盖）
COMPUTE_WORKERS = max(4, (os.cpu_count() or 8))

# ==================== 默认 fetch / 评估区间 ====================
# fetch 区间（panel 落盘窗口）— 提前到 2010 给所有因子留 ≥4 年 warm-up，
# 使得最复杂的 8 期 PIT 滚动因子（npf_mrq_sue8 / np_*_rank 等）能从 2014-01-02 起有有效值
DEFAULT_START_DATE = "20100101"
DEFAULT_END_DATE = "20260527"

# 评估区间（IC / ICIR 计算窗口）— 保持原值，跟历史对齐数字（PROGRESS §四 + 2026-05-28 对齐报告）可比
DEFAULT_EVAL_START_DATE = "20160101"
DEFAULT_EVAL_END_DATE = "20251231"
