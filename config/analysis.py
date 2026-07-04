"""
[第 3 层｜因子分析产物] factor-inventory/。

总账(inventory) / IC序列矩阵(ic_series) / 相关性(correlation) / 对比(comparison) 等
数值分析产物——代码与数据分离，统一落数据根下（受 FACTOR_REPL_DATA_ROOT 控制）。
只被分析脚本（scripts/build_factor_inventory.py / factor_correlation.py）消费。
"""

from .base import _DATA_ROOT

__all__ = ["INVENTORY_ROOT"]

INVENTORY_ROOT = _DATA_ROOT / "factor-inventory"
