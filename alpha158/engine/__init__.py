"""Alpha158 因子引擎（158 个量价技术因子）。

- panel_operators.PanelOperators : 宽表算子底座（Ref/Mean/Std/Slope/Rsquare/...）
- factors.Alpha158Panel          : 158 因子定义（9 K线 + 4 价格 + 23×5 rolling + 6×5 量）
- groups.factor_group            : 因子名 → 组映射（kline/price/rolling/volume）

后复权数据加载见 core.data.adjusted_panels（通用基础设施，非 alpha158 专属）。
入口脚本见 alpha158/ 目录上层（build/daily_update/smoke_replay/verify_repro/plot）。
"""
from .factors import Alpha158Panel
from .panel_operators import PanelOperators

__all__ = ["Alpha158Panel", "PanelOperators"]
