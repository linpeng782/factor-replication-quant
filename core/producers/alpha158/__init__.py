"""Alpha158 因子库（从 my-alpha-engine 保真抽取）。

- panel_operators.PanelOperators : 宽表算子底座（Ref/Mean/Std/Slope/Rsquare/...）
- factors.Alpha158Panel          : 158 因子定义（9 K线 + 4 价格 + 23×5 rolling + 6×5 量）
- loader                         : 后复权数据加载（raw per-stock + 稀疏 cum_factor → 宽表）

入口脚本见 alpha158/ 目录（build/daily_update/smoke_replay/verify_repro/plot）；总览见 alpha158/README.md。
"""
from .factors import Alpha158Panel
from .panel_operators import PanelOperators

__all__ = ["Alpha158Panel", "PanelOperators"]
