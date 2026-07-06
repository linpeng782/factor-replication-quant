"""
因子复现项目配置（门面）。

内部按「依赖层」拆到子模块，但对外仍是单一 `config` 命名空间——
`import config`、`from config import X`、`config._MKT` 全部照旧可用，调用点零改动。

分层（依赖单向，自上而下）：
  base            数据根 + 后端开关控制台（第 0 层地基，所有层都依赖）
  market_data     market-data/ 评估输入（第 1 层）
  factors_input   因子产线原料：日频行情 / 基本面 / 分钟（第 2 层）
  factors_output  因子产线产物：raw/cleaned/neu 三阶段 + 评估输出（第 2 层）
  ml              ML 训练产物（第 3 层，仅 ML 线消费）
  analysis        factor-inventory 分析产物（第 3 层，仅分析脚本消费）
  params          复权字段 / 计算参数 / 默认区间（横切）

各子模块以 __all__ 声明其对外常量；本门面全量汇总（下方 import *）。
数据后端切换详见 base.py 顶部说明。
"""

from .base import *
from .market_data import *
from .factors_input import *
from .factors_output import *
from .ml import *
from .analysis import *
from .params import *

# 私有名（下划线开头，`import *` 不会带上）——外部脚本以 config._MKT / config._DATA_ROOT /
# config._FACTORS 直接访问（带 # noqa: SLF001），故显式再导出保持兼容。
from .base import _DATA_ROOT, _MKT, _FACTORS, _REPO_ROOT, _IS_MINUTE_DQUANT
