"""
[第 0 层｜地基] 数据根 + 后端开关控制台。

★ 全项目所有后端开关集中在此解析完毕；上层模块（market_data / factors / ml / ...）
  只消费这里解析好的结果，不再各自读 os.environ。

两类语义（务必分清）：
  【消费轴】   跟随主开关 DATA_BACKEND，决定"读哪套数据、产物落哪"：
              ALPHA158 / ML_HT / FUNDAMENTAL / MASK。
              一处 export DATA_BACKEND=rq 即整体回退旧 rq 基准；
              细分 env 若【显式设置】则单独覆盖（向后兼容老脚本）。
  【生产隔离轴】 MINUTE —— 故意【不随】主开关。
              dquant 会把整个 factors/{raw,cleaned,neu} + output 重定向到并行 -dquant 目录；
              与消费轴语义不同，强行合并会静默破坏因子发现。

数据根路径由环境变量 FACTOR_REPL_DATA_ROOT 控制（默认 NFS 远端路径）。
本地跑时设置 `export FACTOR_REPL_DATA_ROOT=~/factor-repl-data`，所有路径自动重定向。
"""

import os
from pathlib import Path

__all__ = [
    "DATA_BACKEND",
    "ALPHA158_BACKEND",
    "ML_HT_BACKEND",
    "FUNDAMENTAL_BACKEND",
    "MASK_BACKEND",
    "MINUTE_BACKEND",
]

# ==================== 数据根（可通过环境变量覆盖） ====================
_DATA_ROOT = Path(
    os.environ.get(
        "FACTOR_REPL_DATA_ROOT",
        "/nfs/ofs-prediction/peterzhenglinpeng",
    )
)
# 数据根下两个「角色桶」根（按角色组织，非项目名）：
_MKT = _DATA_ROOT / "market-data"     # 所有评估输入数据，由 data_fetching/ 维护
_FACTORS = _DATA_ROOT / "factors"     # 因子三阶段产出根

# 项目代码仓库根（config/ 的上一级）；供 OUTPUT_DIR / ML_LOGS_DIR 等「项目相对路径」用。
_REPO_ROOT = Path(__file__).parent.parent

# ==================== 后端开关控制台 ====================
# —— 主开关：env DATA_BACKEND=dquant (默认) | rq ——
DATA_BACKEND = os.environ.get("DATA_BACKEND", "dquant")

# —— 消费轴子开关（缺省继承主开关；显式 env 覆盖，勿再各自读 os.environ）——
# alpha158 原料后端：rq → rq OHLCV + rq 复权因子；dquant → jy/dquant OHLCV + jy 复权（已投产，默认）
ALPHA158_BACKEND = os.environ.get("ALPHA158_DATA_BACKEND", DATA_BACKEND)
# ml_ht 训练流水线后端：rq → 读 alpha158/ 产 ht/；dquant → 读 alpha158-dquant/ 产 ht_dquant/（默认）
ML_HT_BACKEND = os.environ.get("ML_HT_BACKEND", DATA_BACKEND)
# 基本面后端：rq → fundamentals/ + rqdatac 行业/universe；dquant → fundamentals-dquant/ + 本地行业(零 rqdatac)
FUNDAMENTAL_BACKEND = os.environ.get("FUNDAMENTAL_DATA_BACKEND", DATA_BACKEND)
# mask 后端：rq → market-data/masks/（历史基准）；dquant → backtest_engine/cache_dir_dquant/（日更）
# 两份 schema 完全一致（order_book_id/datetime/is_st/is_suspended/is_limit_up/is_new_stock），
# 仅日期覆盖与取值不同 → 指向不同文件即可，不重建 schema。
MASK_BACKEND = os.environ.get("MASK_BACKEND", DATA_BACKEND)

# —— 生产隔离轴开关：env MINUTE_DATA_BACKEND=rq (默认) | dquant，独立于主开关 ——
# rq     → minute/raw + rq ex_cum_factor + intermediate-cache + factors/{raw,cleaned,neu} + output/
# dquant → minute-dquant/raw(与 rq raw bit 同) + jy adjfactor + intermediate-cache-dquant
#          + factors/{raw,cleaned,neu}-dquant + output-dquant/（全并行目录，零污染 rq 基线）
MINUTE_BACKEND = os.environ.get("MINUTE_DATA_BACKEND", "rq")
_IS_MINUTE_DQUANT = MINUTE_BACKEND == "dquant"
