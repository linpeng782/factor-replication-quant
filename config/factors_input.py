"""
[第 2 层｜因子原料] 因子产线的输入数据：日频行情 / 复权因子 / 基本面 PIT / 分钟数据。

数据流：原料（本文件）→ 算子图算因子 → 产物（factors_output.py）。
按数据流顺序排列：日频 → 基本面 → 分钟。

后端归属：
  日频行情   随 ALPHA158_BACKEND（消费轴）
  基本面     随 FUNDAMENTAL_BACKEND（消费轴）
  分钟       随 MINUTE_BACKEND（生产隔离轴）
"""

from .base import (
    _DATA_ROOT,
    _MKT,
    ALPHA158_BACKEND,
    FUNDAMENTAL_BACKEND,
    _IS_MINUTE_DQUANT,
)

__all__ = [
    # 日频行情原料
    "RAW_OHLCV_DIR",
    "EX_FACTORS_DIR",
    "INSTRUMENTS_INFO_PATH",
    "TRADING_CALENDAR_PATH",
    # 基本面原料
    "FUNDAMENTALS_DIR",
    "INDUSTRY_PANEL_ZX_DQUANT_PATH",
    # 分钟原料
    "MINUTE_RAW_DIR",
    "MINUTE_EX_FACTORS_DIR",
    "INTERMEDIATE_CACHE_DIR",
]

# ============================================================
# 日频行情原料（alpha158 生产原料 + 复权因子，消费轴随 ALPHA158_BACKEND）
# ============================================================
# 磁盘只存「原始价(不复权) + 稀疏 cum_factor」，复权读时实时算（core.producers.alpha158.loader）。
# 由 data_fetching/ 产出/日更。daily/ 与 minute/ 对称。
if ALPHA158_BACKEND == "dquant":
    _RAW_OHLCV_ROOT = _MKT / "daily_dquant"
    RAW_OHLCV_DIR = _RAW_OHLCV_ROOT / "stock-ohlcv-dquant"     # 逐股原始日频 OHLCV (dquant/jy 源)
    EX_FACTORS_DIR = _RAW_OHLCV_ROOT / "stock-ex-factors-jy"   # 逐股稀疏复权因子 (jy adjfactor)
else:
    _RAW_OHLCV_ROOT = _MKT / "daily"
    RAW_OHLCV_DIR = _RAW_OHLCV_ROOT / "stock-ohlcv"            # 逐股原始日频 OHLCV (rq 源)
    EX_FACTORS_DIR = _RAW_OHLCV_ROOT / "stock-ex-factors"      # 逐股稀疏复权因子 (rq ex_factor)
INSTRUMENTS_INFO_PATH = _RAW_OHLCV_ROOT / "instruments_info.parquet"   # 股票基本信息（待补）
TRADING_CALENDAR_PATH = _RAW_OHLCV_ROOT / "trading_calendar.parquet"   # 交易日历（待补）

# ============================================================
# 基本面 PIT 基础数据（cxl 生产原料，消费轴随 FUNDAMENTAL_BACKEND）
# ============================================================
# get_factor 点位字段（mrq/ttm/估值）每日快照，按字段 WIDE(date×stock)；
# 每日 append 当天快照、历史永不改写 = as-first-reported 冻结 PIT（防漂移/前视）。
# 由 data_fetching/fundamentals.py 产出/日更。见 docs/cxl_fundamental_incremental_design.md
#   rq     → market-data/fundamentals/ + rqdatac 行业/universe + 因子落 factors/raw/cxl/
#   dquant → market-data/fundamentals-dquant/（fundamentals_dquant.py 产出）+ 本地行业面板
#            + universe 取面板列（零 rqdatac）+ namespace cxl→cxl-dquant（四处产物自动并行隔离）
FUNDAMENTALS_DIR = _MKT / ("fundamentals-dquant" if FUNDAMENTAL_BACKEND == "dquant" else "fundamentals")
# dquant 中信一级行业日频宽面板（industry_dquant.py 产出；因子生产 fetch custom 用。
# 评估中性化仍用 INDUSTRY_PANEL_ZX_PATH，消费轴不受此开关影响）
INDUSTRY_PANEL_ZX_DQUANT_PATH = _MKT / "industry-dquant/industry_panel_zx_dquant.parquet"

# ============================================================
# 分钟级因子数据（生产隔离轴，随 MINUTE_BACKEND）
# ============================================================
# 分钟原始（不复权）按日分片：minute/raw/<YYYY-MM-DD>.parquet（全股一日一文件），复权读时实时算
# （core.minute_data.load_adjusted_minute_window）。dquant 端用平行 minute-dquant/ 树
# （连 minute_first_appearance.parquet 一并隔离）。见 docs/minute_incremental_design.md
MINUTE_RAW_DIR = (_MKT / "minute-dquant" / "raw") if _IS_MINUTE_DQUANT else (_MKT / "minute" / "raw")
# 分钟读时复权因子：dquant 用 jy adjfactor（与 alpha158-dquant 同源），rq 用 rq ex_cum_factor。
# 独立于全局 EX_FACTORS_DIR（受 ALPHA158_BACKEND 控制），使分钟复权口径不被 alpha158 后端牵连。
MINUTE_EX_FACTORS_DIR = (
    _MKT / "daily_dquant" / "stock-ex-factors-jy" if _IS_MINUTE_DQUANT
    else _MKT / "daily" / "stock-ex-factors"
)
# 分钟→日频特征中间缓存（可再生；superset 缓存身份 hash 不含数据后端 → dquant 必须并行目录防覆盖 golden）
INTERMEDIATE_CACHE_DIR = _DATA_ROOT / ("intermediate-cache-dquant" if _IS_MINUTE_DQUANT else "intermediate-cache")
