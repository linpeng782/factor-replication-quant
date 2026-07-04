"""
[第 2 层｜因子生产] 因子产线的原料输入 + 三阶段产物 + 评估输出。

分两块：
  (A) 原料：逐股原始日频行情 / 复权因子 / 基本面 PIT / 分钟数据（factor 计算的输入）
  (B) 产物：raw→cleaned→neu 三阶段基址 + alpha158 + OUTPUT_DIR（factor 计算的产出）

⚠️【生产隔离轴】MINUTE=dquant 时，三阶段产物 + output 整体重定向到并行 -dquant 目录
   （隔离，不污染 rq 基线）。见 base.py 开关说明。
"""

from .base import (
    _DATA_ROOT,
    _MKT,
    _FACTORS,
    _REPO_ROOT,
    ALPHA158_BACKEND,
    FUNDAMENTAL_BACKEND,
    _IS_MINUTE_DQUANT,
)

__all__ = [
    # (B) 因子三阶段产物 + 评估输出
    "RAW_FACTOR_BASE",
    "CLEANED_FACTOR_BASE",
    "NEU_FACTOR_BASE",
    "ALPHA158_RAW_BASE",
    "RET20_PANEL_PATH",
    "OUTPUT_DIR",
    # (A) 原料：逐股日频行情
    "RAW_OHLCV_DIR",
    "EX_FACTORS_DIR",
    "INSTRUMENTS_INFO_PATH",
    "TRADING_CALENDAR_PATH",
    # (A) 原料：基本面
    "FUNDAMENTALS_DIR",
    "INDUSTRY_PANEL_ZX_DQUANT_PATH",
    # (A) 原料：分钟
    "MINUTE_DATA_DIR",
    "MINUTE_RAW_DIR",
    "MINUTE_EX_FACTORS_DIR",
    "INTERMEDIATE_CACHE_DIR",
]

# ============================================================
# (B) 因子三阶段产物 <stage>/<source>/<group>/ + 评估输出
# ============================================================
# 按阶段(raw/cleaned/neu) × 来源(cxl/kysec/founder/...) × 分组分桶；
# namespace=<source>/<group> 由 spec 路径推导（见 spec_resolver.resolve_namespace）。
# ⚠️【生产隔离轴】MINUTE=dquant 时整体重定向到并行 -dquant 目录。
RAW_FACTOR_BASE = _FACTORS / ("raw-dquant" if _IS_MINUTE_DQUANT else "raw")
CLEANED_FACTOR_BASE = _FACTORS / ("cleaned-dquant" if _IS_MINUTE_DQUANT else "cleaned")
NEU_FACTOR_BASE = _FACTORS / ("neu-dquant" if _IS_MINUTE_DQUANT else "neu")

# alpha158 raw 产物（消费轴，随 ALPHA158_BACKEND）：dquant 期写并行目录 alpha158-dquant/，
# 避免覆盖 rq 基准（验毕迁移后可改回 alpha158/）
ALPHA158_RAW_BASE = (
    _FACTORS / "raw" / "alpha158-dquant" if ALPHA158_BACKEND == "dquant"
    else _FACTORS / "raw" / "alpha158"
)
# 辅助面板（跨因子共享，不属三阶段产物，存 helpers/ 下）
# Ret20：20日后复权收益面板（宽表），APM 截面回归去动量用（scripts/build_ret20_panel.py 产出）
RET20_PANEL_PATH = _FACTORS / "helpers" / "ret20_panel.parquet"

# 项目内输出目录（相对项目代码，不受 _DATA_ROOT 影响）：报告、图片等评估输出。
# ⚠️【生产隔离轴】MINUTE=dquant 时改用 output-dquant/
OUTPUT_DIR = _REPO_ROOT / ("output-dquant" if _IS_MINUTE_DQUANT else "output")

# ============================================================
# (A) 原料：逐股原始日频行情（alpha158 生产原料 + 复权因子，消费轴随 ALPHA158_BACKEND）
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
# (A) 原料：基本面 PIT 基础数据（cxl 生产原料，消费轴随 FUNDAMENTAL_BACKEND）
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
# (A) 原料：分钟级因子数据（生产隔离轴，随 MINUTE_BACKEND）
# ============================================================
# 后复权 per-stock 1m parquet 目录（旧版，烤死复权；迁移期保留作对齐基准）
MINUTE_DATA_DIR = _MKT / "minute" / "stock_data_1m_post"
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
