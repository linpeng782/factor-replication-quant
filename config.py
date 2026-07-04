"""
因子复现项目配置

包含 mask / vwap 等预计算数据的路径配置。

数据根路径由环境变量 FACTOR_REPL_DATA_ROOT 控制（默认 NFS 远端路径）。
本地跑时设置 `export FACTOR_REPL_DATA_ROOT=~/factor-repl-data`，所有路径自动重定向。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
本文件结构（单向流水线，自上而下，每个常量只赋值一次，绝不回头覆盖）：
  第 1 段  数据根          _DATA_ROOT（env FACTOR_REPL_DATA_ROOT 覆盖）+ 两个角色桶根
  第 2 段  后端开关控制台   5 个开关集中解析，分「消费轴 / 生产隔离轴」两组
  第 3 段  路径派生         全部路径常量，用第 2 段解析好的开关一次性派生
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
from pathlib import Path

# ==================== 第 1 段：数据根（可通过环境变量覆盖） ====================
_DATA_ROOT = Path(
    os.environ.get(
        "FACTOR_REPL_DATA_ROOT",
        "/nfs/ofs-prediction/peterzhenglinpeng",
    )
)
# 数据根下两个「角色桶」根（按角色组织，非项目名）：
_MKT = _DATA_ROOT / "market-data"     # 所有评估输入数据，由 data_fetching/ 维护
_FACTORS = _DATA_ROOT / "factors"     # 因子三阶段产出根

# ==================== 第 2 段：后端开关控制台 ====================
# ★ 全项目所有后端开关集中在此解析完毕；下方第 3 段只消费这些结果，不再读 os.environ。
#
# 两类语义（务必分清）：
#   【消费轴】   跟随主开关 DATA_BACKEND，决定"读哪套数据、产物落哪"：
#               ALPHA158 / ML_HT / FUNDAMENTAL / MASK。
#               一处 export DATA_BACKEND=rq 即整体回退旧 rq 基准；
#               细分 env 若【显式设置】则单独覆盖（向后兼容老脚本）。
#   【生产隔离轴】 MINUTE —— 故意【不随】主开关，默认恒为 rq。
#               dquant 会把整个 factors/{raw,cleaned,neu} + output 重定向到并行 -dquant 目录；
#               与消费轴语义不同，强行合并会静默破坏因子发现，需显式 export MINUTE_DATA_BACKEND=dquant。

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

# ==================== 第 3 段：路径派生 ====================
# 下方每个常量只赋值一次；受开关影响的用三元/分支就地决定，绝不"先定义后覆盖"。

# ---------- market-data/：评估输入数据 ----------
# mask（消费轴，随 MASK_BACKEND）：两份 schema 一致，仅指向不同文件
_MASK_DIR = (
    _DATA_ROOT / "backtest_engine" / "cache_dir_dquant" if MASK_BACKEND == "dquant"
    else _MKT / "masks"
)
COMBO_MASK_PATH = _MASK_DIR / "combo_mask_long.parquet"
NEW_STOCK_MASK_PATH = _MASK_DIR / "new_stock_mask_long.parquet"

VWAP_POST_PATH = _MKT / "prices/vwap_post.parquet"
# PIT canonical vwap 宽表面板（build_labels.py 副产）；取代 VWAP_POST_PATH 作 forward_returns 源
VWAP_PANEL_PATH = _MKT / "prices/vwap_panel.parquet"
# 预算的 forward_return_{N}d.parquet；评估直读，缺失 horizon 回退 vwap_panel 现算
LABELS_DIR = _MKT / "labels"

# 行业 + 市值面板（中性化用，data_fetching/ 产出）
INDUSTRY_PANEL_ZX_PATH = _MKT / "industry/industry_panel_zx.parquet"
MARKET_CAP_PANEL_PATH = _MKT / "market_cap/market_cap_panel.parquet"
# 中信一级行业指数日收益面板（T×33；联合动量因子用，data_fetching/industry_index.py 产出）
INDUSTRY_INDEX_RETURN_PATH = _MKT / "industry/industry_index_return.parquet"

# 指数分段收益（APM 回归的市场参照序列，data_fetching/index_segments.py 产出）
INDEX_DIR = _MKT / "index"
# 000985 中证全指日频四段收益：ret_overnight_idx/ret_am_idx/ret_pm_idx/ret_pm_late_idx
INDEX_SEGMENTS_PATH = INDEX_DIR / "000985_segments.parquet"

# ---------- factors/：因子三阶段产出 <stage>/<source>/<group>/ ----------
# 按阶段(raw/cleaned/neu) × 来源(cxl/kysec/founder/...) × 分组分桶；
# namespace=<source>/<group> 由 spec 路径推导（见 spec_resolver.resolve_namespace）。
# ⚠️【生产隔离轴】MINUTE=dquant 时，三阶段 + output 整体重定向到并行 -dquant 目录（隔离，不污染 rq 基线）。
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

# ---------- 项目内输出目录（相对项目代码，不受 _DATA_ROOT 影响） ----------
# 报告、图片等评估输出；⚠️【生产隔离轴】MINUTE=dquant 时改用 output-dquant/
OUTPUT_DIR = Path(__file__).parent / ("output-dquant" if _IS_MINUTE_DQUANT else "output")

# ---------- factor-inventory/：因子分析产物 ----------
# 总账(inventory)/IC序列矩阵(ic_series)/相关性(correlation)/对比(comparison) 等数值分析产物——
# 代码与数据分离，统一落数据根下（受 FACTOR_REPL_DATA_ROOT 控制）。
INVENTORY_ROOT = _DATA_ROOT / "factor-inventory"

# ---------- ml/：ML 训练流水线产物 ----------
# LightGBM 因子合成（repo 顶层 ml/ 包产出）：代码线(repo 的 ml/) 与数据线(此处)分离。
ML_ROOT = _DATA_ROOT / "ml"
ML_HT_BASE = ML_ROOT / ("ht_dquant" if ML_HT_BACKEND == "dquant" else "ht")  # rq→ht/  dquant→ht_dquant/
ML_MODELS_DIR = ML_ROOT / "models"            # lgbm 模型 + 超参 + RobustZScore 尺子
ML_PREDICTIONS_DIR = ML_ROOT / "predictions"  # ŷ 面板 + test 评估(IC)
ML_DATASETS_DIR = ML_ROOT / "datasets"        # (可选) train/valid/test 矩阵，便于复跑
ML_SIGNALS_DIR = ML_ROOT / "signals"          # 回测可读信号：每日排序选股名单 txt（export_signal.py 产出）
# 日志落在【repo 内 ml/logs/】（方便查看，受 .gitignore 排除），非数据根
ML_LOGS_DIR = Path(__file__).parent / "ml" / "logs"   # 每次 run 的训练日志

# ---------- minute/：分钟级因子数据（生产隔离轴，随 MINUTE_BACKEND） ----------
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

# ---------- 逐股原始日频行情（alpha158 生产原料 + 复权因子，消费轴随 ALPHA158_BACKEND） ----------
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

# ---------- 基本面 PIT 基础数据（cxl 生产原料，消费轴随 FUNDAMENTAL_BACKEND） ----------
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
