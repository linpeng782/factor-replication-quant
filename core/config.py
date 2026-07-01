"""
因子复现项目配置

包含 mask / vwap 等预计算数据的路径配置。

数据根路径由环境变量 FACTOR_REPL_DATA_ROOT 控制（默认 NFS 远端路径）。
本地跑时设置 `export FACTOR_REPL_DATA_ROOT=~/factor-repl-data`，所有路径自动重定向。
"""

import os
from pathlib import Path

# ==================== 数据根路径（可通过环境变量覆盖） ====================
_DATA_ROOT = Path(
    os.environ.get(
        "FACTOR_REPL_DATA_ROOT",
        "/nfs/ofs-prediction/peterzhenglinpeng",
    )
)

# ==================== 输入数据：market-data/ ====================
# 按「角色」组织（非项目名）：所有评估输入数据归于 market-data/，由 data_fetching/ 维护。
_MKT = _DATA_ROOT / "market-data"

# mask 路径随主开关 DATA_BACKEND 切换（属「消费轴」，与 ALPHA158/ML_HT 同）：
#   rq     → market-data/masks/                    （旧 mask，历史基准）
#   dquant → backtest_engine/cache_dir_dquant/      （dquant 日更 mask，与 combo_mask 同步推进到最新交易日）
# 两份 schema 完全一致（order_book_id/datetime/is_st/is_suspended/is_limit_up/is_new_stock），
# 仅日期覆盖与取值不同 → 指向不同文件即可，不重建 schema。env MASK_BACKEND 可单独覆盖（向后兼容）。
_MASK_BACKEND = os.environ.get("MASK_BACKEND", os.environ.get("DATA_BACKEND", "dquant"))
MASK_BACKEND = _MASK_BACKEND                      # 公开：消费者用此
_MASK_DIR = (
    _DATA_ROOT / "backtest_engine" / "cache_dir_dquant" if _MASK_BACKEND == "dquant"
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

# ==================== 项目内输出目录 ====================
# 报告、图片等评估输出（项目代码相对路径，不受 _DATA_ROOT 影响）
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ==================== 因子产出：factors/<stage>/<source>/<group>/ ====================
# 按「阶段」(raw/cleaned/neu) × 「来源」(cxl/kysec/founder/...) × 「分组」(研报/系列) 分桶。
# namespace=<source>/<group> 由 spec 路径推导（见 spec_resolver.resolve_namespace）；最终路径：
#   <BASE>/<source>/<group>/<factor>.parquet
_FACTORS = _DATA_ROOT / "factors"
RAW_FACTOR_BASE = _FACTORS / "raw"           # 原始因子（spec 引擎/批量库产出）
CLEANED_FACTOR_BASE = _FACTORS / "cleaned"   # 清洗后（MAD+zscore+mask）
NEU_FACTOR_BASE = _FACTORS / "neu"           # 行业市值中性化后（生产用版本）

# ==================== 数据后端主开关（统一入口）====================
# env DATA_BACKEND=dquant (默认) | rq —— 一处切换 ML「消费侧」数据后端。
#   它只驱动【消费轴】：ALPHA158_DATA_BACKEND + ML_HT_BACKEND（读哪套 alpha158、产物落 ht/ht_dquant）。
#   细分 env 若【显式设置】则覆盖主开关（向后兼容：老脚本设 ALPHA158_DATA_BACKEND=rq 仍生效）。
# ⚠️ MINUTE_DATA_BACKEND 故意【不随】主开关：它是「生产隔离轴」，dquant 会把整个 RAW_FACTOR_BASE
#    重定向到 factors/raw-dquant，使 alpha158-dquant/cxl 等（在 factors/raw 下）发现不到——
#    两轴语义不同，强行合并会静默破坏因子发现。分钟因子生产仍需单独 export MINUTE_DATA_BACKEND=dquant。
_DATA_BACKEND = os.environ.get("DATA_BACKEND", "dquant")
DATA_BACKEND = _DATA_BACKEND                      # 公开：主开关解析值

# alpha158 数据后端开关: env ALPHA158_DATA_BACKEND 覆盖；缺省继承主开关 DATA_BACKEND
#   rq     → rq 原始 OHLCV + rq 复权因子（既有路径,所有历史产物基准）
#   dquant → jy/dquant 原始 OHLCV + jy 复权因子（已验证投入生产，现为默认）
_ALPHA158_BACKEND = os.environ.get("ALPHA158_DATA_BACKEND", _DATA_BACKEND)
ALPHA158_BACKEND = _ALPHA158_BACKEND              # 公开：消费者用此，勿再各自读 os.environ

# alpha158 raw 产物：为避免 dquant 迁移期覆盖 rq 基准，
# dquant 后端时写到并行目录 factors/raw/alpha158-dquant/（验毕迁移后可改回 alpha158/）
ALPHA158_RAW_BASE = (
    _FACTORS / "raw" / "alpha158-dquant" if _ALPHA158_BACKEND == "dquant"
    else _FACTORS / "raw" / "alpha158"
)

# ml_ht 训练流水线后端开关: env ML_HT_BACKEND 覆盖；缺省继承主开关 DATA_BACKEND
#   rq     → 读 factors/raw/alpha158/ + 产 ml/ht/          (历史 rq 训练基准)
#   dquant → 读 factors/raw/alpha158-dquant/ + 产 ml/ht_dquant/  (已验证，现为默认)
# 与 ALPHA158 同属【消费轴】，由主开关统一驱动；显式 env 仍可单独覆盖（向后兼容）。
# 一次切换: export DATA_BACKEND=rq（回退旧 rq 基准）；细粒度仍可单独 export ALPHA158_DATA_BACKEND / ML_HT_BACKEND
_ML_HT_BACKEND = os.environ.get("ML_HT_BACKEND", _DATA_BACKEND)
ML_HT_BACKEND = _ML_HT_BACKEND                    # 公开：消费者用此，勿再各自读 os.environ

# 辅助面板（跨因子共享；不属于三阶段产物，存 helpers/ 下）
# Ret20：20日后复权收益面板（宽表），APM 截面回归去动量用（scripts/build_ret20_panel.py 产出）
RET20_PANEL_PATH = _FACTORS / "helpers" / "ret20_panel.parquet"

# ==================== 因子分析产物：factor-inventory/ ====================
# 总账(inventory) / IC序列矩阵(ic_series) / 相关性(correlation) / 对比(comparison) 等
# 数值分析产物——代码与数据分离，统一落数据根下（受 FACTOR_REPL_DATA_ROOT 控制）。
INVENTORY_ROOT = _DATA_ROOT / "factor-inventory"

# ==================== ML 训练流水线产物：ml/ ====================
# LightGBM 因子合成（repo 顶层 ml/ 包产出）：模型 / 预测 / (可选)数据集。
# 代码线(repo 的 ml/) 与 数据线(此处) 分离，与 factors/ factor-inventory/ 平级。
ML_ROOT = _DATA_ROOT / "ml"
ML_HT_BASE = ML_ROOT / ("ht_dquant" if _ML_HT_BACKEND == "dquant" else "ht")  # rq→ht/  dquant→ht_dquant/
ML_MODELS_DIR = ML_ROOT / "models"            # lgbm 模型 + 超参 + RobustZScore 尺子
ML_PREDICTIONS_DIR = ML_ROOT / "predictions"  # ŷ 面板 + test 评估(IC)
ML_DATASETS_DIR = ML_ROOT / "datasets"        # (可选) train/valid/test 矩阵，便于复跑
ML_SIGNALS_DIR = ML_ROOT / "signals"          # 回测可读信号：每日排序选股名单 txt（export_signal.py 产出）
# 日志落在【repo 内 ml/logs/】（方便查看，受 .gitignore 排除），非数据根
ML_LOGS_DIR = Path(__file__).parent.parent / "ml" / "logs"   # 每次 run 的训练日志

# 分钟级因子：后复权 per-stock 1m parquet 目录（旧版，烤死复权；迁移期保留作对齐基准）
MINUTE_DATA_DIR = _MKT / "minute" / "stock_data_1m_post"
# ==================== 分钟因子数据后端开关（rq → dquant 迁移；与 _ALPHA158_BACKEND 完全独立）====================
# env MINUTE_DATA_BACKEND=rq (默认) | dquant
#   rq     → minute/raw (rqdatac 源) + rq ex_cum_factor + intermediate-cache + factors/{raw,cleaned,neu} + output/
#   dquant → minute-dquant/raw (dquant source="rq",与 rq raw bit 同) + jy adjfactor(与 alpha158-dquant 一致)
#            + intermediate-cache-dquant + factors/{raw,cleaned,neu}-dquant + output-dquant/
# 不设此开关时，下列所有路径 byte 级等同历史 rq 产物（零污染）；dquant 端全部走并行目录，互不覆盖。
# ⚠️【生产隔离轴】不随主开关 DATA_BACKEND：dquant 会重定向整个 RAW_FACTOR_BASE→factors/raw-dquant，
#    与「消费轴」语义不同（见顶部主开关说明）。默认恒为 rq，需显式 export MINUTE_DATA_BACKEND=dquant。
_MINUTE_BACKEND = os.environ.get("MINUTE_DATA_BACKEND", "rq")
MINUTE_BACKEND = _MINUTE_BACKEND                  # 公开
_IS_MINUTE_DQUANT = _MINUTE_BACKEND == "dquant"

# 分钟原始（不复权）按日分片目录（新版：minute/raw/<YYYY-MM-DD>.parquet，全股一日一文件）
# 复权在读时实时算（core.minute_data.load_adjusted_minute_window）。见 docs/minute_incremental_design.md
# dquant 端用平行的 minute-dquant/ 树（连 minute_first_appearance.parquet 一并隔离）。
MINUTE_RAW_DIR = (_MKT / "minute-dquant" / "raw") if _IS_MINUTE_DQUANT else (_MKT / "minute" / "raw")
# 分钟读时复权因子目录：dquant 端用 jy adjfactor（与 alpha158-dquant 同源），rq 端用 rq ex_cum_factor。
# 独立于全局 EX_FACTORS_DIR（受 _ALPHA158_BACKEND 控制），使分钟复权口径不被 alpha158 后端牵连。
MINUTE_EX_FACTORS_DIR = (
    _MKT / "daily_dquant" / "stock-ex-factors-jy" if _IS_MINUTE_DQUANT
    else _MKT / "daily" / "stock-ex-factors"
)
# 分钟→日频特征 中间缓存（可再生；superset 缓存身份 hash 不含数据后端 → dquant 必须并行目录防覆盖 golden）
INTERMEDIATE_CACHE_DIR = _DATA_ROOT / ("intermediate-cache-dquant" if _IS_MINUTE_DQUANT else "intermediate-cache")

# dquant 分钟后端：因子三阶段 + 评估产物全部重定向到并行目录（隔离，不污染 rq 基线）。
if _IS_MINUTE_DQUANT:
    RAW_FACTOR_BASE = _FACTORS / "raw-dquant"
    CLEANED_FACTOR_BASE = _FACTORS / "cleaned-dquant"
    NEU_FACTOR_BASE = _FACTORS / "neu-dquant"
    OUTPUT_DIR = Path(__file__).parent.parent / "output-dquant"

# ==================== 逐股原始日频行情（alpha158 生产原料 + 复权因子） ====================
# 磁盘只存「原始价(不复权) + 稀疏 cum_factor」，复权在读时实时算（core.producers.alpha158.loader）。
# 由 data_fetching/产出/日更。daily/ 与 minute/ 对称。
# 后端开关 _ALPHA158_BACKEND 已在前面定义（与 ALPHA158_RAW_BASE 同区,便于集中维护）。
if _ALPHA158_BACKEND == "dquant":
    _RAW_OHLCV_ROOT = _MKT / "daily_dquant"
    RAW_OHLCV_DIR = _RAW_OHLCV_ROOT / "stock-ohlcv-dquant"      # 逐股原始日频 OHLCV (dquant/jy 源)
    EX_FACTORS_DIR = _RAW_OHLCV_ROOT / "stock-ex-factors-jy"   # 逐股稀疏复权因子 (jy adjfactor)
else:
    _RAW_OHLCV_ROOT = _MKT / "daily"
    RAW_OHLCV_DIR = _RAW_OHLCV_ROOT / "stock-ohlcv"             # 逐股原始日频 OHLCV (rq 源)
    EX_FACTORS_DIR = _RAW_OHLCV_ROOT / "stock-ex-factors"      # 逐股稀疏复权因子 (rq ex_factor)
INSTRUMENTS_INFO_PATH = _RAW_OHLCV_ROOT / "instruments_info.parquet"   # 股票基本信息（待补）
TRADING_CALENDAR_PATH = _RAW_OHLCV_ROOT / "trading_calendar.parquet"   # 交易日历（待补）

# ==================== 基本面 PIT 基础数据（cxl 生产原料） ====================
# get_factor 点位字段（mrq/ttm/估值）的每日快照，按字段 WIDE（date×stock）。
# 每日 append 当天快照、历史永不改写 = as-first-reported 冻结 PIT（防漂移/前视）。
# 由 data_fetching/fundamentals.py 产出/日更；fetch 算子 api=get_factor 改读此处。
# 见 docs/cxl_fundamental_incremental_design.md
FUNDAMENTALS_DIR = _MKT / "fundamentals"                 # market-data/fundamentals/<field>.parquet

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
