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

COMBO_MASK_PATH = _MKT / "masks/combo_mask_long.parquet"
NEW_STOCK_MASK_PATH = _MKT / "masks/new_stock_mask_long.parquet"

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
ML_MODELS_DIR = ML_ROOT / "models"            # lgbm 模型 + 超参 + RobustZScore 尺子
ML_PREDICTIONS_DIR = ML_ROOT / "predictions"  # ŷ 面板 + test 评估(IC)
ML_DATASETS_DIR = ML_ROOT / "datasets"        # (可选) train/valid/test 矩阵，便于复跑
ML_SIGNALS_DIR = ML_ROOT / "signals"          # 回测可读信号：每日排序选股名单 txt（export_signal.py 产出）
# 日志落在【repo 内 ml/logs/】（方便查看，受 .gitignore 排除），非数据根
ML_LOGS_DIR = Path(__file__).parent.parent / "ml" / "logs"   # 每次 run 的训练日志

# 分钟级因子：后复权 per-stock 1m parquet 目录（旧版，烤死复权；迁移期保留作对齐基准）
MINUTE_DATA_DIR = _MKT / "minute" / "stock_data_1m_post"
# 分钟原始（不复权）按日分片目录（新版：minute/raw/<YYYY-MM-DD>.parquet，全股一日一文件）
# 复权在读时实时算（core.minute_data.load_adjusted_minute_window）。见 docs/minute_incremental_design.md
MINUTE_RAW_DIR = _MKT / "minute" / "raw"
# 分钟→日频特征 中间缓存（可再生；market-data/factors/factor-inventory 的同级兄弟）
INTERMEDIATE_CACHE_DIR = _DATA_ROOT / "intermediate-cache"

# ==================== 逐股原始日频行情（alpha158 生产原料 + 复权因子） ====================
# 磁盘只存「原始价(不复权) + 稀疏 cum_factor」，复权在读时实时算（core.producers.alpha158.loader）。
# 由 data_fetching/产出/日更。daily/ 与 minute/ 对称。
_RAW_OHLCV_ROOT = _MKT / "daily"
RAW_OHLCV_DIR = _RAW_OHLCV_ROOT / "stock-ohlcv"          # 逐股原始日频 OHLCV
EX_FACTORS_DIR = _RAW_OHLCV_ROOT / "stock-ex-factors"    # 逐股稀疏复权因子（日频/分钟共用）
INSTRUMENTS_INFO_PATH = _RAW_OHLCV_ROOT / "instruments_info.parquet"   # 股票基本信息（待补）
TRADING_CALENDAR_PATH = _RAW_OHLCV_ROOT / "trading_calendar.parquet"   # 交易日历（待补）

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
