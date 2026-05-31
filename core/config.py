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
# 按「角色」组织（非项目名）：所有评估输入数据归于 market-data/，由 stock-data-fetching 维护。
_MKT = _DATA_ROOT / "market-data"

COMBO_MASK_PATH = _MKT / "masks/combo_mask_long.parquet"
NEW_STOCK_MASK_PATH = _MKT / "masks/new_stock_mask_long.parquet"

VWAP_POST_PATH = _MKT / "prices/vwap_post.parquet"
# PIT canonical vwap 宽表面板（build_labels.py 副产）；取代 VWAP_POST_PATH 作 forward_returns 源
VWAP_PANEL_PATH = _MKT / "prices/vwap_panel.parquet"
# 预算的 forward_return_{N}d.parquet；评估直读，缺失 horizon 回退 vwap_panel 现算
LABELS_DIR = _MKT / "labels"

# 行业 + 市值面板（中性化用，stock-data-fetching 产出）
INDUSTRY_PANEL_ZX_PATH = _MKT / "industry/industry_panel_zx.parquet"
MARKET_CAP_PANEL_PATH = _MKT / "market_cap/market_cap_panel.parquet"

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

# 分钟级因子用：原始 per-stock 分钟 parquet 目录（用户日更）+ 中间产物缓存目录
MINUTE_DATA_DIR = _MKT / "minute" / "stock_data_1m_post"
INTERMEDIATE_CACHE_DIR = _DATA_ROOT / "factor-replication/intermediate_cache"

# ==================== 默认 fetch / 评估区间 ====================
# fetch 区间（panel 落盘窗口）— 提前到 2010 给所有因子留 ≥4 年 warm-up，
# 使得最复杂的 8 期 PIT 滚动因子（npf_mrq_sue8 / np_*_rank 等）能从 2014-01-02 起有有效值
DEFAULT_START_DATE = "20100101"
DEFAULT_END_DATE = "20260527"

# 评估区间（IC / ICIR 计算窗口）— 保持原值，跟历史对齐数字（PROGRESS §四 + 2026-05-28 对齐报告）可比
DEFAULT_EVAL_START_DATE = "20160101"
DEFAULT_EVAL_END_DATE = "20251231"
