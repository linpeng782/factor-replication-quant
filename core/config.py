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

# ==================== 外部预计算数据路径 ====================
# 这些路径指向 my-alpha-engine / backtest_engine 已生产的数据

COMBO_MASK_PATH = _DATA_ROOT / "backtest_engine/cache_dir/combo_mask_long.parquet"

NEW_STOCK_MASK_PATH = _DATA_ROOT / "backtest_engine/cache_dir/new_stock_mask_long.parquet"

VWAP_POST_PATH = _DATA_ROOT / "backtest_engine/cache_dir/vwap_post.parquet"
# PIT canonical vwap 宽表面板（由 my-alpha-engine/full_build/build_labels.py 副产）
# 取代 VWAP_POST_PATH 作为评估的 forward_returns 源——后者依赖米筐 adjust_type="post_volume" 黑盒，
# 这个由本地 raw OHLCV + ex_factor 手动复权派生，全管线可审计 PIT。
VWAP_PANEL_PATH = _DATA_ROOT / "my-alpha-engine/meta-data/vwap_panel.parquet"
# 预算的 forward_return_{N}d.parquet 由 my-alpha-engine/full_build/build_labels.py 产出。
# replication 评估直接读取，与 alpha-engine 共享同一文件 → bit-exact。
# 缺失的 horizon 会 fallback 到 vwap_panel.parquet 现算。
LABELS_DIR = _DATA_ROOT / "my-alpha-engine/labels"

# ==================== 项目内输出目录 ====================
# 报告、图片等评估输出（项目代码相对路径，不受 _DATA_ROOT 影响）
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ==================== 外部因子数据目录 ====================
# 因子 panel 已统一到 my-alpha-engine 的 factor-panel/<producer>/ 命名空间下。
# 本项目（spec engine）产物落在 spec/ 子目录下。
_PANEL_BASE = _DATA_ROOT / "my-alpha-engine"
RAW_FACTOR_DIR = _PANEL_BASE / "factor-panel" / "spec"
CLEANED_FACTOR_DIR = _PANEL_BASE / "cleaned-factor-panel" / "spec"

# 分钟级因子用：原始 per-stock 分钟 parquet 目录（用户日更）+ 中间产物缓存目录
MINUTE_DATA_DIR = _DATA_ROOT / "backtest_engine/cache_dir/stock_data_1m_post"
INTERMEDIATE_CACHE_DIR = _DATA_ROOT / "factor-replication/intermediate_cache"

# ==================== 默认评估区间 ====================
# CLI 不显式指定 --start-date/--end-date 时使用
DEFAULT_START_DATE = "20160101"
DEFAULT_END_DATE = "20251231"
