"""
因子复现项目配置

包含 mask / vwap 等预计算数据的路径配置
"""

from pathlib import Path

# ==================== 外部预计算数据路径 ====================
# 这些路径指向 my-alpha-engine / backtest_engine 已生产的数据

COMBO_MASK_PATH = Path(
    "/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir/combo_mask_long.parquet"
)

NEW_STOCK_MASK_PATH = Path(
    "/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir/new_stock_mask_long.parquet"
)

VWAP_POST_PATH = Path(
    "/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir/vwap_post.parquet"
)

# ==================== 项目内输出目录 ====================
# 报告、图片等评估输出
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# ==================== 外部因子数据目录 ====================
# 因子 panel 已统一到 my-alpha-engine 的 factor-panel/<producer>/ 命名空间下。
# 本项目（spec engine）产物落在 spec/ 子目录下。
_PANEL_BASE = Path("/nfs/ofs-prediction/peterzhenglinpeng/my-alpha-engine")
RAW_FACTOR_DIR = _PANEL_BASE / "factor-panel" / "spec"
CLEANED_FACTOR_DIR = _PANEL_BASE / "cleaned-factor-panel" / "spec"

# 分钟级因子用：原始 per-stock 分钟 parquet 目录（用户日更）+ 中间产物缓存目录
MINUTE_DATA_DIR = Path(
    "/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir/stock_data_1m_post"
)
INTERMEDIATE_CACHE_DIR = Path(
    "/nfs/ofs-prediction/peterzhenglinpeng/factor-replication/intermediate_cache"
)

# ==================== 默认评估区间 ====================
# CLI 不显式指定 --start-date/--end-date 时使用
DEFAULT_START_DATE = "20160101"
DEFAULT_END_DATE = "20251231"
