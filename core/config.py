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
# 所有原始因子统一放在 raw_factor/，清洗后统一放在 cleaned_factor/
_BASE_FACTOR_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/factor-replication")
RAW_FACTOR_DIR = _BASE_FACTOR_DIR / "raw_factor"
CLEANED_FACTOR_DIR = _BASE_FACTOR_DIR / "cleaned_factor"

# ==================== 默认评估区间 ====================
# CLI 不显式指定 --start-date/--end-date 时使用
DEFAULT_START_DATE = "20160101"
DEFAULT_END_DATE = "20251231"
