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

# ==================== 输出目录 ====================
# 报告、图片等评估输出
OUTPUT_DIR = Path(__file__).parent.parent / "output"

# 因子 parquet 输出（外部数据目录）
FACTOR_OUTPUT_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/factor-replication")
