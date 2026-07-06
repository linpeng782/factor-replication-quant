"""
[第 1 层｜评估输入] market-data/ 下的评估输入数据。

由 data_fetching/ 产出/日更；因子生产线与 ML 训练线都从这里读输入
（mask / vwap / labels / 行业 / 市值 / 指数）。
"""

from .base import _DATA_ROOT, _MKT, MASK_BACKEND

__all__ = [
    "COMBO_MASK_PATH",
    "NEW_STOCK_MASK_PATH",
    "VWAP_PANEL_PATH",
    "LABELS_DIR",
    "INDUSTRY_PANEL_ZX_PATH",
    "MARKET_CAP_PANEL_PATH",
    "INDUSTRY_INDEX_RETURN_PATH",
    "INDEX_DIR",
    "INDEX_SEGMENTS_PATH",
]

# mask（消费轴，随 MASK_BACKEND）：两份 schema 一致，仅指向不同文件
_MASK_DIR = (
    _DATA_ROOT / "backtest_engine" / "cache_dir_dquant" if MASK_BACKEND == "dquant"
    else _MKT / "masks"
)
COMBO_MASK_PATH = _MASK_DIR / "combo_mask_long.parquet"
NEW_STOCK_MASK_PATH = _MASK_DIR / "new_stock_mask_long.parquet"

# PIT canonical vwap 宽表面板（build_labels.py 副产）；forward_returns 的价格源
# 评估 fallback：缺失 horizon 时从 vwap_panel 现算 forward_return
VWAP_PANEL_PATH = _MKT / "labels/vwap_panel.parquet"
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
