"""mask 加载（thin wrapper → alpha_shared.cleaning.mask_loader）。

通过本 wrapper 注入项目 config 路径；调用方接口与抽取前完全一致。
"""

from typing import Optional, Union
from pathlib import Path

import pandas as pd

from alpha_shared.cleaning.mask_loader import load_filter_masks as _impl
from core import config


def load_filter_masks(
    start: Optional[str] = None,
    end: Optional[str] = None,
    reindex_columns: Optional[pd.Index] = None,
    combo_mask_path: Optional[Union[str, Path]] = None,
    new_stock_mask_path: Optional[Union[str, Path]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """与抽取前签名一致：path 缺省时回退到 config.COMBO_MASK_PATH / NEW_STOCK_MASK_PATH。"""
    return _impl(
        combo_mask_path=combo_mask_path or config.COMBO_MASK_PATH,
        new_stock_mask_path=new_stock_mask_path or config.NEW_STOCK_MASK_PATH,
        start=start,
        end=end,
        reindex_columns=reindex_columns,
    )
