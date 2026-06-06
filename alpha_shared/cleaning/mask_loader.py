"""
交易状态过滤：从外部 combo_mask + new_stock_mask 加载 pre/post 过滤掩码
============================================================

数据源（由调用方传入路径，本模块不依赖任何项目 config）：
    1. combo_mask_path（典型来源：backtest_engine 项目产出）
       长表，列 [order_book_id, datetime, is_st, is_suspended, is_limit_up]
    2. new_stock_mask_path
       长表，列 [order_book_id, datetime, is_new_stock]

实盘时序逻辑:
    T 日盘后:
      1. 用 pre_mask 过滤 ST / 停牌 / 新股（这三类不参与因子计算，避免污染分布）
      2. 在干净横截面上做 winsorize + zscore
    T+1 日开盘前:
      3. 用 post_mask 过滤涨停（涨停股参与了截面标准化但不下单）

shift(-1) 语义:
    combo_mask 中的 is_xxx / is_new_stock 都是 T 日"当天"状态。
    本模块在 unstack 后 shift(-1)，使 T 日的 mask = T+1 日的状态，
    最终语义：T 日因子信号可在 T+1 日开盘下单。
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger


def _load_long_to_wide(
    parquet_path: Path,
    value_col: str,
) -> pd.DataFrame:
    """
    加载长表 parquet 中的指定布尔列，unstack 成 (T, N) 宽表，并 shift(-1)。

    返回:
        wide: pd.DataFrame, index=datetime, columns=order_book_id, dtype=bool
              T 日的值 = T+1 日的原始状态（最后一行保守置 False）
    """
    long_df = pd.read_parquet(
        parquet_path, columns=["order_book_id", "datetime", value_col]
    )
    long_df["datetime"] = pd.to_datetime(long_df["datetime"])

    wide_raw = (
        long_df.set_index(["datetime", "order_book_id"])[value_col]
        .unstack(level="order_book_id")
        .sort_index()
    )

    # shift(-1)：T 日 mask 反映 T+1 状态（用 numpy 直接位移，避免 dtype 退化为 object）
    arr = wide_raw.to_numpy(dtype=bool)
    shifted = np.empty_like(arr)
    shifted[:-1] = arr[1:]
    shifted[-1] = False  # 最后一行保守置 False
    return pd.DataFrame(shifted, index=wide_raw.index, columns=wide_raw.columns)


def _slice_and_reindex(
    wide: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
    reindex_columns: Optional[pd.Index],
    fill_value: bool,
    name: str,
) -> pd.DataFrame:
    """对宽表做时间区间过滤 + 列对齐。"""
    if start is not None or end is not None:
        wide = wide.loc[start:end]

    if reindex_columns is not None:
        missing = set(reindex_columns) - set(wide.columns)
        if missing:
            logger.warning(
                f"[filters/{name}] {len(missing)} 只股票在 mask 中不存在，"
                f"将填 {fill_value}；样例: {sorted(missing)[:5]}"
            )
        wide = wide.reindex(columns=reindex_columns, fill_value=fill_value)
    return wide


def load_filter_masks(
    combo_mask_path: Union[str, Path],
    new_stock_mask_path: Union[str, Path],
    start: Optional[str] = None,
    end: Optional[str] = None,
    reindex_columns: Optional[pd.Index] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载 pre/post 过滤掩码（T 日信号 → T+1 日交易）。

    pre_mask  (T, N) bool: True = 参与因子分布（标准化）
        = NOT is_st AND NOT is_suspended AND NOT is_new_stock
        ST / 停牌 / 新股 这三类股票 T 日就已知不可交易，提前过滤避免污染分布。

    post_mask (T, N) bool: True = 可执行下单
        = NOT is_limit_up
        涨停股是正常可观测股票，应参与截面标准化；但 T+1 日不能买入，故最终过滤。

    最终可交易 = pre_mask & post_mask。

    所有 mask 都经过 shift(-1)：T 日的 mask = T+1 日的原始状态。
    （reindex_columns 缺失股票 pre_mask 填 False，post_mask 填 True，
      最终组合为 False，即缺失股票不可交易）

    参数:
        combo_mask_path      : combo_mask parquet 路径（**必填**，由 caller 注入）
        new_stock_mask_path  : new_stock_mask parquet 路径（**必填**，由 caller 注入）
        start, end           : 时间区间过滤（YYYY-MM-DD），None 表示不过滤
        reindex_columns      : 目标股票池；传入后对齐列

    返回:
        (pre_mask, post_mask): tuple[pd.DataFrame, pd.DataFrame]
    """
    combo_path = Path(combo_mask_path)
    new_stock_path = Path(new_stock_mask_path)
    if not combo_path.exists():
        raise FileNotFoundError(f"combo_mask 文件不存在: {combo_path}")
    if not new_stock_path.exists():
        raise FileNotFoundError(f"new_stock_mask 文件不存在: {new_stock_path}")

    # 1. 加载三类 combo_mask 列 + new_stock_mask（unstack + shift(-1)）
    is_st = _load_long_to_wide(combo_path, "is_st")
    is_suspended = _load_long_to_wide(combo_path, "is_suspended")
    is_limit_up = _load_long_to_wide(combo_path, "is_limit_up")
    is_new_stock = _load_long_to_wide(new_stock_path, "is_new_stock")

    # 2. 把 new_stock 的索引对齐到 combo 的索引（理论上完全对齐，做一次保险）
    is_new_stock = is_new_stock.reindex(
        index=is_st.index, columns=is_st.columns, fill_value=False
    )

    # 3. 组合 pre_mask 和 post_mask
    pre_mask_full = ~(is_st | is_suspended | is_new_stock)
    post_mask_full = ~is_limit_up

    # 4. 时间区间 + 列对齐
    #    pre_mask 缺失股票 fill_value=False（缺失视为不可交易）
    #    post_mask 缺失股票 fill_value=True（仅控制涨停过滤；最终乘上 pre_mask 后仍为 False）
    pre_mask = _slice_and_reindex(
        pre_mask_full, start, end, reindex_columns, fill_value=False, name="pre"
    )
    post_mask = _slice_and_reindex(
        post_mask_full, start, end, reindex_columns, fill_value=True, name="post"
    )

    # 5. 日志统计
    final = pre_mask & post_mask
    logger.info(
        f"[filters] 加载完成: shape={pre_mask.shape}, "
        f"区间={pre_mask.index.min().date()} ~ {pre_mask.index.max().date()}"
    )
    logger.info(
        f"[filters] pre_mask 通过率={pre_mask.values.mean():.2%} "
        f"(过滤 ST/停牌/新股)"
    )
    logger.info(
        f"[filters] post_mask 通过率={post_mask.values.mean():.2%} (过滤涨停)"
    )
    logger.info(
        f"[filters] 最终可交易比例={final.values.mean():.2%} (pre_mask & post_mask)"
    )

    return pre_mask, post_mask
