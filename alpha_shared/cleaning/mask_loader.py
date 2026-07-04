"""
交易状态过滤：从外部 combo_mask + new_stock_mask 加载 can_buy / not_limit_up 掩码
============================================================

数据源（由调用方传入路径，本模块不依赖任何项目 config）：
    1. combo_mask_path（典型来源：backtest_engine 项目产出）
       长表，列 [order_book_id, datetime, is_st, is_suspended, is_limit_up]
    2. new_stock_mask_path
       长表，列 [order_book_id, datetime, is_new_stock]

实盘时序逻辑:
    T 日盘后:
      1. 用 can_buy_mask 过滤 ST / 停牌 / 新股（这三类不参与因子计算，避免污染分布）
      2. 在干净横截面上做 winsorize + zscore
    T+1 日开盘前:
      3. 用 not_limit_up_mask 过滤涨停（涨停股参与了截面标准化但不下单）

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


def _long_to_wide(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    长表某状态列 → (T, N) 宽表。纯格式转换，不做任何时序位移。

    未上市 / 退市 / 数据缺失的格子 unstack 后是 NaN → 显式按「状态未知=阻断」
    处理：fillna(True) 使该格子被 ~status 排除出可买池（宁可错杀不可漏放）。

    参数:
        long_df  : 已 set_index([datetime, order_book_id]) 的长表
        value_col: 状态列名（is_st / is_suspended / is_limit_up / is_new_stock）
    """
    wide = long_df[value_col].unstack(level="order_book_id").sort_index()
    return wide.fillna(True).astype(bool)  # 未知状态 = 阻断（显式决策，勿改成 False）


def _use_next_day_status(wide: pd.DataFrame) -> pd.DataFrame:
    """
    T 日行使用 T+1 日的状态（实现为 shift(-1)：把明天的状态拉回今天的行）。

    业务背景：T 日盘后算因子 → T+1 日开盘下单。周二盘后需要知道的是
    「周三停不停牌 / 涨不涨停」，所以把周三的状态搬到周二行。
    末行没有明日数据 → 保守置 False（不可交易）。

    前提：行索引必须升序且无重复（按行位置位移，日期缺失/乱序会静默错位一天，
    时序错一天整个回测就全错）——进函数先校验，坏数据当场报错。
    """
    idx = wide.index
    if not (idx.is_monotonic_increasing and idx.is_unique):
        raise ValueError("mask 日期索引必须升序且无重复，否则按行位移会静默错位")
    arr = wide.to_numpy(dtype=bool)
    shifted = np.empty_like(arr)
    shifted[:-1] = arr[1:]   # T 行 ← T+1 行
    shifted[-1] = False      # 末日无明日数据，保守不可交易
    return pd.DataFrame(shifted, index=idx, columns=wide.columns)


def load_filter_masks(
    combo_mask_path: Union[str, Path],
    new_stock_mask_path: Union[str, Path],
    start: Optional[str] = None,
    end: Optional[str] = None,
    reindex_columns: Optional[pd.Index] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载交易状态过滤掩码（T 日信号 → T+1 日交易）。

    can_buy_mask  (T, N) bool: True = 参与因子分布（标准化）
        = NOT is_st AND NOT is_suspended AND NOT is_new_stock
        ST / 停牌 / 新股 这三类股票 T 日就已知不可交易，提前过滤避免污染分布。

    not_limit_up_mask (T, N) bool: True = 可执行下单
        = NOT is_limit_up
        涨停股是正常可观测股票，应参与截面标准化；但 T+1 日不能买入，故最终过滤。

    最终可交易 = can_buy_mask & not_limit_up_mask。

    所有 mask 都经过 shift(-1)：T 日的 mask = T+1 日的原始状态。
    （reindex_columns 缺失股票 can_buy_mask 填 False，not_limit_up_mask 填 True，
      最终组合为 False，即缺失股票不可交易）

    参数:
        combo_mask_path      : combo_mask parquet 路径（**必填**，由 caller 注入）
        new_stock_mask_path  : new_stock_mask parquet 路径（**必填**，由 caller 注入）
        start, end           : 时间区间过滤（YYYY-MM-DD），None 表示不过滤
        reindex_columns      : 目标股票池；传入后对齐列

    返回:
        (can_buy_mask, not_limit_up_mask): tuple[pd.DataFrame, pd.DataFrame]
    """
    combo_path = Path(combo_mask_path)
    new_stock_path = Path(new_stock_mask_path)
    if not combo_path.exists():
        raise FileNotFoundError(f"combo_mask 文件不存在: {combo_path}")
    if not new_stock_path.exists():
        raise FileNotFoundError(f"new_stock_mask 文件不存在: {new_stock_path}")

    # ── 1. 原始状态（T 日当天）：combo 一次 I/O 读 3 列，new_stock 单独文件 ──
    combo = pd.read_parquet(
        combo_path,
        columns=["order_book_id", "datetime", "is_st", "is_suspended", "is_limit_up"],
    )
    combo["datetime"] = pd.to_datetime(combo["datetime"])
    combo = combo.set_index(["datetime", "order_book_id"])

    new_stock = pd.read_parquet(
        new_stock_path, columns=["order_book_id", "datetime", "is_new_stock"]
    )
    new_stock["datetime"] = pd.to_datetime(new_stock["datetime"])
    new_stock = new_stock.set_index(["datetime", "order_book_id"])

    # ── 2. 换到 T+1 视角（T 日盘后决策，看的是明天的状态）──
    is_st = _use_next_day_status(_long_to_wide(combo, "is_st"))
    is_suspended = _use_next_day_status(_long_to_wide(combo, "is_suspended"))
    is_limit_up = _use_next_day_status(_long_to_wide(combo, "is_limit_up"))
    is_new_stock = _use_next_day_status(_long_to_wide(new_stock, "is_new_stock"))

    # new_stock 网格对齐到 combo 网格（理论上完全一致，做一次保险；
    # combo 里有而 new_stock 里没有的格子补 False = 非新股，不因缺数据误杀）
    is_new_stock = is_new_stock.reindex(
        index=is_st.index, columns=is_st.columns, fill_value=False
    )

    # ── 3. 组合成两个决策掩码 ──
    #    can_buy      资格过滤：ST/停牌/新股 → 连因子池都不进（污染截面分布）
    #    not_limit_up 价格过滤：涨停股进池参与标准化，但 T+1 下不了单
    can_buy_mask = ~(is_st | is_suspended | is_new_stock)
    not_limit_up_mask = ~is_limit_up

    # ── 4. 裁剪区间 + 对齐目标股票池 ──
    if start is not None or end is not None:
        can_buy_mask = can_buy_mask.loc[start:end]
        not_limit_up_mask = not_limit_up_mask.loc[start:end]

    if reindex_columns is not None:
        missing = set(reindex_columns) - set(can_buy_mask.columns)
        if missing:
            logger.warning(
                f"[filters] {len(missing)} 只股票不在 mask 中"
                f"（can_buy 补 False / not_limit_up 补 True）；样例: {sorted(missing)[:5]}"
            )
        # 不对称补值：can_buy 是基础过滤，缺数据 = 不可买；
        # not_limit_up 是附加过滤，不因缺数据误杀（最终 can_buy & not_limit_up 仍为 False，由 can_buy 兜底）
        can_buy_mask = can_buy_mask.reindex(columns=reindex_columns, fill_value=False)
        not_limit_up_mask = not_limit_up_mask.reindex(columns=reindex_columns, fill_value=True)

    # 5. 日志统计
    final = can_buy_mask & not_limit_up_mask
    logger.info(
        f"[filters] 加载完成: shape={can_buy_mask.shape}, "
        f"区间={can_buy_mask.index.min().date()} ~ {can_buy_mask.index.max().date()}"
    )
    logger.info(
        f"[filters] can_buy_mask 通过率={can_buy_mask.values.mean():.2%} "
        f"(过滤 ST/停牌/新股)"
    )
    logger.info(
        f"[filters] not_limit_up_mask 通过率={not_limit_up_mask.values.mean():.2%} (过滤涨停)"
    )
    logger.info(
        f"[filters] 最终可交易比例={final.values.mean():.2%} (can_buy_mask & not_limit_up_mask)"
    )

    return can_buy_mask, not_limit_up_mask
