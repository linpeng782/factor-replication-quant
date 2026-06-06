"""
未来收益率面板构造
------------------------------------------------------------
T 日信号 → T+1 价格买 → T+1+N 价格卖

    return_N[T] = (price[T+1+N] - price[T+1]) / price[T+1]
                = price.pct_change(N).shift(-N-1)[T]

价格列使用 vwap（后复权），更贴近实盘可执行价。

这样避免了未来函数：T 日用到的 price[T+1] 与 price[T+1+N] 均为
T+1 日及之后才能观测到的数据。
"""

from typing import Iterable

import pandas as pd
from loguru import logger


def build_forward_returns(
    price_panel: pd.DataFrame,
    horizons: Iterable[int],
) -> dict:
    """
    批量构造多持有期未来收益率面板。

    参数:
        price_panel: (T, N) 后复权 vwap 面板
        horizons   : 持有期列表（天）。**无默认值**——必须由 caller 显式传入，
                     避免两个下游项目对默认 horizon 假设不一致。

    返回:
        dict[int -> DataFrame (T, N)]，键为 N，值为对应收益率面板。
    """
    returns = {}
    for n in horizons:
        # fill_method=None：停牌日的 NaN 不向前填充，避免复牌当天的收益被错误拉长
        r = price_panel.pct_change(n, fill_method=None).shift(-n - 1)
        returns[n] = r
        logger.info(
            f"[returns] forward_return_{n}d shape={r.shape}，"
            f"非空比例={r.notna().values.mean():.2%}"
        )
    return returns
