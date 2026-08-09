"""
HLVolReducer —— 国盛"高/低位放量"因子的 minute→日频 归约（量价淘金系列④）
============================================================
论文构造（paper.md §二/§3.1）：股票每日的波动率用当日分钟数据计算，
即【1 分钟收益率序列的标准差】；"高/低位"用日收盘价在过去 20 日中的相对位置刻画。

reducer 只产两列逐股日频原始值：
  intraday_vol : 当日 1 分钟收益率序列 std（ddof=1，日内相邻分钟收盘价比值）
  close_d      : 当日最后一个有效分钟的收盘价（后复权，供 20 日窗口内排序/求均值）

20 日窗口内"按价格分组看波动 / 按波动分组看价格"的占比全部留 L3
（spec 图的 rolling_group_ratio 算子），reducer 不做跨日运算 → warmup=0。

口径说明：
  - 收益率按【位序相邻】计算（与 dazzle 一致），缺分钟处 NaN 不参与 std；
  - 日内比值读时复权 cf 抵消 → intraday_vol 不依赖 ex-factors；close_d 为后复权价，
    跨日可比（20 日排序仅要求同股窗口内单调一致，后复权满足）；
  - 当日有效分钟收益 < _MIN_RETURNS（长时间盘中停牌/异常）→ 该日 intraday_vol NaN。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from core.operators.minute_engine import MinuteReducer, register_reducer

_MIN_RETURNS = 30   # 当日有效分钟收益数下限，低于则 intraday_vol=NaN（防停牌日噪声）
_DDOF = 1           # std 自由度（样本 std，与 pandas 默认一致）


@register_reducer("minute_hlvol")
class HLVolReducer(MinuteReducer):
    """高/低位放量归约：单股分钟 → 日频 [intraday_vol, close_d]。warmup=0。"""

    version = "v1"
    superset_columns: List[str] = ["intraday_vol", "close_d"]
    warmup = 0

    def __init__(self, cache_key: str, min_returns: int = _MIN_RETURNS):
        self.cache_key = cache_key
        self.min_returns = min_returns
        self.params = {"min_returns": min_returns}

    @classmethod
    def from_step(cls, step: Dict) -> "HLVolReducer":
        return cls(step["cache_key"], int(step.get("min_returns", _MIN_RETURNS)))

    def reduce(self, ob: str, raw: pd.DataFrame) -> pd.DataFrame:
        if "datetime" not in raw.columns or raw.empty:
            return pd.DataFrame()
        raw = raw.copy()
        dt = pd.to_datetime(raw["datetime"])
        raw["date"] = dt.dt.normalize()
        raw["mod"] = dt.dt.hour * 60 + dt.dt.minute
        raw = raw.drop_duplicates(subset=["date", "mod"], keep="last")

        wide_close = raw.pivot(index="date", columns="mod", values="close").sort_index()
        if wide_close.empty:
            return pd.DataFrame()

        C = wide_close.to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = C[:, 1:] / C[:, :-1] - 1.0          # 位序相邻分钟收益，缺分钟→NaN
        n_valid = np.isfinite(rr).sum(axis=1)
        vol = np.full(C.shape[0], np.nan)
        rows_ok = n_valid >= max(self.min_returns, 2)
        if rows_ok.any():
            vol[rows_ok] = np.nanstd(rr[rows_ok], axis=1, ddof=_DDOF)

        # 当日最后一个有效分钟收盘价
        idx_last = wide_close.shape[1] - 1 - np.argmax(np.isfinite(C[:, ::-1]), axis=1)
        close_d = C[np.arange(C.shape[0]), idx_last]
        close_d[~np.isfinite(C).any(axis=1)] = np.nan

        out = pd.DataFrame(
            {"intraday_vol": vol, "close_d": close_d}, index=wide_close.index
        ).reset_index()
        out["order_book_id"] = ob
        return out.loc[:, ["order_book_id", "date"] + self.superset_columns]
