"""
TideReducer —— 方正"潮汐"因子的 minute→日频 归约（市场微观结构系列②）
============================================================
论文构造（paper.md §3.1-3.3）：当日分钟按位序，邻域成交量 = 前后各4分钟(共9)之和；
  顶峰 t = argmax(邻域量)；涨潮 m = argmin over [5,t-1]；退潮 n = argmin over [t+1,233]。
日频产出 3 个"速率"（带符号），L3 再做 20 日 rolling：
  full_tide_rate     = (Cn-Cm)/Cm/(n-m)                         → 全潮汐(mean20)
  strong/weak 半潮汐 = 以 t 为界拆涨潮(m→t)/退潮(t→n)，按 Vm<Vn 定强弱：
     涨潮速率 = (Ct-Cm)/Cm/(t-m)；退潮速率 = (Cn-Ct)/Ct/(n-t)
     Vm<=Vn → 涨潮为强；否则 退潮为强                          → 强势(mean20)/弱势(mean20|std20)

warmup=0（纯日内，无跨日依赖）；close 用读时复权（日内比值复权抵消，稳健）。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from core.operators.minute_engine import MinuteReducer, register_reducer

_NEIGHBOR = 4  # 邻域 ±4 分钟（共 9）
_DROP_OPEN = 4  # 剔除开盘前 4 分钟（搜索从位序 5 起）
_DROP_CLOSE = 7  # 剔除收盘后 7 分钟（搜索到位序 233 止，N=240）
_MIN_MINUTES = 200  # 当日有效分钟过少（停牌/异常）→ 该日 NaN


def _neighborhood_sum(V: np.ndarray) -> np.ndarray:
    """每列 = 前后各4列之和（共9，边界截断）。V: (n_days, n_cols)。cumsum 矢量化。"""
    n, ncols = V.shape
    cs = np.concatenate(
        [np.zeros((n, 1)), np.cumsum(V, axis=1)], axis=1
    )  # (n, ncols+1)
    j = np.arange(ncols)
    hi = np.minimum(j + _NEIGHBOR + 1, ncols)
    lo = np.maximum(j - _NEIGHBOR, 0)
    return cs[:, hi] - cs[:, lo]


@register_reducer("minute_tide")
class TideReducer(MinuteReducer):
    """潮汐归约：单股分钟 → 日频 [full/strong/weak 潮汐速率]。warmup=0。"""

    version = "v1"
    superset_columns: List[str] = [
        "full_tide_rate",
        "strong_halftide_rate",
        "weak_halftide_rate",
    ]
    warmup = 0

    def __init__(self, cache_key: str):
        self.cache_key = cache_key
        self.params = {
            "neighbor": _NEIGHBOR,
            "drop_open": _DROP_OPEN,
            "drop_close": _DROP_CLOSE,
        }

    @classmethod
    def from_step(cls, step: Dict) -> "TideReducer":
        return cls(step["cache_key"])

    def reduce(self, ob: str, raw: pd.DataFrame) -> pd.DataFrame:
        if "datetime" not in raw.columns or raw.empty:
            return pd.DataFrame()
        raw = raw.copy()
        raw["date"] = pd.to_datetime(raw["datetime"]).dt.normalize()
        raw["mod"] = (
            pd.to_datetime(raw["datetime"]).dt.hour * 60
            + pd.to_datetime(raw["datetime"]).dt.minute
        )
        raw = raw.drop_duplicates(subset=["date", "mod"], keep="last")

        wide_vol = raw.pivot(index="date", columns="mod", values="volume").sort_index()
        if wide_vol.empty:
            return pd.DataFrame()
        cols = wide_vol.columns  # 升序 mod → 列位序 = 当日第几分钟
        wide_close = raw.pivot(index="date", columns="mod", values="close").reindex(
            index=wide_vol.index, columns=cols
        )

        n_minutes = wide_vol.notna().sum(axis=1).to_numpy()  # 当日有效分钟数
        ncols = wide_vol.shape[1]
        # ⚠️ 论文顺序（L116）：先剔除开盘前 _DROP_OPEN / 收盘后 _DROP_CLOSE，再在裁剪序列上算邻域。
        # 否则开盘集合竞价巨量会经 ±4 邻域【泄漏】进首个保留分钟，顶峰恒贴左 → 大量 NaN。
        c_lo, c_hi = _DROP_OPEN, ncols - _DROP_CLOSE  # 保留列 [c_lo, c_hi)（位序 5..233）
        V = wide_vol.fillna(0.0).to_numpy(dtype=float)[:, c_lo:c_hi]
        C = wide_close.to_numpy(dtype=float)[:, c_lo:c_hi]
        nb = _neighborhood_sum(V)                     # 邻域在裁剪后序列上算
        n_days, L = V.shape

        full = np.full(n_days, np.nan)
        strong = np.full(n_days, np.nan)
        weak = np.full(n_days, np.nan)

        for i in range(n_days):
            if n_minutes[i] < _MIN_MINUTES or L < 3:
                continue
            nbi = nb[i]
            t = int(np.argmax(nbi))                   # 顶峰：裁剪序列内全局最大
            if t <= 0 or t + 1 >= L:                  # 顶峰贴边 → 涨潮/退潮其一无效
                continue
            m = int(np.argmin(nbi[:t]))
            n = (t + 1) + int(np.argmin(nbi[t + 1:]))
            Cm, Ct, Cn = C[i, m], C[i, t], C[i, n]
            Vm, Vn = nbi[m], nbi[n]
            if (
                not (np.isfinite(Cm) and np.isfinite(Ct) and np.isfinite(Cn))
                or Cm <= 0
                or Ct <= 0
            ):
                continue
            full[i] = (Cn - Cm) / Cm / (n - m)
            rise = (Ct - Cm) / Cm / (t - m)  # 涨潮速率
            ebb = (Cn - Ct) / Ct / (n - t)  # 退潮速率
            if Vm <= Vn:  # 涨潮起点更低 → 涨潮为强势
                strong[i], weak[i] = rise, ebb
            else:
                strong[i], weak[i] = ebb, rise

        out = pd.DataFrame(
            {
                "full_tide_rate": full,
                "strong_halftide_rate": strong,
                "weak_halftide_rate": weak,
            },
            index=wide_vol.index,
        ).reset_index()
        out["order_book_id"] = ob
        return out.loc[:, ["order_book_id", "date"] + self.superset_columns]
