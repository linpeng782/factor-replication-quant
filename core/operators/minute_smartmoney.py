"""
SmartMoneyReducer —— 开源"聪明钱因子 2.0"的 minute→日频 归约（市场微观结构系列③）
============================================================
论文构造（paper.md §2 表1 + §3）：对每股、每日 t：
  1) 回溯过去 10 个交易日的分钟；
  2) 聪明度 S = |R| / V^β（R=分钟收益, V=分钟成交量；β 默认 0.1，原始版 0.5）；
  3) 按 S 降序排，取【成交量累计占比前 cutoff(20%)】的分钟 = 聪明钱交易；
  4) VWAP_smart = 聪明钱分钟的量加权均价；VWAP_all = 全部分钟量加权均价；
  6) 聪明钱因子 Q = VWAP_smart / VWAP_all（L3 做比值）。

★ 第三种 reducer 形态：**跨日逻辑在 reducer 内**（10日池化分位选择，无法拆成"日内归约+L3滚动"）
  → warmup=10（reducer 自报其跨日窗口；引擎据此喂 overlap）。close 用读时复权。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from core.operators.minute_engine import MinuteReducer, register_reducer

_LOOKBACK = 10      # 回溯交易日数（含当日）
_BETA = 0.1         # S = |R| / V^β（论文最优 0.1；原始版 0.5）
_CUTOFF = 0.20      # 成交量累计占比阈值（聪明钱前 20%）


@register_reducer("minute_smartmoney")
class SmartMoneyReducer(MinuteReducer):
    """聪明钱归约：单股分钟 → 日频 [smart_vwap, all_vwap]（10日池化）。warmup=10。"""

    version = "v1"
    superset_columns: List[str] = ["smart_vwap", "all_vwap"]

    def __init__(self, cache_key: str, beta: float = _BETA, cutoff: float = _CUTOFF,
                 lookback: int = _LOOKBACK):
        self.cache_key = cache_key
        self.beta = beta
        self.cutoff = cutoff
        self.lookback = lookback
        self.params = {"beta": beta, "cutoff": cutoff, "lookback": lookback}
        self.warmup = lookback   # 跨日窗口在 reducer 内 → 回看 lookback 日

    @classmethod
    def from_step(cls, step: Dict) -> "SmartMoneyReducer":
        return cls(
            step["cache_key"],
            float(step.get("beta", _BETA)),
            float(step.get("cutoff", _CUTOFF)),
            int(step.get("lookback", _LOOKBACK)),
        )

    def reduce(self, ob: str, raw: pd.DataFrame) -> pd.DataFrame:
        if "datetime" not in raw.columns or raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        dt = pd.to_datetime(df["datetime"])
        df["date"] = dt.dt.normalize()
        df = df.sort_values("datetime").reset_index(drop=True)

        close = df["close"].to_numpy(dtype=float)
        vol = df["volume"].to_numpy(dtype=float)
        # 分钟收益（日内，每日首分钟无前值 → NaN，不参与 S 排序但计入 vwap_all）
        R = df.groupby("date")["close"].pct_change().to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            S = np.abs(R) / np.power(np.where(vol > 0, vol, np.nan), self.beta)
        pv = close * vol                               # 量加权分子（close 后复权）

        # 每个交易日在排序数组中的边界（用于按 [t-9, t] 连续切片）
        dates = df["date"].to_numpy()
        uniq, first_idx = np.unique(dates, return_index=True)   # uniq 升序, first_idx 每日起点
        order = np.argsort(first_idx)
        uniq = uniq[order]; first_idx = first_idx[order]
        end_idx = np.append(first_idx[1:], len(df))             # 每日终点(exclusive)
        n_days = len(uniq)
        L = self.lookback

        smart_vwap = np.full(n_days, np.nan)
        all_vwap = np.full(n_days, np.nan)
        for i in range(L - 1, n_days):                          # 需满 L 日
            lo = first_idx[i - L + 1]
            hi = end_idx[i]
            v = vol[lo:hi]; p = pv[lo:hi]; s = S[lo:hi]
            tot_v = v.sum()
            if tot_v <= 0:
                continue
            all_vwap[i] = p.sum() / tot_v
            # 按 S 降序（NaN 排最后），取成交量累计占比 ≤ cutoff 的分钟为聪明钱
            sk = np.where(np.isfinite(s), s, -np.inf)
            o = np.argsort(sk)[::-1]
            cumfrac = np.cumsum(v[o]) / tot_v
            k = int(np.searchsorted(cumfrac, self.cutoff, side="left")) + 1  # 含跨过阈值那一分钟
            k = min(k, len(o))
            sel = o[:k]
            sv = v[sel].sum()
            if sv > 0:
                smart_vwap[i] = p[sel].sum() / sv

        out = pd.DataFrame(
            {"smart_vwap": smart_vwap, "all_vwap": all_vwap},
            index=pd.DatetimeIndex(uniq, name="date"),
        ).reset_index()
        out["order_book_id"] = ob
        return out.loc[:, ["order_book_id", "date"] + self.superset_columns]
