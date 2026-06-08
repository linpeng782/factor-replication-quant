"""
DazzleReducer —— 方正"适度冒险"因子的 minute→日频 归约（市场微观结构系列①）
============================================================
论文构造（paper.md §3.1-3.3）：剔除开收盘后，逐分钟成交量增量 ΔV[m]=V[m]-V[m-1]；
当日 ΔV 的 mean/std 定阈值，ΔV>mean+surge_k·std 的分钟 = "激增时刻"。两条日频代理：
  dazzle_ret（日耀眼收益率）= 激增时刻【当分钟收益率】的均值；
  dazzle_vol（日耀眼波动率）= 每个激增时刻"耀眼 window 分钟"[m, m+window-1] 内分钟收益率
                              的 std，再对当日所有激增时刻求均值。
"适度"(|日值-截面均值|)、月均/月稳/合成 全部留 L3（spec 图），reducer 只产逐股日频原始值。

warmup=0（纯日内，无跨日依赖）；收益率/ΔV 为日内比值/同日量，读时复权 cf 抵消 → 不依赖
stock-ex-factors（与 tide 同）。位序处理：在剔开收盘后的序列上按【位序】算相邻（不用带午休
gap 的 minute_of_day），ΔV/收益的 [0] 位无前值置 NaN，不参与阈值与激增。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from core.operators.minute_engine import MinuteReducer, register_reducer

_SURGE_K = 1.0    # 激增阈值 = mean + k·std（论文 k=1）
_WINDOW = 5       # "耀眼 5 分钟"：激增时刻及随后 4 分钟
_DROP_OPEN = 4    # 剔除开盘前 4 分钟（与 tide 一致；论文仅言"剔除开盘"未给具体分钟数）
_DROP_CLOSE = 7   # 剔除收盘后 7 分钟（同上）
_MIN_MINUTES = 200  # 当日有效分钟过少（停牌/异常）→ 该日 NaN
_DDOF = 1         # std 自由度（样本 std，与 pandas rolling std 默认一致）


@register_reducer("minute_dazzle")
class DazzleReducer(MinuteReducer):
    """适度冒险归约：单股分钟 → 日频 [dazzle_vol, dazzle_ret]。warmup=0。"""

    version = "v1"
    superset_columns: List[str] = ["dazzle_vol", "dazzle_ret"]
    warmup = 0

    def __init__(self, cache_key: str, surge_k: float = _SURGE_K, window: int = _WINDOW,
                 drop_open: int = _DROP_OPEN, drop_close: int = _DROP_CLOSE):
        self.cache_key = cache_key
        self.surge_k = surge_k
        self.window = window
        self.drop_open = drop_open
        self.drop_close = drop_close
        self.params = {
            "surge_k": surge_k,
            "window": window,
            "drop_open": drop_open,
            "drop_close": drop_close,
        }

    @classmethod
    def from_step(cls, step: Dict) -> "DazzleReducer":
        return cls(
            step["cache_key"],
            float(step.get("surge_k", _SURGE_K)),
            int(step.get("window", _WINDOW)),
            int(step.get("drop_open", _DROP_OPEN)),
            int(step.get("drop_close", _DROP_CLOSE)),
        )

    def reduce(self, ob: str, raw: pd.DataFrame) -> pd.DataFrame:
        if "datetime" not in raw.columns or raw.empty:
            return pd.DataFrame()
        raw = raw.copy()
        dt = pd.to_datetime(raw["datetime"])
        raw["date"] = dt.dt.normalize()
        raw["mod"] = dt.dt.hour * 60 + dt.dt.minute
        raw = raw.drop_duplicates(subset=["date", "mod"], keep="last")

        wide_vol = raw.pivot(index="date", columns="mod", values="volume").sort_index()
        if wide_vol.empty:
            return pd.DataFrame()
        cols = wide_vol.columns  # 升序 mod → 列位序 = 当日第几分钟
        wide_close = raw.pivot(index="date", columns="mod", values="close").reindex(
            index=wide_vol.index, columns=cols
        )

        n_minutes = wide_vol.notna().sum(axis=1).to_numpy()
        ncols = wide_vol.shape[1]
        c_lo, c_hi = self.drop_open, ncols - self.drop_close  # 保留 [c_lo, c_hi)
        V = wide_vol.to_numpy(dtype=float)[:, c_lo:c_hi]      # 不 fillna：缺分钟→ΔV/收益 NaN 不计
        C = wide_close.to_numpy(dtype=float)[:, c_lo:c_hi]
        n_days, L = V.shape

        dazzle_vol = np.full(n_days, np.nan)
        dazzle_ret = np.full(n_days, np.nan)
        w = self.window

        for i in range(n_days):
            if n_minutes[i] < _MIN_MINUTES or L < 3:
                continue
            Vi, Ci = V[i], C[i]
            dv = np.full(L, np.nan)
            dv[1:] = Vi[1:] - Vi[:-1]                 # 成交量增量（位序相邻）
            rr = np.full(L, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                rr[1:] = Ci[1:] / Ci[:-1] - 1.0       # 分钟收益率
            valid_dv = np.isfinite(dv)
            if valid_dv.sum() < 2:
                continue
            d = dv[valid_dv]
            mu, sd = d.mean(), d.std(ddof=_DDOF)
            if not np.isfinite(sd):
                continue
            thr = mu + self.surge_k * sd
            surge_q = np.where(valid_dv & (dv > thr))[0]   # 激增时刻位序
            if surge_q.size == 0:
                continue

            # 日耀眼收益率：激增时刻当分钟收益率均值
            rv = rr[surge_q]
            rv = rv[np.isfinite(rv)]
            if rv.size:
                dazzle_ret[i] = rv.mean()

            # 日耀眼波动率：每激增时刻 [q, q+w-1] 窗口内分钟收益率 std，再求均值
            stds = []
            for q in surge_q:
                win = rr[q : min(q + w, L)]
                win = win[np.isfinite(win)]
                if win.size >= 2:
                    stds.append(win.std(ddof=_DDOF))
            if stds:
                dazzle_vol[i] = float(np.mean(stds))

        out = pd.DataFrame(
            {"dazzle_vol": dazzle_vol, "dazzle_ret": dazzle_ret},
            index=wide_vol.index,
        ).reset_index()
        out["order_book_id"] = ob
        return out.loc[:, ["order_book_id", "date"] + self.superset_columns]
