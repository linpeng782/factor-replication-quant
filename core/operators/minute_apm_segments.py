"""
APMSegmentsReducer —— APM 因子族的 minute→日频 段收益归约
============================================================
产出 4 个日频段收益（OVP/AVP/APM 全族共用 superset）：

  ret_overnight  前日 PM 收盘 → 当日 AM 开盘（跨日，后复权抵消）
  ret_am         当日 AM 开盘 → 11:30 收盘
  ret_pm         13:00 后首 bar 开盘 → 15:00 收盘
  ret_pm_late    14:00 后首 bar 开盘 → 15:00 收盘   ← APM_1 用

warmup=1（取前日 PM close 算 overnight；引擎喂 1 日 overlap）。
复权稳健：overnight 跨日价均已由 load_adjusted_minute_window 做后复权，
  ex_cum_factor 累积口径，今日/昨日价格可直接相除。AM/PM 为日内比值，复权抵消。

段边界（mod = hour*60+minute，A 股 1m bar 尾时刻）：
  AM  : mod ∈ [570, 690]  (09:30~11:30)
  PM  : mod ∈ [780, 900]  (13:00~15:00)
  PM↑ : mod ∈ [840, 900]  (14:00~15:00)
  收盘 bar = 全日最后 bar；开盘 bar = 各段最早 bar。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from core.operators.minute_engine import MinuteReducer, register_reducer

_AM_LO,  _AM_HI  = 570, 690   # 09:30 ~ 11:30
_PM_LO,  _PM_HI  = 780, 900   # 13:00 ~ 15:00
_PML_LO          = 840        # 14:00+
_MIN_MINUTES = 100            # 有效 bar 数过少（停牌/异常）→ 该日全 NaN


@register_reducer("minute_apm_segments")
class APMSegmentsReducer(MinuteReducer):
    """APM 段收益归约：单股分钟 → 日频 [ret_overnight, ret_am, ret_pm, ret_pm_late]。
    warmup=1（需前日收盘价算隔夜）。"""

    version = "v1"
    superset_columns: List[str] = ["ret_overnight", "ret_am", "ret_pm", "ret_pm_late"]
    warmup = 1

    def __init__(self, cache_key: str):
        self.cache_key = cache_key
        self.params: Dict = {}   # 无自由参数；段边界固定 A 股交易时段

    @classmethod
    def from_step(cls, step: Dict) -> "APMSegmentsReducer":
        return cls(step["cache_key"])

    def reduce(self, ob: str, raw: pd.DataFrame) -> pd.DataFrame:
        if "datetime" not in raw.columns or raw.empty:
            return pd.DataFrame()
        raw = raw.copy()
        dt = pd.to_datetime(raw["datetime"])
        raw["date"] = dt.dt.normalize()
        raw["mod"]  = dt.dt.hour * 60 + dt.dt.minute
        raw = raw.drop_duplicates(subset=["date", "mod"], keep="last")

        # ── 段边界 bar 宽表 ──────────────────────────────────────────
        # pivot(date × mod)，用 open/close 两张宽表
        wide_o = raw.pivot(index="date", columns="mod", values="open").sort_index()
        wide_c = raw.pivot(index="date", columns="mod", values="close").sort_index()
        dates  = wide_o.index
        n_days = len(dates)

        if n_days < 1:
            return pd.DataFrame()

        am_cols  = [c for c in wide_o.columns if _AM_LO  <= c <= _AM_HI]
        pm_cols  = [c for c in wide_o.columns if _PM_LO  <= c <= _PM_HI]
        pml_cols = [c for c in wide_o.columns if _PML_LO <= c <= _PM_HI]

        def _first_open(cols):
            if not cols:
                return pd.Series(np.nan, index=dates)
            sub = wide_o[sorted(cols)]
            return sub.apply(lambda r: r.dropna().iloc[0] if r.notna().any() else np.nan, axis=1)

        def _last_close(cols):
            if not cols:
                return pd.Series(np.nan, index=dates)
            sub = wide_c[sorted(cols)]
            return sub.apply(lambda r: r.dropna().iloc[-1] if r.notna().any() else np.nan, axis=1)

        am_open  = _first_open(am_cols)   # 首 AM bar open（今日开盘）
        am_close = _last_close(am_cols)   # 末 AM bar close（11:30）
        pm_open  = _first_open(pm_cols)   # 首 PM bar open（~13:01）
        pm_close = _last_close(pm_cols)   # 末 PM bar close（15:00 = 全日收）
        pml_open = _first_open(pml_cols)  # 首 14:00 bar open

        # 有效 bar 数守门（停牌/异常日 → 全 NaN）
        n_bars = raw.groupby("date")["mod"].count()
        invalid = n_bars < _MIN_MINUTES

        with np.errstate(divide="ignore", invalid="ignore"):
            # 隔夜：今开 / 前日全收 - 1（shift(1) 对齐前日）
            prev_pm_close = pm_close.shift(1)
            ret_overnight = am_open / prev_pm_close - 1.0

            ret_am       = am_close  / am_open  - 1.0
            ret_pm       = pm_close  / pm_open  - 1.0
            ret_pm_late  = pm_close  / pml_open - 1.0

        # 应用守门（停牌日整日 NaN）
        for s in (ret_overnight, ret_am, ret_pm, ret_pm_late):
            s[invalid] = np.nan

        out = pd.DataFrame({
            "order_book_id": ob,
            "date":          dates,
            "ret_overnight": ret_overnight.values,
            "ret_am":        ret_am.values,
            "ret_pm":        ret_pm.values,
            "ret_pm_late":   ret_pm_late.values,
        })
        return out.loc[:, ["order_book_id", "date"] + self.superset_columns]
