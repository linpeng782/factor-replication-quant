"""
ml_core.splits —— 时间切分 + embargo（统一代码，数值各入口保留）
============================================================
train / valid / test 三段，段间留 embargo 月（> horizon 天）防标签泄露。
两条线数值本就一致（train≤2017-11-30 / valid 2018-01-01~2019-11-30 / test≥2020-01-01），
差异仅 test 末日：ml 卡 2026-03-31（label 可兑现），ml_ht 开口到末日。
统一切分逻辑，数值用 SplitConfig 传入（重构期各入口保留各自数值，保证零漂移）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# embargo：2017-12 与 2019-12 整月落空（不进任何段），间隔 > 20 日 horizon
DEFAULT_SPLIT = {
    "train": (None, "2017-11-30"),
    "valid": ("2018-01-01", "2019-11-30"),
    "test": ("2020-01-01", None),
}


@dataclass
class SplitConfig:
    segments: dict[str, tuple[str | None, str | None]]

    @classmethod
    def default(cls) -> "SplitConfig":
        return cls(segments=dict(DEFAULT_SPLIT))


def assign_segments(dates: pd.DatetimeIndex, cfg: SplitConfig | None = None) -> dict[str, np.ndarray]:
    """对每个交易日打 train/valid/test 段标签 → {seg: (T,) bool over dates}。

    embargo 月不落入任何段（即不被任一 (lo,hi) 覆盖的日期，三段都 False）。
    """
    cfg = cfg or SplitConfig.default()
    out: dict[str, np.ndarray] = {}
    for seg, (lo, hi) in cfg.segments.items():
        m = np.ones(len(dates), dtype=bool)
        if lo is not None:
            m &= np.asarray(dates >= pd.Timestamp(lo))
        if hi is not None:
            m &= np.asarray(dates <= pd.Timestamp(hi))
        out[seg] = m
    return out
