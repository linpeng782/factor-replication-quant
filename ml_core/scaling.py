"""
ml_core.scaling —— 标准化策略（模型驱动，可插拔）
============================================================
两套刻意不同的口径，做成策略对象（绝不强行统一）：

  WholeSetRobustZ      （LGBM/ml）：全集 per-feature 稳健标准化，train 段 fit，**有状态、持久化**
      z = (x - median) / (1.4826·MAD)，median/MAD 仅由 train 段全部 (日×股) 单元算
      保留跨日水位 + 抗极值；valid/test 仅 transform；实盘复用同一把尺（防 train/serve skew）
      对齐 ml.preprocess.RobustZScoreScaler。

  DailyCrossSectionMAD （MLP/ml_ht）：逐日截面 MAD去极值(±5) + zscore，**无状态**（每天用当天截面）
      逐日：±5·1.4826·MAD 截断 → (x-mean)/std；统计量来自当日截面，不跨日、不持久化
      对齐 ml_ht.dataset.preprocess / build_predict_set（仅用于 has_factor=ALL 的稠密输入）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_MAD = 1.4826


class Standardizer(ABC):
    name: str
    stateful: bool

    @abstractmethod
    def fit(self, X: np.ndarray, dates: np.ndarray | None = None) -> "Standardizer": ...

    @abstractmethod
    def transform(self, X: np.ndarray, dates: np.ndarray | None = None) -> np.ndarray: ...

    def fit_transform(self, X, dates=None):
        return self.fit(X, dates).transform(X, dates)

    def save(self, path: Path) -> None:        # 无状态策略默认空实现
        pass

    @classmethod
    def load(cls, path: Path, feature_names: list[str]) -> "Standardizer":
        return cls()


class WholeSetRobustZ(Standardizer):
    """全集 per-feature RobustZScore（对齐 ml.preprocess.RobustZScoreScaler）。"""

    name = "whole_set_robust_z"
    stateful = True

    def __init__(self) -> None:
        self.median_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: np.ndarray, dates: np.ndarray | None = None) -> "WholeSetRobustZ":
        med = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - med), axis=0) * _MAD
        self.median_ = med
        self.scale_ = np.where(mad > 0, mad, np.nan)   # 常数列 scale=0 → NaN（该列 transform 后全 NaN）
        return self

    def transform(self, X: np.ndarray, dates: np.ndarray | None = None) -> np.ndarray:
        if self.median_ is None:
            raise RuntimeError("WholeSetRobustZ 尚未 fit")
        return ((X - self.median_) / self.scale_).astype(np.float32)

    def save(self, path: Path) -> None:
        pd.DataFrame({"median": self.median_, "scale": self.scale_},
                     index=self.feature_names_).to_parquet(path)

    @classmethod
    def load(cls, path: Path, feature_names: list[str]) -> "WholeSetRobustZ":
        df = pd.read_parquet(path).reindex(feature_names)
        sc = cls()
        sc.median_ = df["median"].to_numpy(dtype=np.float64)
        sc.scale_ = df["scale"].to_numpy(dtype=np.float64)
        sc.feature_names_ = list(feature_names)
        return sc


class DailyCrossSectionMAD(Standardizer):
    """逐日截面 MAD去极值(±5) + zscore，无状态（对齐 ml_ht.dataset.preprocess）。"""

    name = "daily_cross_section_mad"
    stateful = False

    def fit(self, X: np.ndarray, dates: np.ndarray | None = None) -> "DailyCrossSectionMAD":
        return self   # 无状态

    def transform(self, X: np.ndarray, dates: np.ndarray | None = None) -> np.ndarray:
        if dates is None:
            raise ValueError("DailyCrossSectionMAD.transform 需要逐样本 dates 以按日分组")
        z = np.zeros_like(X, dtype=np.float32)
        for d in np.unique(dates):
            m = dates == d
            day = X[m]
            if len(day) < 2:
                continue
            med = np.median(day, axis=0, keepdims=True)
            mad = np.median(np.abs(day - med), axis=0, keepdims=True)
            mad = np.where(mad < 1e-8, 1.0, mad)
            lo, hi = med - 5.0 * _MAD * mad, med + 5.0 * _MAD * mad
            day = np.clip(day, lo, hi)
            mu = day.mean(axis=0, keepdims=True)
            sigma = day.std(axis=0, keepdims=True)
            sigma = np.where(sigma < 1e-8, 1.0, sigma)
            z[m] = (day - mu) / sigma
        return z
