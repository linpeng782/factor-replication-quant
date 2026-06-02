"""
RobustZScoreScaler —— 全集 per-feature 稳健标准化（fit/transform 分离）
============================================================
口径（对齐国金证券《之十》§1.2）：
  - **全集**（whole-dataset，非逐日截面）：每个特征用【训练段所有 (日×股) 单元】算一个
    median 和 MAD，全程用这一对，从而**保留跨日水位 + 抗极值**。
  - **fit 只用训练段**：median/MAD 仅由 train 段拟合；valid/test 仅 transform（防数据泄露）。
  - 公式：z = (x − median) / (1.4826 · MAD)，MAD = median(|x − median|)。
  - **不做额外 clip**（国金原文仅靠 MAD 自身抗极值，无截断步骤）。

用法：
    sc = RobustZScoreScaler().fit(train_long[feat_cols])
    z_train = sc.transform(train_long[feat_cols])
    z_test  = sc.transform(test_long[feat_cols])
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MAD_SCALE = 1.4826


class RobustZScoreScaler:
    """全集 per-feature 稳健标准化器。median_/scale_ 仅在 fit() 时由训练数据算出。"""

    def __init__(self) -> None:
        self.median_: pd.Series | None = None      # 每特征中位数
        self.scale_: pd.Series | None = None        # 每特征 1.4826·MAD（==0 → NaN，transform 后该列全 NaN）
        self.columns_: list[str] | None = None

    def fit(self, X: pd.DataFrame) -> "RobustZScoreScaler":
        """X: (样本, 特征) 训练段长表。逐列(axis=0)算 median 与 1.4826·MAD（nan-safe）。"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X 必须是 DataFrame（样本×特征）")
        self.columns_ = list(X.columns)
        med = X.median(axis=0, skipna=True)
        mad = (X.sub(med, axis=1)).abs().median(axis=0, skipna=True) * _MAD_SCALE
        # scale==0（常数列）→ NaN，避免除零；transform 后该列全 NaN，下游可剔除
        scale = mad.where(mad > 0, np.nan)
        self.median_ = med
        self.scale_ = scale
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """用 fit 好的 median_/scale_ 标准化 X（列对齐 columns_）。"""
        if self.median_ is None:
            raise RuntimeError("尚未 fit")
        X = X.reindex(columns=self.columns_)
        return X.sub(self.median_, axis=1).div(self.scale_, axis=1)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
