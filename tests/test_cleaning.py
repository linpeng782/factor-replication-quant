"""cleaning 模块单测：MAD / zscore / inf 处理 / mask 加载。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_shared.cleaning.preprocess import (
    prepare_factor,
    standardize_zscore,
    winsorize_mad,
)


# ──────────────────────────────────────────────────────────
# winsorize_mad
# ──────────────────────────────────────────────────────────


def test_winsorize_mad_clips_outliers():
    """3-MAD 之外的极端值应被 clip 到 boundary。"""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [2.0, 3.0, 4.0, 5.0, 6.0],
        "c": [3.0, 4.0, 5.0, 6.0, 7.0],
        "outlier": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
    })
    out = winsorize_mad(df, n=3.0)
    # 极端值应明显小于 1000
    assert (out["outlier"] < 100).all(), f"outlier 未被 clip：{out['outlier'].values}"


def test_winsorize_mad_preserves_nan():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 4.0, np.nan]})
    out = winsorize_mad(df, n=3.0)
    assert out.isna().equals(df.isna())


def test_winsorize_mad_zero_mad_row():
    """常数行（mad=0）应原样保留，不被压缩。"""
    df = pd.DataFrame({"a": [5.0], "b": [5.0], "c": [5.0]})
    out = winsorize_mad(df, n=3.0)
    np.testing.assert_array_equal(out.values, df.values)


# ──────────────────────────────────────────────────────────
# standardize_zscore
# ──────────────────────────────────────────────────────────


def test_zscore_mean_zero_std_one():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    df = pd.DataFrame([df.values.flatten()] * 1, columns=df["a"].astype(str))
    # 单行横截面
    df = pd.DataFrame({"S0": [1.0], "S1": [2.0], "S2": [3.0], "S3": [4.0], "S4": [5.0]})
    out = standardize_zscore(df)
    np.testing.assert_allclose(out.values.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(out.values.std(ddof=0), 1.0, atol=1e-12)


def test_zscore_zero_std_row_to_nan():
    """std=0 的行应整行 NaN（避免除零）。"""
    df = pd.DataFrame({"a": [3.0], "b": [3.0], "c": [3.0]})
    out = standardize_zscore(df)
    assert out.isna().all().all()


# ──────────────────────────────────────────────────────────
# prepare_factor: end-to-end + inf 处理
# ──────────────────────────────────────────────────────────


def test_prepare_factor_inf_handled():
    """YOLO 引擎可能产 inf → 必须先变 NaN，否则会污染整个截面分布。"""
    cols = ["A", "B", "C", "D", "E"]
    dates = pd.date_range("2020-01-01", periods=3)
    factor = pd.DataFrame(
        [[1.0, 2.0, np.inf, 4.0, 5.0],
         [2.0, 3.0, 4.0, 5.0, -np.inf],
         [3.0, 4.0, 5.0, 6.0, 7.0]],
        index=dates, columns=cols,
    )
    can_buy_mask = pd.DataFrame(True, index=dates, columns=cols)
    not_limit_up_mask = pd.DataFrame(True, index=dates, columns=cols)

    out = prepare_factor(factor, can_buy_mask, not_limit_up_mask, mad_n=3.0)
    # 第一行 inf 已变 NaN（其他列做了正常 zscore，不受 inf 污染）
    finite_first_row = out.iloc[0].dropna()
    assert len(finite_first_row) == 4  # 5 列中除 inf 那列剩 4 列
    np.testing.assert_allclose(finite_first_row.mean(), 0.0, atol=1e-10)


def test_prepare_factor_mask_applied():
    cols = ["A", "B", "C"]
    dates = pd.date_range("2020-01-01", periods=2)
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        index=dates, columns=cols,
    )
    # B 列 can_buy_mask=False
    can_buy_mask = pd.DataFrame(
        [[True, False, True], [True, False, True]],
        index=dates, columns=cols,
    )
    # C 列 not_limit_up_mask=False
    not_limit_up_mask = pd.DataFrame(
        [[True, True, False], [True, True, False]],
        index=dates, columns=cols,
    )
    out = prepare_factor(factor, can_buy_mask, not_limit_up_mask, mad_n=3.0)
    # B 应整列 NaN（can_buy_mask False），C 应整列 NaN（not_limit_up_mask False，post-标准化阶段 mask 掉）
    assert out["B"].isna().all()
    assert out["C"].isna().all()
