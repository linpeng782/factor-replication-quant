"""
面板类滚动算子契约测试：rolling_weighted_mean / rolling_sorted_subset。

两者都走「长表 → (date × stock) 宽表 → numpy 向量化 → 按原行序写回」，
测试用小规模长表手算对拍，覆盖：加权归一化、指数衰减、NaN 跳过、
排序切割、mask 剔除无效日、min_periods/min_valid 守门。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.operators import Context, OpRegistry
from core.operators import rolling_sorted_subset, rolling_weighted_mean  # noqa: F401 触发注册


class _MockFetcher:
    pass


def _long(dates, stocks, cols: dict) -> pd.DataFrame:
    """按 (stock, date) 顺序构造长表；cols 值为 (n_stock, n_date) 嵌套列表。"""
    rows = []
    for si, s in enumerate(stocks):
        for di, d in enumerate(dates):
            row = {"order_book_id": s, "date": d}
            row.update({c: v[si][di] for c, v in cols.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def _run(action: str, df: pd.DataFrame, step: dict) -> pd.Series:
    ctx = Context(factor_name="t")
    ctx.dataframes["data"] = df
    OpRegistry.get(action)(ctx, step, _MockFetcher())
    return ctx.get_df("data")[step["output_column"]]


DATES = pd.date_range("2024-01-01", periods=6, freq="D")


# ── rolling_weighted_mean ────────────────────────────────


def test_weighted_mean_normalizes_by_weight_sum():
    df = _long(DATES, ["A"], {
        "r": [[0.01, 0.02, -0.01, 0.03, 0.00, -0.02]],
        "w": [[1.0, 3.0, 1.0, 1.0, 2.0, 2.0]],
    })
    out = _run("rolling_weighted_mean", df, {
        "source_column": "r", "weight_column": "w", "window": 3,
        "min_periods": 3, "output_column": "y"})
    # t=2: (1*0.01 + 3*0.02 + 1*(-0.01)) / 5
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest_approx((0.01 + 0.06 - 0.01) / 5.0)
    # t=5: (1*0.03 + 2*0.00 + 2*(-0.02)) / 5
    assert out.iloc[5] == pytest_approx((0.03 + 0.0 - 0.04) / 5.0)


def test_weighted_mean_exp_decay_and_nan_skip():
    df = _long(DATES, ["A"], {
        "r": [[0.01, np.nan, -0.01, 0.03, 0.02, -0.02]],
        "w": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
    })
    scale = 2.0
    out = _run("rolling_weighted_mean", df, {
        "source_column": "r", "weight_column": "w", "window": 3,
        "decay_scale": scale, "min_periods": 2, "output_column": "y"})
    # t=3 窗口 [t1(NaN 跳过), t2, t3]；衰减 i=0→t3, i=1→t2
    d0, d1 = 1.0, np.exp(-1 / scale)
    assert out.iloc[3] == pytest_approx((d0 * 0.03 + d1 * -0.01) / (d0 + d1))
    # t=2 窗口 [t0, t1(NaN), t2] → 有效 2 天，权重 exp(0)、exp(-2/scale)
    d2 = np.exp(-2 / scale)
    assert out.iloc[2] == pytest_approx((1.0 * -0.01 + d2 * 0.01) / (1.0 + d2))


def test_weighted_mean_min_periods_gate():
    df = _long(DATES, ["A"], {
        "r": [[0.01, np.nan, np.nan, np.nan, 0.02, 0.03]],
        "w": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
    })
    out = _run("rolling_weighted_mean", df, {
        "source_column": "r", "weight_column": "w", "window": 3,
        "min_periods": 2, "output_column": "y"})
    assert np.isnan(out.iloc[3])          # 窗口内仅 0 个有效日
    assert not np.isnan(out.iloc[5])      # 窗口内 2 个有效日


# ── rolling_sorted_subset ────────────────────────────────


def test_sorted_subset_low_sum():
    # 振幅升序取最低 floor(4*0.5)=2 天的收益和
    df = _long(DATES, ["A"], {
        "v": [[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]],
        "s": [[9.0, 1.0, 8.0, 2.0, 7.0, 3.0]],
    })
    out = _run("rolling_sorted_subset", df, {
        "source_column": "v", "source_column_sort": "s", "window": 4,
        "frac": 0.5, "select": "low", "agg": "sum", "min_valid": 4,
        "output_column": "y"})
    # t=3 窗口 t0..t3，s=[9,1,8,2] → 最低两天 t1(0.02)+t3(0.04)
    assert out.iloc[3] == pytest_approx(0.06)
    # t=5 窗口 t2..t5，s=[8,2,7,3] → t3(0.04)+t5(0.06)
    assert out.iloc[5] == pytest_approx(0.10)
    assert np.isnan(out.iloc[2])


def test_sorted_subset_high_and_mean():
    df = _long(DATES, ["A"], {
        "v": [[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]],
        "s": [[9.0, 1.0, 8.0, 2.0, 7.0, 3.0]],
    })
    out = _run("rolling_sorted_subset", df, {
        "source_column": "v", "source_column_sort": "s", "window": 4,
        "frac": 0.5, "select": "high", "agg": "mean", "min_valid": 4,
        "output_column": "y"})
    # t=3 窗口 s=[9,1,8,2] → 最高两天 t0(0.01)+t2(0.03)，mean=0.02
    assert out.iloc[3] == pytest_approx(0.02)


def test_sorted_subset_mask_excludes_days():
    # mask=0 的 t1 应完全不参与（既不排序也不计入有效日数）
    df = _long(DATES, ["A"], {
        "v": [[0.01, 99.0, 0.03, 0.04, 0.05, 0.06]],
        "s": [[9.0, 0.001, 8.0, 2.0, 7.0, 3.0]],
        "m": [[1.0, 0.0, 1.0, 1.0, 1.0, 1.0]],
    })
    out = _run("rolling_sorted_subset", df, {
        "source_column": "v", "source_column_sort": "s", "mask_column": "m",
        "window": 4, "frac": 0.5, "select": "low", "agg": "sum", "min_valid": 3,
        "output_column": "y"})
    # t=3 有效日 = t0,t2,t3（t1 被 mask 掉）→ k=floor(3*0.5)=1 → s 最小是 t3(2.0) → 0.04
    assert out.iloc[3] == pytest_approx(0.04)


def test_sorted_subset_multi_stock_alignment():
    df = _long(DATES, ["A", "B"], {
        "v": [[0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
              [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]],
        "s": [[9.0, 1.0, 8.0, 2.0, 7.0, 3.0],
              [1.0, 9.0, 2.0, 8.0, 3.0, 7.0]],
    })
    out = _run("rolling_sorted_subset", df, {
        "source_column": "v", "source_column_sort": "s", "window": 4,
        "frac": 0.5, "select": "low", "agg": "sum", "min_valid": 4,
        "output_column": "y"})
    df = df.assign(y=out)
    a = df[df.order_book_id == "A"].set_index("date")["y"]
    b = df[df.order_book_id == "B"].set_index("date")["y"]
    assert a.loc[DATES[3]] == pytest_approx(0.06)           # t1+t3
    assert b.loc[DATES[3]] == pytest_approx(0.10 + 0.30)    # t0+t2


def pytest_approx(x, tol=1e-6):
    """轻量近似比较（避免为单文件引入 pytest.approx 语义差异）。"""
    class _A:
        def __eq__(self, other):
            return abs(float(other) - float(x)) <= tol
        def __repr__(self):
            return f"≈{x}"
    return _A()
