"""evaluation 模块单测：合成数据 → 已知答案断言。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_shared.evaluation.ic import compute_ic_series, compute_ic_report
from alpha_shared.evaluation.layered import layered_backtest
from alpha_shared.evaluation.returns import build_forward_returns


# ──────────────────────────────────────────────────────────
# IC: 完美正相关 → IC ≈ 1
# ──────────────────────────────────────────────────────────


def test_ic_perfect_correlation():
    """因子 = 收益时，每日 spearman IC 应 = 1.0"""
    dates = pd.date_range("2020-01-01", periods=20)
    cols = [f"S{i}" for i in range(50)]
    rng = np.random.default_rng(42)
    factor = pd.DataFrame(rng.normal(size=(20, 50)), index=dates, columns=cols)
    forward_ret = factor.copy()  # 完美相关

    ic = compute_ic_series(factor, forward_ret, method="spearman")
    assert (ic.dropna() > 0.999).all(), f"IC should ≈ 1, got {ic.values}"


def test_ic_perfect_anticorrelation():
    """因子 = -收益时，spearman IC 应 = -1.0"""
    dates = pd.date_range("2020-01-01", periods=20)
    cols = [f"S{i}" for i in range(50)]
    rng = np.random.default_rng(123)
    factor = pd.DataFrame(rng.normal(size=(20, 50)), index=dates, columns=cols)
    forward_ret = -factor

    ic = compute_ic_series(factor, forward_ret, method="spearman")
    assert (ic.dropna() < -0.999).all()


def test_ic_report_columns():
    """compute_ic_report 应返回所有 horizon × 关键指标"""
    dates = pd.date_range("2020-01-01", periods=50)
    cols = [f"S{i}" for i in range(30)]
    rng = np.random.default_rng(7)
    factor = pd.DataFrame(rng.normal(size=(50, 30)), index=dates, columns=cols)
    fwd_returns = {
        h: pd.DataFrame(rng.normal(size=(50, 30)), index=dates, columns=cols)
        for h in (5, 10)
    }

    report, ic_series = compute_ic_report(factor, fwd_returns, method="spearman")
    assert set(report.index) == {"5d", "10d"}
    for col in ["ic_mean", "ic_std", "icir", "ic_t", "pct_positive"]:
        assert col in report.columns, f"missing {col}"
    assert set(ic_series.keys()) == {5, 10}


# ──────────────────────────────────────────────────────────
# Layered backtest：单调因子 → 单调分组收益
# ──────────────────────────────────────────────────────────


def test_layered_monotonicity():
    """构造完美单调因子：因子大的股票收益必然大。
    分组应严格单调，monotonicity ≈ +1，sharpe(LongShort) > 0。
    """
    n_days, n_stocks = 100, 50
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    cols = [f"S{i:02d}" for i in range(n_stocks)]

    rng = np.random.default_rng(0)
    # 因子：每只股票一个固定 rank（0..49）
    rank = np.arange(n_stocks, dtype=float)
    factor = pd.DataFrame(
        np.tile(rank, (n_days, 1)),
        index=dates,
        columns=cols,
    )
    # 1d return: 0.001 * rank + 噪声 → 高 rank 股票期望收益高
    return_1d = pd.DataFrame(
        0.001 * rank + rng.normal(scale=0.005, size=(n_days, n_stocks)),
        index=dates,
        columns=cols,
    )

    result = layered_backtest(factor, return_1d, n=5, g=5)

    # 完美单调因子：5 组年化收益严格递增
    summary = result["summary"]
    group_returns = [summary.loc[f"G{i}", "ann_return"] for i in range(1, 6)]
    assert all(group_returns[i] < group_returns[i + 1] for i in range(4)), (
        f"groups not strictly monotonic: {group_returns}"
    )
    assert summary.loc["LongShort", "sharpe"] > 0
    # monotonicity 至少为正（指数增长会让 Pearson 偏低，不强求 > 0.9）
    assert result["monotonicity"] > 0.5


# ──────────────────────────────────────────────────────────
# Forward returns: 已知 price_panel → 已知 return
# ──────────────────────────────────────────────────────────


def test_forward_returns_simple():
    """price[T+1+N] / price[T+1] - 1，shift(-N-1)。"""
    dates = pd.date_range("2020-01-01", periods=10)
    # 每只股票每天涨 1%
    price = pd.DataFrame(
        np.tile(1.01 ** np.arange(10).reshape(-1, 1), (1, 3)),
        index=dates,
        columns=["A", "B", "C"],
    )
    rets = build_forward_returns(price, horizons=[1])
    r1 = rets[1]
    # T=0: r = (price[2]/price[1]) - 1 = 1.01^2 / 1.01 - 1 = 1.01 - 1 = 0.01
    # 容差 1e-12
    expected = 0.01
    nan_mask = r1.isna()
    finite = r1.where(~nan_mask).dropna(how="all")
    np.testing.assert_allclose(
        finite.values, expected, atol=1e-12,
        err_msg=f"expected ~{expected}, got\n{finite}",
    )


def test_forward_returns_horizons_required():
    """horizons 必填，不传应报错。"""
    price = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(TypeError):
        build_forward_returns(price)
