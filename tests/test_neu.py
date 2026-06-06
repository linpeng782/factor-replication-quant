"""neutralize 对拍 statsmodels.OLS（FWL ≡ 原版逐日满哑变量回归）+ 边界用例。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from alpha_shared.neu import neutralize


def _make_panels(n_days=30, n_stocks=60, n_ind=6, seed=0, nan_frac=0.1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    stocks = [f"S{i:03d}" for i in range(n_stocks)]
    factor = pd.DataFrame(rng.normal(size=(n_days, n_stocks)), index=dates, columns=stocks)
    size = pd.DataFrame(rng.uniform(20, 5000, size=(n_days, n_stocks)), index=dates, columns=stocks)
    ind_labels = np.array([f"IND{j}" for j in range(n_ind)])
    industry = pd.DataFrame(
        ind_labels[rng.integers(0, n_ind, size=(n_days, n_stocks))], index=dates, columns=stocks
    )
    # 随机打洞测三重交集
    holes = rng.random((n_days, n_stocks)) < nan_frac
    factor = factor.mask(holes)
    return factor, industry, size


def _ols_resid_one_day(y, ind, s):
    """原版口径：满哑变量 + log市值, hasconst=False, 取残差。"""
    D = pd.get_dummies(ind).astype(float)
    X = pd.concat([D, np.log(s).rename("log_sz")], axis=1)
    return sm.OLS(y.astype(float), X.astype(float), hasconst=False, missing="drop").fit().resid


def test_fwl_equals_ols_multiday():
    factor, industry, size = _make_panels()
    out = neutralize(factor, industry, size, min_samples=2, restandardize=False)

    max_diff = 0.0
    for d in factor.index:
        mask = factor.loc[d].notna() & industry.loc[d].notna() & size.loc[d].notna()
        cols = mask[mask].index
        if len(cols) < 3:
            continue
        ref = _ols_resid_one_day(factor.loc[d, cols], industry.loc[d, cols], size.loc[d, cols])
        got = out.loc[d, cols]
        max_diff = max(max_diff, float((ref - got).abs().max()))
    assert max_diff < 1e-10, f"FWL vs OLS max_diff={max_diff:.2e}"


def test_singleton_industry_resid_zero():
    """独占某行业的股票，当天残差应为 0（满哑变量完美拟合）。"""
    dates = pd.bdate_range("2020-01-01", periods=1)
    stocks = ["A", "B", "C", "D"]
    factor = pd.DataFrame([[0.5, 1.5, -0.3, 0.9]], index=dates, columns=stocks)
    industry = pd.DataFrame([["X", "X", "X", "LONE"]], index=dates, columns=stocks)  # D 独占 LONE
    size = pd.DataFrame([[100.0, 200.0, 300.0, 400.0]], index=dates, columns=stocks)
    out = neutralize(factor, industry, size, min_samples=2, restandardize=False)
    assert abs(out.loc[dates[0], "D"]) < 1e-12


def test_min_samples_guard():
    """有效股票数 < max(行业数+2, min_samples) 的日子整天 NaN。"""
    factor, industry, size = _make_panels(n_days=5, n_stocks=10, n_ind=3)
    out = neutralize(factor, industry, size, min_samples=50, restandardize=False)  # 阈值远超10股
    assert out.notna().sum().sum() == 0


def test_grid_alignment_reindex():
    """industry/size 形状/股票集与 factor 不同时，按 factor 网格对齐，缺失处 NaN。"""
    factor, industry, size = _make_panels(n_days=10, n_stocks=40)
    # industry 少几只股 + size 少几天 → 这些位置中性化后应为 NaN
    industry2 = industry.iloc[:, :35]
    size2 = size.iloc[3:, :]
    out = neutralize(factor, industry2, size2, min_samples=2, restandardize=False)
    assert out.shape == factor.shape
    # 被砍掉的股票列（35:40）应全 NaN
    assert out.iloc[:, 35:].notna().sum().sum() == 0
    # 被砍掉的日期（前3天）应全 NaN
    assert out.iloc[:3].notna().sum().sum() == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✅ {name}")
    print("ALL PASSED")
