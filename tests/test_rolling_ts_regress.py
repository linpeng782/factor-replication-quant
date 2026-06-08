"""rolling_ts_regress 核心数学单测：矢量化 rolling OLS → δ → stat 对手算一致。"""
import numpy as np
import pandas as pd

from core.operators.rolling_ts_regress import _rolling_stat


def _ref_stat(r1, r2, R1, R2, W, min_obs):
    """朴素 Python 实现：逐窗口 OLS（np.polyfit）→ δ → t 值。"""
    T = len(r1)
    out = np.full(T, np.nan)
    for t in range(W - 1, T):
        sl = slice(t - W + 1, t + 1)
        y = np.concatenate([r1[sl], r2[sl]])
        X = np.concatenate([R1[sl], R2[sl]])
        m = np.isfinite(y) & np.isfinite(X)
        if m.sum() < 2:
            continue
        b, a = np.polyfit(X[m], y[m], 1)  # slope, intercept
        eps = np.where(m, y - (a + b * X), np.nan)
        delta = eps[:W] - eps[W:]
        dv = delta[np.isfinite(delta)]
        if len(dv) < min_obs:
            continue
        sd = dv.std(ddof=1)
        if sd > 1e-12:
            out[t] = dv.mean() / (sd / np.sqrt(len(dv)))
    return out


def test_rolling_stat_matches_naive():
    rng = np.random.RandomState(0)
    T = 60
    R1 = rng.randn(T) * 0.01
    R2 = rng.randn(T) * 0.01
    r1 = 0.8 * R1 + rng.randn(T) * 0.005
    r2 = 0.5 * R2 + rng.randn(T) * 0.005
    W = 20
    got = _rolling_stat(r1, r2, R1, R2, W, min_obs=10)
    ref = _ref_stat(r1, r2, R1, R2, W, min_obs=10)
    both = np.isfinite(got) & np.isfinite(ref)
    assert both.sum() > 30
    assert np.allclose(got[both], ref[both], atol=1e-9), \
        f"max diff = {np.abs(got[both]-ref[both]).max():.2e}"
    # NaN 位置一致
    assert np.array_equal(np.isnan(got), np.isnan(ref))


def test_rolling_stat_nan_warmup():
    """前 W-1 个必为 NaN。"""
    T, W = 30, 20
    x = np.ones(T)
    got = _rolling_stat(x, x, x, x, W, min_obs=5)
    assert np.isnan(got[: W - 1]).all()


def test_rolling_stat_handles_missing():
    """含 NaN 输入不崩，有效观测不足时输出 NaN。"""
    r1 = np.array([np.nan] * 25, dtype=float)
    r2 = np.ones(25)
    R1 = np.ones(25)
    R2 = np.ones(25)
    got = _rolling_stat(r1, r2, R1, R2, 20, min_obs=10)
    assert np.isnan(got).all()  # seg1 全 NaN → δ 不足
