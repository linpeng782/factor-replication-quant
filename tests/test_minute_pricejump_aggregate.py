"""
minute_pricejump_aggregate 算子冒烟测试（paper_33）
==================================================

跑 1 只股票的真实数据，验证：
  1. schema 正确（superset 列齐全）
  2. warmup 日（前 std_window 日）所有特征列为 NaN
  3. 跳跃守恒：peak + ridge + 适中跳跃 + valley = 240（完整交易日）
     等价表达：jump_count = peak_count + ridge_count + 适中跳跃数；
              jump_count + valley_count = 当日有效分钟数
  4. 缓存命中：第 2 次 _process_stock 应当从 parquet load
  5. 时区检查
  6. ridge_interval_n = max(ridge_count - 1, 0)
  7. peak ∩ ridge = ∅（互斥）
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import MINUTE_DATA_DIR
from core.operators.minute_pricejump_aggregate import (
    _SUPERSET_COLUMNS,
    _compute_one_stock,
    _process_stock,
    _resolve_cache_dir,
)


SMOKE_OB = "000001.XSHE"
SMOKE_STD_WINDOW = 20
SMOKE_STD_THRESHOLD = 1.0


@pytest.fixture(scope="module")
def stock_df():
    src = MINUTE_DATA_DIR / f"{SMOKE_OB}.parquet"
    assert src.exists(), f"测试需要源数据 {src}"
    return _compute_one_stock(SMOKE_OB, src, SMOKE_STD_WINDOW, SMOKE_STD_THRESHOLD)


def test_timezone_is_naive_or_shanghai():
    src = MINUTE_DATA_DIR / f"{SMOKE_OB}.parquet"
    raw = pd.read_parquet(src, columns=["datetime"])
    tz = raw["datetime"].dt.tz
    assert tz is None or str(tz) in (
        "Asia/Shanghai",
        "Asia/Shanghai+08:00",
    ), f"意外的时区 {tz!r}"


def test_schema_full_superset(stock_df: pd.DataFrame):
    expected = ["order_book_id", "date"] + _SUPERSET_COLUMNS
    assert list(stock_df.columns) == expected
    assert (stock_df["order_book_id"] == SMOKE_OB).all()


def test_warmup_days_are_nan(stock_df: pd.DataFrame):
    head = stock_df.iloc[:SMOKE_STD_WINDOW]
    assert head.loc[:, _SUPERSET_COLUMNS].isna().all().all(), (
        "warmup 日（前 std_window 日）所有特征列应为 NaN"
    )
    after = stock_df.iloc[SMOKE_STD_WINDOW : SMOKE_STD_WINDOW + 30]
    assert after["jump_count"].notna().all()


def test_jump_valley_complement(stock_df: pd.DataFrame):
    """jump_count + valley_count = 当日有效分钟数（完整交易日 240）。"""
    valid = stock_df.dropna(subset=["jump_count", "valley_count"])
    total = valid["jump_count"] + valid["valley_count"]
    assert (total > 0).all()
    assert (total <= 240).all(), f"单日总分钟数 > 240：{total[total > 240].head()}"
    full_day_ratio = (total == 240).mean()
    assert full_day_ratio > 0.9, f"完整 240 分钟交易日比例={full_day_ratio:.2f}"


def test_peak_ridge_disjoint(stock_df: pd.DataFrame):
    """价峰、价岭互斥：peak ∧ ridge = ∅，因此 peak_count + ridge_count ≤ jump_count。"""
    valid = stock_df.dropna(subset=["peak_count", "ridge_count", "jump_count"])
    sum_pr = valid["peak_count"] + valid["ridge_count"]
    assert (sum_pr <= valid["jump_count"]).all(), (
        "peak + ridge 不应超过 jump_count（peak/ridge 都是 jump 的子集）"
    )


def test_ridge_interval_n_equals_ridge_count_minus_one(stock_df: pd.DataFrame):
    valid = stock_df.dropna(subset=["ridge_count", "ridge_interval_n"])
    expected = (valid["ridge_count"] - 1).clip(lower=0)
    assert (valid["ridge_interval_n"] == expected).all()


def test_valley_vwap_sane(stock_df: pd.DataFrame):
    """valley_vwap 在 0 < daily_low ≤ valley_vwap ≤ daily_high 范围内（非 NaN 时）。"""
    valid = stock_df.dropna(subset=["valley_vwap", "daily_low", "daily_high"])
    valid = valid[valid["valley_vwap"] > 0]
    assert (valid["valley_vwap"] >= valid["daily_low"] * 0.99).all(), "valley_vwap < daily_low"
    assert (valid["valley_vwap"] <= valid["daily_high"] * 1.01).all(), "valley_vwap > daily_high"


def test_cache_hit_then_miss():
    cache_dir = _resolve_cache_dir("_smoketest_pj", SMOKE_STD_WINDOW, SMOKE_STD_THRESHOLD)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.time()
        df1, status1 = _process_stock(SMOKE_OB, SMOKE_STD_WINDOW, SMOKE_STD_THRESHOLD, cache_dir)
        t1 = time.time() - t0
        assert status1 == "miss"

        t0 = time.time()
        df2, status2 = _process_stock(SMOKE_OB, SMOKE_STD_WINDOW, SMOKE_STD_THRESHOLD, cache_dir)
        t2 = time.time() - t0
        assert status2 == "hit"
        assert t2 * 5 < t1

        assert df1.shape == df2.shape
        for col in df1.columns:
            a = df1[col].fillna(-9.99e30) if df1[col].dtype.kind == "f" else df1[col]
            b = df2[col].fillna(-9.99e30) if df2[col].dtype.kind == "f" else df2[col]
            assert (a == b).all(), f"cache 列 {col} 不一致"
    finally:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
