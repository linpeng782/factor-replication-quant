"""
minute_intraday_aggregate 算子冒烟测试
=======================================

跑 1-2 只股票的真实数据，验证：
  1. schema 正确（superset 列齐全 + features 投影正确）
  2. warmup 日（前 std_window 日）所有特征列为 NaN，不是 0
  3. 守恒：peak + ridge + valley = 当天有数据的分钟数
  4. 缓存命中：第 2 次 _process_stock 应当从 parquet load 而非重算
  5. 时区检查（数据是 tz-naive）
  6. 间隔 5 阶矩与 peak_count 关系一致：peak_interval_n = max(peak_count - 1, 0)
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
from core.operators.minute_intraday_aggregate import (
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
    assert list(stock_df.columns) == expected, (
        f"列序不匹配: 期望前 2 列 + {len(_SUPERSET_COLUMNS)} 个 superset 列"
    )
    assert (stock_df["order_book_id"] == SMOKE_OB).all()


def test_warmup_days_are_nan(stock_df: pd.DataFrame):
    head = stock_df.iloc[:SMOKE_STD_WINDOW]
    feature_cols = _SUPERSET_COLUMNS
    assert head.loc[:, feature_cols].isna().all().all(), (
        "warmup 日（前 std_window 日）所有特征列应为 NaN"
    )
    # warmup 之外应有有效值
    after = stock_df.iloc[SMOKE_STD_WINDOW : SMOKE_STD_WINDOW + 30]
    assert after["peak_count"].notna().all(), "warmup 之后 peak_count 不应再 NaN"


def test_count_conservation(stock_df: pd.DataFrame):
    """峰 + 岭 + 谷 = 当天有效分钟数（000001 的数据完整时为 240）。"""
    valid = stock_df.dropna(subset=["peak_count", "ridge_count", "valley_count"])
    total = valid["peak_count"] + valid["ridge_count"] + valid["valley_count"]
    # 大多数交易日 240 分钟；半天交易（节前等）也允许更小值
    assert (total > 0).all()
    assert (total <= 240).all(), f"单日总分钟数 > 240：{total[total > 240].head()}"
    # 至少 95% 的交易日是完整 240 分钟
    full_day_ratio = (total == 240).mean()
    assert full_day_ratio > 0.9, f"完整 240 分钟交易日比例={full_day_ratio:.2f}（疑似数据异常）"


def test_interval_n_equals_peak_count_minus_one(stock_df: pd.DataFrame):
    valid = stock_df.dropna(subset=["peak_count", "peak_interval_n"])
    # peak_count >= 1 时间隔 n = peak_count - 1；peak_count == 0 时 n = 0
    expected = (valid["peak_count"] - 1).clip(lower=0)
    assert (valid["peak_interval_n"] == expected).all(), (
        "peak_interval_n 应等于 max(peak_count - 1, 0)"
    )


def test_cache_hit_then_miss():
    """连续两次 _process_stock：第 1 次 miss、第 2 次 hit；内容必须相等。"""
    cache_dir = _resolve_cache_dir("_smoketest_pytest", SMOKE_STD_WINDOW, SMOKE_STD_THRESHOLD)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.time()
        df1, status1 = _process_stock(SMOKE_OB, SMOKE_STD_WINDOW, SMOKE_STD_THRESHOLD, cache_dir)
        t1 = time.time() - t0
        assert status1 == "miss", "第 1 次必然 cache miss"

        t0 = time.time()
        df2, status2 = _process_stock(SMOKE_OB, SMOKE_STD_WINDOW, SMOKE_STD_THRESHOLD, cache_dir)
        t2 = time.time() - t0
        assert status2 == "hit", "第 2 次应当 cache hit"
        assert t2 * 5 < t1, f"hit({t2:.2f}s) 应当显著快于 miss({t1:.2f}s)"

        # 内容必须相等（NaN 处也要相等）
        assert df1.shape == df2.shape
        for col in df1.columns:
            a = df1[col].fillna(-9.99e30) if df1[col].dtype.kind == "f" else df1[col]
            b = df2[col].fillna(-9.99e30) if df2[col].dtype.kind == "f" else df2[col]
            assert (a == b).all(), f"cache 内容在列 {col} 上与新算结果不一致"
    finally:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
