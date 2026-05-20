"""
算子拓展单元测试
验证:
1. rank 算子: pct/ascending/method 参数
2. transform 算子: ffill/bfill method
3. fetch 算子: custom API
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.operators import OpRegistry
from core.operators import fetch, compute, filter, rank, transform, rolling  # noqa: F401 触发注册


class MockFetcher:
    """Mock DataFetcher，不依赖 rqdatac"""

    def __init__(self):
        self._rq = None

    def get_factor(self, order_book_ids, field, date=None, start_date=None, end_date=None):
        return pd.DataFrame()

    def fetch_pit(self, order_book_ids, fields, start_quarter, end_quarter, statements="latest"):
        return pd.DataFrame()


# ──────────────────────────────────────────
# 测试 1: rank 算子
# ──────────────────────────────────────────

def test_rank_pct_false():
    """测试顺序排名 (pct=False)"""
    ctx = {}
    fetcher = MockFetcher()
    df = pd.DataFrame({
        "order_book_id": ["A", "B", "C", "D"],
        "roic_ttm": [0.15, 0.12, 0.10, 0.08],
    })
    ctx["data"] = df

    step = {
        "action": "rank",
        "method": "rank",
        "rank_column": "roic_ttm",
        "pct": False,
        "ascending": False,
        "output": "ranked",
    }

    op = OpRegistry.get("rank")
    result = op(ctx, step, fetcher)

    ranks = result["roic_ttm_rank"].tolist()
    assert ranks == [1.0, 2.0, 3.0, 4.0], f"顺序排名错误: {ranks}"
    print("✅ test_rank_pct_false 通过")


def test_rank_method_min():
    """测试 method='min' 并列处理"""
    ctx = {}
    fetcher = MockFetcher()
    df = pd.DataFrame({
        "order_book_id": ["A", "B", "C", "D"],
        "roic_ttm": [0.15, 0.12, 0.12, 0.08],  # B 和 C 并列
    })
    ctx["data"] = df

    step = {
        "action": "rank",
        "method": "rank",
        "rank_column": "roic_ttm",
        "pct": False,
        "ascending": False,
        "rank_method": "min",
        "output": "ranked",
    }

    op = OpRegistry.get("rank")
    result = op(ctx, step, fetcher)

    ranks = result["roic_ttm_rank"].tolist()
    assert ranks == [1.0, 2.0, 2.0, 4.0], f"method=min 并列处理错误: {ranks}"
    print("✅ test_rank_method_min 通过")


def test_rank_industry_rank():
    """测试行业内排名"""
    ctx = {}
    fetcher = MockFetcher()
    df = pd.DataFrame({
        "order_book_id": ["A", "B", "C", "D", "E", "F"],
        "roic_ttm": [0.15, 0.12, 0.10, 0.20, 0.18, 0.05],
        "industry": ["地产", "地产", "地产", "医药", "医药", "医药"],
    })
    ctx["data"] = df

    step = {
        "action": "rank",
        "method": "industry_rank",
        "rank_column": "roic_ttm",
        "group_column": "industry",
        "pct": False,
        "ascending": False,
        "rank_method": "min",
        "output": "ranked",
    }

    op = OpRegistry.get("rank")
    result = op(ctx, step, fetcher)

    # 地产行业: A(0.15)=1, B(0.12)=2, C(0.10)=3
    # 医药行业: D(0.20)=1, E(0.18)=2, F(0.05)=3
    expected = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    ranks = result["roic_ttm_rank"].tolist()
    assert ranks == expected, f"行业内排名错误: {ranks}"
    print("✅ test_rank_industry_rank 通过")


def test_rank_backward_compatible():
    """测试向后兼容（旧参数依然有效）"""
    ctx = {}
    fetcher = MockFetcher()
    df = pd.DataFrame({
        "order_book_id": ["A", "B", "C"],
        "roic_ttm": [0.15, 0.12, 0.10],
    })
    ctx["data"] = df

    # 旧式调用（无 pct/ascending/rank_method）
    step = {
        "action": "rank",
        "method": "rank",
        "rank_column": "roic_ttm",
        "output": "ranked",
    }

    op = OpRegistry.get("rank")
    result = op(ctx, step, fetcher)

    # 默认 pct=True, ascending=True, method="average"
    ranks = result["roic_ttm_rank"].tolist()
    assert len(ranks) == 3
    assert all(0 <= r <= 1 for r in ranks), f"百分比排名范围错误: {ranks}"
    print("✅ test_rank_backward_compatible 通过")


def test_rank_industry_rank_with_date():
    """测试 long 格式带 date 列的行业内排名（按 [date, industry] 分组）"""
    ctx = {}
    fetcher = MockFetcher()
    df = pd.DataFrame({
        "order_book_id": ["A", "B", "C", "D", "E", "F"] * 2,
        "date": ["2024-01-01"] * 6 + ["2024-01-02"] * 6,
        "roic_ttm": [0.15, 0.12, 0.10, 0.20, 0.18, 0.05,
                     0.14, 0.11, 0.09, 0.19, 0.17, 0.04],
        "industry": ["地产", "地产", "地产", "医药", "医药", "医药"] * 2,
    })
    ctx["data"] = df

    step = {
        "action": "rank",
        "method": "industry_rank",
        "rank_column": "roic_ttm",
        "group_column": "industry",
        "pct": False,
        "ascending": False,
        "rank_method": "min",
        "output": "ranked",
    }

    op = OpRegistry.get("rank")
    result = op(ctx, step, fetcher)

    # 每天每个行业内独立排名
    # 2024-01-01 地产: A(0.15)=1, B(0.12)=2, C(0.10)=3
    # 2024-01-01 医药: D(0.20)=1, E(0.18)=2, F(0.05)=3
    # 2024-01-02 地产: A(0.14)=1, B(0.11)=2, C(0.09)=3
    # 2024-01-02 医药: D(0.19)=1, E(0.17)=2, F(0.04)=3
    expected = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0] * 2
    ranks = result["roic_ttm_rank"].tolist()
    assert ranks == expected, f"带日期的行业内排名错误: {ranks}"
    print("✅ test_rank_industry_rank_with_date 通过")


def test_rank_market_rank_with_date():
    """测试 long 格式带 date 列的全市场排名（按 date 分组）"""
    ctx = {}
    fetcher = MockFetcher()
    df = pd.DataFrame({
        "order_book_id": ["A", "B", "C", "D"] * 2,
        "date": ["2024-01-01"] * 4 + ["2024-01-02"] * 4,
        "roic_ttm": [0.15, 0.12, 0.10, 0.08,
                     0.14, 0.11, 0.09, 0.07],
    })
    ctx["data"] = df

    step = {
        "action": "rank",
        "method": "rank",
        "rank_column": "roic_ttm",
        "pct": False,
        "ascending": False,
        "output": "ranked",
    }

    op = OpRegistry.get("rank")
    result = op(ctx, step, fetcher)

    # 每天全市场独立排名
    # 2024-01-01: A(0.15)=1, B(0.12)=2, C(0.10)=3, D(0.08)=4
    # 2024-01-02: A(0.14)=1, B(0.11)=2, C(0.09)=3, D(0.07)=4
    expected = [1.0, 2.0, 3.0, 4.0] * 2
    ranks = result["roic_ttm_rank"].tolist()
    assert ranks == expected, f"带日期的全市场排名错误: {ranks}"
    print("✅ test_rank_market_rank_with_date 通过")


# ──────────────────────────────────────────
# 测试 2: transform 算子 (ffill/bfill)
# ──────────────────────────────────────────

def test_transform_ffill_wide():
    """测试 wide 格式 ffill"""
    ctx = {}
    fetcher = MockFetcher()
    dates = pd.date_range("2024-01-01", periods=5)
    df = pd.DataFrame(
        {
            "A": [1.0, np.nan, np.nan, 2.0, np.nan],
            "B": [np.nan, 3.0, np.nan, np.nan, 4.0],
        },
        index=dates,
    )
    ctx["data"] = df

    step = {
        "action": "transform",
        "method": "ffill",
        "input": "data",
        "output": "filled",
    }

    op = OpRegistry.get("transform")
    result = op(ctx, step, fetcher)

    assert result["A"].tolist() == [1.0, 1.0, 1.0, 2.0, 2.0], f"A 列 ffill 错误: {result['A'].tolist()}"
    assert result["B"].tolist()[1:5] == [3.0, 3.0, 3.0, 4.0], f"B 列 ffill 错误: {result['B'].tolist()}"
    print("✅ test_transform_ffill_wide 通过")


def test_transform_ffill_long():
    """测试 long 格式 ffill (groupby)"""
    ctx = {}
    fetcher = MockFetcher()
    df = pd.DataFrame({
        "order_book_id": ["A", "A", "A", "B", "B", "B"],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"] * 2,
        "value": [1.0, np.nan, np.nan, np.nan, 2.0, np.nan],
    })
    ctx["data"] = df

    step = {
        "action": "transform",
        "method": "ffill",
        "input": "data",
        "group_column": "order_book_id",
        "columns": ["value"],
        "output": "filled",
    }

    op = OpRegistry.get("transform")
    result = op(ctx, step, fetcher)

    expected_A = [1.0, 1.0, 1.0]
    expected_B = [np.nan, 2.0, 2.0]  # B 第一行 NaN，后面 ffill
    actual_A = result[result.order_book_id == "A"]["value"].tolist()
    actual_B = result[result.order_book_id == "B"]["value"].tolist()
    assert actual_A == expected_A
    # NaN 不能直接 == 比较
    assert np.isnan(actual_B[0]) and actual_B[1:] == [2.0, 2.0]
    print("✅ test_transform_ffill_long 通过")


def test_transform_bfill():
    """测试 bfill"""
    ctx = {}
    fetcher = MockFetcher()
    df = pd.DataFrame(
        {
            "A": [np.nan, np.nan, 2.0, np.nan, 4.0],
        },
        index=pd.date_range("2024-01-01", periods=5),
    )
    ctx["data"] = df

    step = {
        "action": "transform",
        "method": "bfill",
        "input": "data",
        "output": "filled",
    }

    op = OpRegistry.get("transform")
    result = op(ctx, step, fetcher)

    assert result["A"].tolist() == [2.0, 2.0, 2.0, 4.0, 4.0], f"bfill 错误: {result['A'].tolist()}"
    print("✅ test_transform_bfill 通过")


# ──────────────────────────────────────────
# 测试 3: fetch 算子 (custom API)
# ──────────────────────────────────────────

class MockFetcherWithCustom:
    """支持自定义 API 的 Mock Fetcher"""

    def __init__(self):
        class MockClient:
            def execute(self, command):
                if command == "__test__industry":
                    return [
                        ["地产", "000001.XSHE", "2020-01-01"],
                        ["医药", "000002.XSHE", "2020-01-01"],
                        ["地产", "000001.XSHE", "2023-01-01"],  # 变更
                    ]
                raise ValueError(f"未知命令: {command}")

        class MockRQ:
            client = type("Client", (), {"get_client": lambda self: MockClient()})()

        self._rq = MockRQ()

    def get_factor(self, *args, **kwargs):
        return pd.DataFrame()

    def fetch_pit(self, *args, **kwargs):
        return pd.DataFrame()


# ──────────────────────────────────────────
# 测试 4: rolling 算子
# ──────────────────────────────────────────

def test_rolling_plain_wide():
    """测试普通 rolling（wide 格式，模拟 MA5）"""
    ctx = {}
    fetcher = MockFetcher()
    dates = pd.date_range("2024-01-01", periods=10)
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "B": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        },
        index=dates,
    )
    ctx["data"] = df

    step = {
        "action": "rolling",
        "input": "data",
        "window": 5,
        "min_periods": 3,
        "agg": "mean",
        "fill_method": "none",
        "output": "rolled",
    }

    op = OpRegistry.get("rolling")
    result = op(ctx, step, fetcher)

    # A 列 MA5 (min_periods=3)
    # idx0: [1] -> NaN (不足3)
    # idx1: [1,2] -> NaN (不足3)
    # idx2: [1,2,3] -> 2.0
    # idx3: [1,2,3,4] -> 2.5
    # idx4: [1,2,3,4,5] -> 3.0
    # idx5: [2,3,4,5,6] -> 4.0
    expected_A = [np.nan, np.nan, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    actual_A = result["A"].tolist()
    for i, (a, e) in enumerate(zip(actual_A, expected_A)):
        if pd.isna(e):
            assert pd.isna(a), f"A 列 idx{i} 应为 NaN, 实际={a}"
        else:
            assert abs(a - e) < 1e-10, f"A 列 idx{i} 错误: 期望={e}, 实际={a}"
    print("✅ test_rolling_plain_wide 通过")


def test_rolling_on_changes():
    """测试变化日 rolling（roic_ttm_ind_rnk8 核心逻辑）"""
    ctx = {}
    fetcher = MockFetcher()
    dates = pd.date_range("2024-01-01", periods=20)

    # 模拟某只股票的 ROIC_TTM（每5天变化一次，共4个财报期）
    roic_ttm = pd.DataFrame(
        {"stock": [0.10] * 5 + [0.08] * 5 + [0.06] * 5 + [0.04] * 5},
        index=dates,
    )

    # 模拟该股票的行业内排名（随 ROIC_TTM 变化）
    ind_rank = pd.DataFrame(
        {"stock": [5.0] * 5 + [8.0] * 5 + [12.0] * 5 + [20.0] * 5},
        index=dates,
    )

    ctx["roic_ttm"] = roic_ttm
    ctx["ind_rank"] = ind_rank

    step = {
        "action": "rolling",
        "input": "ind_rank",
        "window": 3,
        "min_periods": 1,
        "agg": "min",
        "on": "roic_ttm",       # 只在 ROIC_TTM 变化的日子上 rolling
        "fill_method": "ffill",
        "columns": ["stock"],
        "output": "factor",
    }

    op = OpRegistry.get("rolling")
    result = op(ctx, step, fetcher)

    # 变化日: idx0(0.10), idx5(0.08), idx10(0.06), idx15(0.04)
    # 变化日排名: 5, 8, 12, 20
    # rolling(3, min_periods=1).min():
    #   idx0: [5] -> min=5
    #   idx5: [5,8] -> min=5
    #   idx10: [5,8,12] -> min=5
    #   idx15: [8,12,20] -> min=8
    # ffill 后所有行:
    #   idx0-4: 5
    #   idx5-9: 5
    #   idx10-14: 5
    #   idx15-19: 8

    expected = [5.0] * 15 + [8.0] * 5
    actual = result["stock"].tolist()
    assert actual == expected, f"变化日 rolling 错误: 期望={expected}, 实际={actual}"
    print("✅ test_rolling_on_changes 通过")


def test_rolling_on_changes_with_nan():
    """测试变化日 rolling 时源数据有 NaN 的情况"""
    ctx = {}
    fetcher = MockFetcher()
    dates = pd.date_range("2024-01-01", periods=10)

    # ROIC_TTM: 前3天 NaN，然后变化
    roic_ttm = pd.DataFrame(
        {"stock": [np.nan, np.nan, np.nan, 0.10, 0.10, 0.10, 0.08, 0.08, 0.08, 0.08]},
        index=dates,
    )

    ind_rank = pd.DataFrame(
        {"stock": [np.nan, np.nan, np.nan, 10.0, 10.0, 10.0, 15.0, 15.0, 15.0, 15.0]},
        index=dates,
    )

    ctx["roic_ttm"] = roic_ttm
    ctx["ind_rank"] = ind_rank

    step = {
        "action": "rolling",
        "input": "ind_rank",
        "window": 3,
        "min_periods": 1,
        "agg": "min",
        "on": "roic_ttm",
        "fill_method": "ffill",
        "columns": ["stock"],
        "output": "factor",
    }

    op = OpRegistry.get("rolling")
    result = op(ctx, step, fetcher)

    # 变化日: idx3(0.10), idx6(0.08)
    # 变化日排名: 10, 15
    # rolling: idx3->10, idx6->min(10,15)=10
    # ffill: idx0-2 NaN, idx3-5=10, idx6-9=10
    expected = [np.nan, np.nan, np.nan, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    actual = result["stock"].tolist()
    for i, (a, e) in enumerate(zip(actual, expected)):
        if pd.isna(e):
            assert pd.isna(a), f"idx{i} 应为 NaN, 实际={a}"
        else:
            assert a == e, f"idx{i} 错误: 期望={e}, 实际={a}"
    print("✅ test_rolling_on_changes_with_nan 通过")


def test_rolling_min_periods():
    """测试 min_periods 不足时返回 NaN"""
    ctx = {}
    fetcher = MockFetcher()
    dates = pd.date_range("2024-01-01", periods=10)

    roic_ttm = pd.DataFrame(
        {"stock": [0.10] * 3 + [0.08] * 3 + [0.06] * 4},
        index=dates,
    )
    ind_rank = pd.DataFrame(
        {"stock": [5.0] * 3 + [8.0] * 3 + [12.0] * 4},
        index=dates,
    )

    ctx["roic_ttm"] = roic_ttm
    ctx["ind_rank"] = ind_rank

    step = {
        "action": "rolling",
        "input": "ind_rank",
        "window": 5,
        "min_periods": 5,       # 要求至少5个变化日
        "agg": "min",
        "on": "roic_ttm",
        "fill_method": "ffill",
        "columns": ["stock"],
        "output": "factor",
    }

    op = OpRegistry.get("rolling")
    result = op(ctx, step, fetcher)

    # 变化日只有3个(0.10, 0.08, 0.06)，不足5个
    # 所以所有变化日的 rolling 结果都是 NaN
    # ffill 后也是 NaN
    assert result["stock"].isna().all(), f"min_periods=5 但只有3个变化日，应全为 NaN"
    print("✅ test_rolling_min_periods 通过")


# ──────────────────────────────────────────
# 测试 3: fetch 算子 (custom API)
# ──────────────────────────────────────────

def test_fetch_custom_api():
    """测试 fetch 的 custom API 模式"""
    ctx = {"_universe": ["000001.XSHE", "000002.XSHE"]}
    fetcher = MockFetcherWithCustom()

    step = {
        "action": "fetch",
        "api": "custom",
        "command": "__test__industry",
        "columns": ["industry", "order_book_id", "start_date"],
        "output": "industry_map",
    }

    op = OpRegistry.get("fetch")
    result = op(ctx, step, fetcher)

    assert len(result) == 3
    assert list(result.columns) == ["industry", "order_book_id", "start_date"]
    assert result["industry"].tolist() == ["地产", "医药", "地产"]
    print("✅ test_fetch_custom_api 通过")


# ──────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("算子拓展单元测试")
    print("=" * 50)

    test_rank_pct_false()
    test_rank_method_min()
    test_rank_industry_rank()
    test_rank_backward_compatible()
    test_rank_industry_rank_with_date()
    test_rank_market_rank_with_date()

    test_transform_ffill_wide()
    test_transform_ffill_long()
    test_transform_bfill()

    test_fetch_custom_api()

    test_rolling_plain_wide()
    test_rolling_on_changes()
    test_rolling_on_changes_with_nan()
    test_rolling_min_periods()

    print("=" * 50)
    print("全部测试通过 ✅")
    print("=" * 50)
