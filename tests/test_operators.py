"""
算子契约单元测试（rank / transform / rolling / fetch-custom）。

新契约要点（与历史版本不同）：
  - ctx 是 core.operators.Context dataclass，不是裸 dict
  - 所有算子用 source_column / output_column 字段；rank 还要 group_by: list
  - 算子 mutate ctx 并返回 None；测试读 ctx.get_df('data')[output_column]
  - transform / rolling / rank 都是 long 表（必须有 order_book_id 列；rank 截面要 date）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.operators import Context, OpRegistry
from core.operators import compute, fetch, filter, rank, rolling, transform  # noqa: F401 触发注册


# ──────────────────────────────────────────────────────────
# 测试工具
# ──────────────────────────────────────────────────────────


class MockFetcher:
    """无网 mock；测试不命中 rqdatac，下游 fetch 用例显式构造。"""

    def __init__(self):
        self._rq = None

    def get_factor(self, *args, **kwargs):
        return pd.DataFrame()

    def fetch_pit(self, *args, **kwargs):
        return pd.DataFrame()


def _ctx_with(name: str, df: pd.DataFrame, universe=None) -> Context:
    c = Context(factor_name="test")
    c.dataframes[name] = df
    if universe is not None:
        c.universe = universe
    return c


def _run(action: str, ctx: Context, step: dict) -> Context:
    OpRegistry.get(action)(ctx, step, MockFetcher())
    return ctx


# ──────────────────────────────────────────────────────────
# rank
# ──────────────────────────────────────────────────────────


def test_rank_cross_section_descending():
    """单截面（单日）按 ROIC 降序排名。"""
    df = pd.DataFrame({
        "date": ["2024-01-01"] * 4,
        "order_book_id": ["A", "B", "C", "D"],
        "roic": [0.15, 0.12, 0.10, 0.08],
    })
    ctx = _ctx_with("data", df)
    _run("rank", ctx, {
        "action": "rank",
        "source_column": "roic",
        "output_column": "roic_rk",
        "group_by": ["date"],
        "ascending": False,
        "rank_method": "min",
    })
    assert ctx.get_df("data")["roic_rk"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_rank_industry_cross_section():
    """行业内 × 日截面排名（group_by=[date, industry]）。"""
    df = pd.DataFrame({
        "date": ["2024-01-01"] * 6,
        "order_book_id": ["A", "B", "C", "D", "E", "F"],
        "roic": [0.15, 0.12, 0.10, 0.20, 0.18, 0.05],
        "industry": ["地产", "地产", "地产", "医药", "医药", "医药"],
    })
    ctx = _ctx_with("data", df)
    _run("rank", ctx, {
        "action": "rank",
        "source_column": "roic",
        "output_column": "roic_rk",
        "group_by": ["date", "industry"],
        "ascending": False,
        "rank_method": "min",
    })
    # 地产: 0.15→1, 0.12→2, 0.10→3 ；医药: 0.20→1, 0.18→2, 0.05→3
    assert ctx.get_df("data")["roic_rk"].tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]


def test_rank_n_bins():
    """n_bins 模式：输出离散组号 1~n_bins，单调递增，bin 数等于 n_bins。"""
    df = pd.DataFrame({
        "date": ["2024-01-01"] * 10,
        "order_book_id": list("ABCDEFGHIJ"),
        "x": np.arange(10, dtype=float),
    })
    ctx = _ctx_with("data", df)
    _run("rank", ctx, {
        "action": "rank",
        "source_column": "x",
        "output_column": "bin",
        "group_by": ["date"],
        "ascending": True,
        "n_bins": 5,
    })
    bins = ctx.get_df("data")["bin"].astype(int).tolist()
    # 公式 min(floor(pct*5)+1, 5) 对 pct=0.1..1.0 给出 [1,2,2,3,3,4,4,5,5,5]
    # 不严格按等量分桶（边界归属上一桶），但 bin 取值在 [1,5]，且单调不减
    assert min(bins) == 1 and max(bins) == 5
    assert all(bins[i] <= bins[i + 1] for i in range(len(bins) - 1))


def test_rank_requires_group_by_list():
    """缺 group_by 必须 raise（防止隐式截面排名）。"""
    df = pd.DataFrame({"order_book_id": ["A"], "x": [1.0]})
    ctx = _ctx_with("data", df)
    try:
        _run("rank", ctx, {
            "action": "rank",
            "source_column": "x",
            "output_column": "x_rk",
        })
    except ValueError as e:
        assert "group_by" in str(e)
        return
    raise AssertionError("未 raise: rank 缺 group_by")


# ──────────────────────────────────────────────────────────
# transform（long 格式，order_book_id 分组）
# ──────────────────────────────────────────────────────────


def test_transform_ffill_long():
    """同股票内前向填充。"""
    df = pd.DataFrame({
        "order_book_id": ["A", "A", "A", "B", "B", "B"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
        "v": [1.0, np.nan, np.nan, np.nan, 2.0, np.nan],
    })
    ctx = _ctx_with("data", df)
    _run("transform", ctx, {
        "action": "transform",
        "method": "ffill",
        "source_column": "v",
        "output_column": "v_ff",
    })
    out = ctx.get_df("data")
    a_vals = out[out.order_book_id == "A"]["v_ff"].tolist()
    b_vals = out[out.order_book_id == "B"]["v_ff"].tolist()
    assert a_vals == [1.0, 1.0, 1.0]
    assert pd.isna(b_vals[0]) and b_vals[1:] == [2.0, 2.0]


def test_transform_diff_quarterly():
    """累计值 → 单季度：q1 保留原值，q2/q3/q4 做组内 diff。"""
    df = pd.DataFrame({
        "order_book_id": ["A"] * 4,
        "quarter": ["2024q1", "2024q2", "2024q3", "2024q4"],
        "rev_cum": [10.0, 25.0, 45.0, 70.0],
    })
    ctx = _ctx_with("data", df)
    _run("transform", ctx, {
        "action": "transform",
        "method": "diff_quarterly",
        "source_column": "rev_cum",
        "output_column": "rev_q",
    })
    # q1=10（保留），q2=25-10=15, q3=45-25=20, q4=70-45=25
    assert ctx.get_df("data")["rev_q"].tolist() == [10.0, 15.0, 20.0, 25.0]


def test_transform_yoy_ratio():
    """yoy 是真比率：(x - x.shift(4)) / x.shift(4)。"""
    df = pd.DataFrame({
        "order_book_id": ["A"] * 6,
        "x": [100.0, 110.0, 120.0, 130.0, 200.0, 220.0],
    })
    ctx = _ctx_with("data", df)
    _run("transform", ctx, {
        "action": "transform",
        "method": "yoy",
        "source_column": "x",
        "output_column": "yoy",
        "periods": 4,
    })
    out = ctx.get_df("data")["yoy"].tolist()
    # idx 0..3: NaN（无前值）；idx 4: (200-100)/100=1.0；idx 5: (220-110)/110=1.0
    assert all(pd.isna(v) for v in out[:4])
    assert out[4] == 1.0 and out[5] == 1.0


# ──────────────────────────────────────────────────────────
# rolling（long 格式）
# ──────────────────────────────────────────────────────────


def test_rolling_plain_long():
    """普通 rolling MA3，按 order_book_id 分组。"""
    df = pd.DataFrame({
        "order_book_id": ["A"] * 5 + ["B"] * 5,
        "date": pd.to_datetime(["2024-01-0" + str(i) for i in range(1, 6)] * 2),
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
    })
    ctx = _ctx_with("data", df)
    _run("rolling", ctx, {
        "action": "rolling",
        "source_column": "x",
        "output_column": "ma3",
        "window": 3,
        "min_periods": 3,
        "agg": "mean",
        "group_by": "order_book_id",
    })
    out = ctx.get_df("data")
    a = out[out.order_book_id == "A"]["ma3"].tolist()
    b = out[out.order_book_id == "B"]["ma3"].tolist()
    assert pd.isna(a[0]) and pd.isna(a[1])
    assert a[2:] == [2.0, 3.0, 4.0]
    assert pd.isna(b[0]) and pd.isna(b[1])
    assert b[2:] == [20.0, 30.0, 40.0]


def test_rolling_change_on_with_ffill():
    """变化日 rolling：只在 change_on 列值变化的行采样，ffill 回所有行。"""
    df = pd.DataFrame({
        "order_book_id": ["A"] * 10,
        "date": pd.to_datetime([f"2024-01-{i:02d}" for i in range(1, 11)]),
        "roic": [0.10] * 3 + [0.08] * 3 + [0.06] * 4,   # 3 个变化日
        "ind_rk": [5.0] * 3 + [8.0] * 3 + [12.0] * 4,
    })
    ctx = _ctx_with("data", df)
    _run("rolling", ctx, {
        "action": "rolling",
        "source_column": "ind_rk",
        "output_column": "ind_rk_min2",
        "window": 2,
        "min_periods": 1,
        "agg": "min",
        "change_on": "roic",
        "fill_method": "ffill",
        "group_by": "order_book_id",
    })
    # 变化日值: [5, 8, 12]；rolling(2,min_periods=1).min(): [5, 5, 8]
    # ffill 回所有行: idx0-2=5, idx3-5=5, idx6-9=8
    expected = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 8.0, 8.0, 8.0, 8.0]
    assert ctx.get_df("data")["ind_rk_min2"].tolist() == expected


def test_rolling_min_periods_insufficient():
    """min_periods 不足时变化日 rolling 全 NaN。"""
    df = pd.DataFrame({
        "order_book_id": ["A"] * 8,
        "date": pd.to_datetime([f"2024-01-{i:02d}" for i in range(1, 9)]),
        "roic": [0.10] * 4 + [0.08] * 4,   # 仅 2 个变化日
        "ind_rk": [5.0] * 4 + [8.0] * 4,
    })
    ctx = _ctx_with("data", df)
    _run("rolling", ctx, {
        "action": "rolling",
        "source_column": "ind_rk",
        "output_column": "ind_rk_min5",
        "window": 5,
        "min_periods": 5,
        "agg": "min",
        "change_on": "roic",
        "fill_method": "ffill",
        "group_by": "order_book_id",
    })
    assert ctx.get_df("data")["ind_rk_min5"].isna().all()


# ──────────────────────────────────────────────────────────
# fetch — 错误路径（custom API 只规范化了一个 command；
# 真实 fetch 行为靠端到端因子回归覆盖，而非 mock 假命令）
# ──────────────────────────────────────────────────────────


def test_fetch_requires_universe():
    """fetch 调用前 ctx.universe 不能为空。"""
    ctx = Context(factor_name="test")
    try:
        OpRegistry.get("fetch")(
            ctx,
            {
                "action": "fetch",
                "api": "get_factor",
                "fields": ["pe_ratio_ttm"],
                "output_column": "pe",
            },
            MockFetcher(),
        )
    except ValueError as e:
        assert "universe" in str(e)
        return
    raise AssertionError("未 raise: fetch 缺 universe")


def test_fetch_rejects_dst_collision():
    """目标 DataFrame 已存在时 fetch 必须 raise（防止覆盖；扩列要走 merge）。"""
    ctx = Context(factor_name="test")
    ctx.universe = ["000001.XSHE"]
    ctx.dataframes["data"] = pd.DataFrame({"order_book_id": ["000001.XSHE"]})
    try:
        OpRegistry.get("fetch")(
            ctx,
            {
                "action": "fetch",
                "api": "get_factor",
                "fields": ["pe_ratio_ttm"],
                "output_column": "pe",
            },
            MockFetcher(),
        )
    except ValueError as e:
        assert "已存在" in str(e) or "merge" in str(e)
        return
    raise AssertionError("未 raise: fetch 目标已存在")
