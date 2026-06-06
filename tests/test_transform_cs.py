"""transform 新增 method cs_demean / abs 的单测（适度冒险族 "适度=|x−截面均值|" 依赖）。"""
import numpy as np
import pandas as pd

from core.operators import Context
from core.operators.transform import op_transform


def _ctx(df):
    c = Context(factor_name="t")
    c.set_df("data", df)
    return c


def test_cs_demean_by_date():
    # 两日，每日截面去均值后组内均值应为 0
    df = pd.DataFrame({
        "order_book_id": ["A", "B", "C", "A", "B", "C"],
        "date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3),
        "x": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
    })
    ctx = _ctx(df)
    op_transform(ctx, {"method": "cs_demean", "source_column": "x",
                       "output_column": "x_dm", "group_by": "date"}, None)
    out = ctx.get_df("data")
    # day1 mean=2 → [-1,0,1]; day2 mean=20 → [-10,0,10]
    assert np.allclose(out["x_dm"].to_numpy(), [-1, 0, 1, -10, 0, 10])
    # 每个日期组内均值≈0
    assert np.allclose(out.groupby("date")["x_dm"].mean().to_numpy(), [0, 0])


def test_cs_demean_default_group_is_date():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "x": [4.0, 6.0],
    })
    ctx = _ctx(df)
    # 不传 group_by → 默认按 date
    op_transform(ctx, {"method": "cs_demean", "source_column": "x", "output_column": "d"}, None)
    assert np.allclose(ctx.get_df("data")["d"].to_numpy(), [-1.0, 1.0])


def test_abs():
    df = pd.DataFrame({"x": [-1.5, 0.0, 2.0, np.nan]})
    ctx = _ctx(df)
    op_transform(ctx, {"method": "abs", "source_column": "x", "output_column": "ax"}, None)
    out = ctx.get_df("data")["ax"].to_numpy()
    assert out[0] == 1.5 and out[1] == 0.0 and out[2] == 2.0 and np.isnan(out[3])


def test_moderate_equiv_demean_then_abs():
    # "适度" = |x − 截面均值| = abs(cs_demean(x))
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"] * 3),
        "x": [1.0, 2.0, 6.0],  # mean=3
    })
    ctx = _ctx(df)
    op_transform(ctx, {"method": "cs_demean", "source_column": "x", "output_column": "dm", "group_by": "date"}, None)
    op_transform(ctx, {"method": "abs", "source_column": "dm", "output_column": "moderate"}, None)
    assert np.allclose(ctx.get_df("data")["moderate"].to_numpy(), [2.0, 1.0, 3.0])
