"""数据转换操作"""

import numpy as np
import pandas as pd
from typing import Any, Dict

from . import OpRegistry, _resolve_input


@OpRegistry.register("transform")
def op_transform(ctx: Dict, step: Dict, fetcher: Any) -> pd.DataFrame:
    """
    action: transform
    数据转换：差分、同比、环比、标准化等
    """
    method = step.get("method", "")
    input_var = step.get("input", step.get("output"))
    output_name = step.get("output", "transformed")

    df = _resolve_input(ctx, input_var)

    if method == "diff_quarterly":
        df = _diff_quarterly(df, step, ctx)
    elif method == "yoy":
        lag = step.get("lag_periods", 4)
        df = _compute_yoy(df, lag, step, ctx)
    elif method == "qoq":
        lag = step.get("lag_periods", 1)
        df = _compute_qoq(df, lag, step, ctx)
    elif method == "zscore":
        df = _compute_zscore(df, step, ctx)
    else:
        raise ValueError(f"不支持的 transform method: {method}")

    ctx[output_name] = df
    return df


def _diff_quarterly(df: pd.DataFrame, step: Dict, ctx: Dict) -> pd.DataFrame:
    """累计值转单季度"""
    id_col = "order_book_id"
    quarter_col = "quarter"

    if id_col in df.index.names and quarter_col in df.index.names:
        df = df.sort_index().copy()
        id_vals = df.index.get_level_values(id_col)
        quarter_vals = df.index.get_level_values(quarter_col)
    elif id_col in df.columns and quarter_col in df.columns:
        df = df.sort_values([id_col, quarter_col]).copy()
        id_vals = df[id_col]
        quarter_vals = df[quarter_col]
    else:
        raise KeyError(f"数据中找不到 {id_col} 和 {quarter_col}")

    exclude = {"info_date", "announcement_date", "if_adjusted", "rice_create_tm", id_col, quarter_col}
    value_cols = [c for c in df.columns if c not in exclude and df[c].dtype.kind in "fi"]

    for col in value_cols:
        new_col = f"{col}_mrq"
        df[new_col] = df.groupby(id_vals)[col].diff().values
        q1_mask = quarter_vals.astype(str).str.endswith("q1")
        df.loc[q1_mask, new_col] = df.loc[q1_mask, col]

    return df


def _compute_yoy(df: pd.DataFrame, lag: int, step: Dict, ctx: Dict) -> pd.DataFrame:
    """计算同比（t vs t-lag）"""
    id_col = "order_book_id"
    value_cols = [c for c in df.columns if c not in (id_col, "quarter", "info_date", "announcement_date")]

    df = df.sort_values([id_col, "quarter"]).copy()
    for col in value_cols:
        lag_col = f"{col}_lag{lag}"
        df[lag_col] = df.groupby(id_col)[col].shift(lag)

    return df


def _compute_qoq(df: pd.DataFrame, lag: int, step: Dict, ctx: Dict) -> pd.DataFrame:
    """计算环比（t vs t-lag）"""
    return _compute_yoy(df, lag, step, ctx)


def _compute_zscore(df: pd.DataFrame, step: Dict, ctx: Dict) -> pd.DataFrame:
    """计算 zscore 标准化"""
    group_col = step.get("group_column", "")
    value_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]

    if group_col and group_col in df.columns:
        for col in value_cols:
            df[f"{col}_zscore"] = df.groupby(group_col)[col].transform(lambda x: (x - x.mean()) / x.std())
    else:
        for col in value_cols:
            df[f"{col}_zscore"] = (df[col] - df[col].mean()) / df[col].std()
    return df
