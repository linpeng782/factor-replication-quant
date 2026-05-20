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
    elif method == "ffill":
        df = _compute_ffill(df, step, ctx)
    elif method == "bfill":
        df = _compute_bfill(df, step, ctx)
    elif method == "diff":
        df = _compute_diff(df, step, ctx)
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


def _compute_ffill(df: pd.DataFrame, step: Dict, ctx: Dict) -> pd.DataFrame:
    """前向填充

    支持两种模式:
    1. wide 格式 (date × order_book_id): 直接 df.ffill(axis=0)
    2. long 格式: 按 order_book_id groupby 后 ffill
    """
    group_col = step.get("group_column", "")
    axis = step.get("axis", 0)
    value_cols = step.get("columns")

    if value_cols is None:
        # 自动识别数值列，排除常见的非数值列
        exclude = {"order_book_id", "date", "quarter", "info_date", "announcement_date", "if_adjusted", "rice_create_tm", group_col}
        value_cols = [c for c in df.columns if c not in exclude and df[c].dtype.kind in "fi"]

    # long 格式且有 group_col
    if group_col and group_col in df.columns:
        for col in value_cols:
            if col in df.columns:
                df[col] = df.groupby(group_col)[col].ffill()
    else:
        # wide 格式 或 无 group_col
        for col in value_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

    return df


def _compute_bfill(df: pd.DataFrame, step: Dict, ctx: Dict) -> pd.DataFrame:
    """后向填充，逻辑同 ffill"""
    group_col = step.get("group_column", "")
    value_cols = step.get("columns")

    if value_cols is None:
        exclude = {"order_book_id", "date", "quarter", "info_date", "announcement_date", "if_adjusted", "rice_create_tm", group_col}
        value_cols = [c for c in df.columns if c not in exclude and df[c].dtype.kind in "fi"]

    if group_col and group_col in df.columns:
        for col in value_cols:
            if col in df.columns:
                df[col] = df.groupby(group_col)[col].bfill()
    else:
        for col in value_cols:
            if col in df.columns:
                df[col] = df[col].bfill()

    return df


def _compute_diff(df: pd.DataFrame, step: Dict, ctx: Dict) -> pd.DataFrame:
    """计算差分（日频/时频）

    参数:
        periods: int, 差分周期，默认 1（即 t - t-1）
        group_column: str, long 格式下按该列 groupby 后 diff
        columns: list[str], 需要差分的列，默认所有数值列
    """
    group_col = step.get("group_column", "")
    periods = step.get("periods", 1)
    value_cols = step.get("columns")

    print(f"   [_compute_diff] ENTER: group_col={group_col}, periods={periods}, cols={value_cols}, df.shape={df.shape}, df.columns={list(df.columns)}")

    if value_cols is None:
        exclude = {"order_book_id", "date", "quarter", "info_date", "announcement_date", "if_adjusted", "rice_create_tm", group_col}
        value_cols = [c for c in df.columns if c not in exclude and df[c].dtype.kind in "fi"]

    if group_col and group_col in df.columns:
        print(f"   [_compute_diff] path=groupby, group_col={group_col} found in df")
        for col in value_cols:
            if col in df.columns:
                before = df[col].iloc[:5].tolist()
                df[col] = df.groupby(group_col)[col].diff(periods=periods)
                after = df[col].iloc[:5].tolist()
                nonnull = df[col].notna().sum()
                print(f"   [_compute_diff] {col}: before={before}, after={after}, nonnull={nonnull}/{len(df)}")
    else:
        print(f"   [_compute_diff] path=no_groupby, group_col={group_col} NOT found in df")
        for col in value_cols:
            if col in df.columns:
                df[col] = df[col].diff(periods=periods)

    print(f"   [_compute_diff] EXIT")
    return df
