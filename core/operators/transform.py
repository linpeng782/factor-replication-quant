"""
transform 算子
============================================================
所有 method 统一契约：
  - 输入：source_column（必填，单数；要处理多列就写多个 step）
  - 输出：output_column（必填）
  - 永远新增列，**永不覆盖**输入列
  - long 格式 + group_by（默认 'order_book_id'）

支持的 method：
  diff             同股票内 t - t-N
  shift            同股票内 t-N（裸 shift；不计算比率）
  yoy / qoq        真正的同比/环比比率 = (x - x.shift(N)) / x.shift(N)
  zscore           按 group_by 分组的 zscore（无 group_by 时全表）
  ffill / bfill    前/后向填充
  diff_quarterly   PIT 财务专用：累计值转单季度（按 quarter 列）
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from . import Context, OpRegistry


@OpRegistry.register("transform")
def op_transform(ctx: Context, step: Dict, fetcher: Any) -> None:
    target_df = step.get("output_dataframe", "data")
    df = ctx.get_df(target_df)

    method = step.get("method")
    if not method:
        raise ValueError("transform: method 必填")

    src = step["source_column"]
    out = step["output_column"]

    handler = _METHOD_DISPATCH.get(method)
    if handler is None:
        raise ValueError(
            f"transform: 不支持的 method={method!r}；可选: {sorted(_METHOD_DISPATCH)}"
        )

    series = handler(df, src, step)
    ctx.add_column(target_df, out, series)


# ── method handlers ───────────────────────────────────────


def _method_diff(df: pd.DataFrame, src: str, step: Dict) -> pd.Series:
    periods = int(step.get("periods", 1))
    group_by = step.get("group_by", "order_book_id")
    if group_by:
        return df.groupby(group_by)[src].diff(periods=periods)
    return df[src].diff(periods=periods)


def _method_shift(df: pd.DataFrame, src: str, step: Dict) -> pd.Series:
    periods = int(step.get("periods", 1))
    group_by = step.get("group_by", "order_book_id")
    if group_by:
        return df.groupby(group_by)[src].shift(periods)
    return df[src].shift(periods)


def _method_yoy(df: pd.DataFrame, src: str, step: Dict) -> pd.Series:
    """真正的同比比率：(x - x.shift(N)) / x.shift(N)"""
    periods = int(step.get("periods", 4))  # 季度数据默认 4 期
    group_by = step.get("group_by", "order_book_id")
    if group_by:
        prev = df.groupby(group_by)[src].shift(periods)
    else:
        prev = df[src].shift(periods)
    denom = prev.replace(0, np.nan)
    return (df[src] - prev) / denom


def _method_qoq(df: pd.DataFrame, src: str, step: Dict) -> pd.Series:
    step_with_default = {**step, "periods": step.get("periods", 1)}
    return _method_yoy(df, src, step_with_default)


def _method_zscore(df: pd.DataFrame, src: str, step: Dict) -> pd.Series:
    group_by = step.get("group_by")
    if group_by:
        return df.groupby(group_by)[src].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    return (df[src] - df[src].mean()) / df[src].std()


def _method_ffill(df: pd.DataFrame, src: str, step: Dict) -> pd.Series:
    group_by = step.get("group_by", "order_book_id")
    if group_by:
        return df.groupby(group_by)[src].ffill()
    return df[src].ffill()


def _method_bfill(df: pd.DataFrame, src: str, step: Dict) -> pd.Series:
    group_by = step.get("group_by", "order_book_id")
    if group_by:
        return df.groupby(group_by)[src].bfill()
    return df[src].bfill()


def _method_winsorize(df: pd.DataFrame, src: str, step: Dict) -> pd.Series:
    """截面 MAD 去极值：按 group_by 分组，组内做 MAD clip。"""
    n = float(step.get("n", 3.0))
    group_by = step.get("group_by", "date")
    mad_scale = 1.4826

    def _mad_clip(s: pd.Series) -> pd.Series:
        med = s.median()
        mad = (s - med).abs().median() * mad_scale
        if mad == 0:
            return s
        return s.clip(lower=med - n * mad, upper=med + n * mad)

    return df.groupby(group_by)[src].transform(_mad_clip)


def _method_diff_quarterly(df: pd.DataFrame, src: str, step: Dict) -> pd.Series:
    """PIT 财务累计值 → 单季度。要求 df 有 quarter 列。Q1 保留原值，其他季度做差分。"""
    if "quarter" not in df.columns:
        raise ValueError("diff_quarterly: 输入 DataFrame 必须包含 quarter 列")
    group_by = step.get("group_by", "order_book_id")
    diffed = df.groupby(group_by)[src].diff()
    q1_mask = df["quarter"].astype(str).str.endswith("q1")
    return diffed.where(~q1_mask, df[src])


_METHOD_DISPATCH = {
    "diff": _method_diff,
    "shift": _method_shift,
    "yoy": _method_yoy,
    "qoq": _method_qoq,
    "zscore": _method_zscore,
    "ffill": _method_ffill,
    "bfill": _method_bfill,
    "diff_quarterly": _method_diff_quarterly,
    "winsorize": _method_winsorize,
}
