"""
row_correlate 算子
============================================================
行向 Pearson 相关系数：每行有两组等长向量 a 和 b，输出 corr(a, b)。

向量化闭式：对 (N_rows, n) 的 A 和 B，
  cov_ab = mean((A - mean_a) * (B - mean_b))
  var_a  = mean((A - mean_a)²)
  var_b  = mean((B - mean_b)²)
  corr   = cov_ab / sqrt(var_a · var_b)

契约：
  - source_columns_a : list[str]    必填，第一组列（n 个）
  - source_columns_b : list[str]    必填，第二组列（必须同样长度 n）
  - method           : str          可选，默认 'pearson'（目前仅支持 pearson）
  - output_column    : str          必填
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from . import Context, OpRegistry


@OpRegistry.register("row_correlate")
def op_row_correlate(ctx: Context, step: Dict, fetcher: Any) -> None:
    target_df = step.get("output_dataframe", "data")
    df = ctx.get_df(target_df)

    a_cols = list(step["source_columns_a"])
    b_cols = list(step["source_columns_b"])
    if len(a_cols) != len(b_cols):
        raise ValueError(
            f"row_correlate: source_columns_a 和 source_columns_b 长度必须相等，"
            f"当前 {len(a_cols)} vs {len(b_cols)}"
        )
    if len(a_cols) < 2:
        raise ValueError("row_correlate: 每组至少需要 2 个列")

    method = step.get("method", "pearson")
    if method != "pearson":
        raise ValueError(f"row_correlate: method={method!r} 暂不支持；目前仅支持 pearson")

    A = df[a_cols].to_numpy(dtype=np.float64)  # (N_rows, n)
    B = df[b_cols].to_numpy(dtype=np.float64)

    # 任一侧含 NaN → 整行结果置 NaN
    nan_mask = np.isnan(A).any(axis=1) | np.isnan(B).any(axis=1)

    mean_a = A.mean(axis=1, keepdims=True)
    mean_b = B.mean(axis=1, keepdims=True)
    A_dev = A - mean_a
    B_dev = B - mean_b

    cov_ab = (A_dev * B_dev).mean(axis=1)
    var_a = (A_dev ** 2).mean(axis=1)
    var_b = (B_dev ** 2).mean(axis=1)

    denom = np.sqrt(var_a * var_b)
    # 避免除零；常数序列 var=0 时相关系数无意义，置 NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom > 0, cov_ab / denom, np.nan)

    result[nan_mask] = np.nan

    ctx.add_column(target_df, step["output_column"], pd.Series(result, index=df.index))
