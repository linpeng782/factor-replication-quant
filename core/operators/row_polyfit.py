"""
row_polyfit 算子
============================================================
行向多列多项式 OLS 回归，取指定阶次系数。numpy 向量化，1300 万行 ~ 秒级。

底层用闭式解 [a, b, c, ...] = pinv(V) @ y，其中 V 是 Vandermonde 矩阵。
对**固定 x**（x_pattern=equispaced 时所有行共享同一 x 向量），pinv(V) 只算一次，
结果就是 (n_rows, n_cols) 矩阵乘以 (n_cols,) 权重向量，单次 numpy 运算。

契约：
  - source_columns_y : list[str]    必填，n 个 y 列
  - source_columns_x : list[str]    可选；如果给，每行 x 不同（变量 x，未实现，先抛错）
  - x_pattern        : str          "equispaced"（默认，0..n-1 等距 + zscore）或
                                    "equispaced_no_zscore"（原值不 zscore）
  - zscore_y         : bool         默认 True；行向对 y 列做 zscore（axis=1, ddof=1）。
                                    跨股票截面因子需要保持 True，让不同体量的公司的 a
                                    形状可比；极少数 per-stock 时序场景才考虑 False。
                                    某行 y 全相等（std=0）时该行结果置 NaN。
  - degree           : int          多项式阶次（默认 2）
  - coefficient      : str          取哪一项: "a{degree}" / ... / "a1" / "a0"
                                    （沿用 numpy.polyfit 约定：高次在前）
  - output_column    : str          必填
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from . import Context, OpRegistry


@OpRegistry.register("row_polyfit")
def op_row_polyfit(ctx: Context, step: Dict, fetcher: Any) -> None:
    target_df = step.get("output_dataframe", "data")
    df = ctx.get_df(target_df)

    y_cols = list(step["source_columns_y"])
    n = len(y_cols)
    if n < 2:
        raise ValueError("row_polyfit: source_columns_y 至少需要 2 个列")

    degree = int(step.get("degree", 2))
    if degree < 1:
        raise ValueError(f"row_polyfit: degree={degree} 必须 >= 1")
    if degree >= n:
        raise ValueError(
            f"row_polyfit: degree={degree} 必须 < 列数 ({n})；"
            f"否则系数过拟合无意义"
        )

    coefficient = step["coefficient"]
    valid_coefs = [f"a{degree - i}" for i in range(degree + 1)]
    if coefficient not in valid_coefs:
        raise ValueError(
            f"row_polyfit: coefficient={coefficient!r} 不合法；"
            f"degree={degree} 时可选 {valid_coefs}（高次在前）"
        )
    coef_idx = valid_coefs.index(coefficient)

    # ── 构造 x 向量 ──
    if "source_columns_x" in step:
        raise NotImplementedError(
            "row_polyfit: 变量 x（source_columns_x）暂未实现，"
            "请用 x_pattern=equispaced 或 equispaced_no_zscore"
        )

    x_pattern = step.get("x_pattern", "equispaced")
    x_raw = np.arange(n, dtype=np.float64)
    if x_pattern == "equispaced":
        x = (x_raw - x_raw.mean()) / x_raw.std(ddof=1)
    elif x_pattern == "equispaced_no_zscore":
        x = x_raw
    else:
        raise ValueError(
            f"row_polyfit: 不支持的 x_pattern={x_pattern!r}；"
            f"可选: equispaced / equispaced_no_zscore"
        )

    # Vandermonde 矩阵 (n, degree+1)，从高次到低次
    V = np.column_stack([x ** (degree - i) for i in range(degree + 1)])
    W = np.linalg.pinv(V)  # (degree+1, n)
    weights = W[coef_idx]  # (n,)

    # ── 行向回归 ──
    Y = df[y_cols].to_numpy(dtype=np.float64)  # (N_rows, n)
    # 任一 y_i NaN → 整行结果置 NaN（在 zscore 之前算 mask，不受 zscore 中除零影响）
    nan_mask = np.isnan(Y).any(axis=1)

    # 行向 zscore Y（默认 True；让跨股票回归在归一化曲线上做，a 系数才可比）
    zscore_y = step.get("zscore_y", True)
    if zscore_y:
        Y_mean = np.nanmean(Y, axis=1, keepdims=True)
        Y_std = np.nanstd(Y, axis=1, ddof=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            Y = np.where(Y_std > 0, (Y - Y_mean) / Y_std, np.nan)
        # std=0（行内 y 全相等）的行也置 NaN
        nan_mask = nan_mask | (Y_std.ravel() == 0)

    result = Y @ weights  # (N_rows,)
    result[nan_mask] = np.nan

    ctx.add_column(target_df, step["output_column"], pd.Series(result, index=df.index))
