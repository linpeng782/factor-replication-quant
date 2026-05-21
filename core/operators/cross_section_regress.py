"""
cross_section_regress 算子
============================================================
按 date 分组做截面 OLS：每个交易日上跨股票回归 y = X·β + ε，输出残差列。

经典用途：
  - 风格剥离（用 size、industry 当 X，剥离风格暴露后的残差为"纯因子"）
  - 嵌套残差化（先把 A 对 B 回归取残差 r_AB，再把 C 对 r_AB 回归取残差）
  - 因子正交化

契约：
  - source_column_y   : str         必填，被回归列
  - source_columns_x  : list[str]   必填，1 个或多个解释变量
  - output_column     : str         必填，输出残差列
  - add_intercept     : bool        默认 True（含截距）
  - date_column       : str         默认 "date"
  - min_samples       : int         默认 30，截面有效样本不足时整组置 NaN

实现：按 date groupby + numpy.linalg.lstsq。Y 或任意 X 含 NaN 的行不参与回归
（输出该行 NaN）；样本数不足 max(p+5, min_samples) 时整组 NaN。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from . import Context, OpRegistry


def _regress_residuals(
    y: np.ndarray,
    X: np.ndarray,
    add_intercept: bool,
    min_samples: int,
) -> np.ndarray:
    """单组截面 OLS，返回残差数组（与输入等长，无效位置 NaN）"""
    n = len(y)
    valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    n_valid = int(valid.sum())
    p = X.shape[1] + (1 if add_intercept else 0)
    residuals = np.full(n, np.nan, dtype=np.float64)

    if n_valid < max(p + 5, min_samples):
        return residuals

    X_v = X[valid]
    if add_intercept:
        X_v = np.column_stack([X_v, np.ones(n_valid)])
    y_v = y[valid]

    try:
        beta, *_ = np.linalg.lstsq(X_v, y_v, rcond=None)
    except np.linalg.LinAlgError:
        return residuals

    residuals_valid = y_v - X_v @ beta
    residuals[np.where(valid)[0]] = residuals_valid
    return residuals


@OpRegistry.register("cross_section_regress")
def op_cross_section_regress(ctx: Context, step: Dict, fetcher: Any) -> None:
    target_df = step.get("output_dataframe", "data")
    df = ctx.get_df(target_df)

    y_col = step["source_column_y"]
    x_cols = list(step["source_columns_x"])
    out_col = step["output_column"]
    add_intercept = bool(step.get("add_intercept", True))
    date_col = step.get("date_column", "date")
    min_samples = int(step.get("min_samples", 30))

    if not x_cols:
        raise ValueError("cross_section_regress: source_columns_x 不能为空")
    if date_col not in df.columns:
        raise ValueError(
            f"cross_section_regress: date_column={date_col!r} 不在 DataFrame 中"
        )

    Y = df[y_col].to_numpy(dtype=np.float64)
    X = df[x_cols].to_numpy(dtype=np.float64)

    result = np.full(len(df), np.nan, dtype=np.float64)
    # groupby(...).indices 直接给位置索引数组，比迭代分组对象快
    groups = df.groupby(date_col, sort=False).indices  # dict[date, np.array]

    for _date, pos in groups.items():
        residuals = _regress_residuals(
            Y[pos], X[pos], add_intercept=add_intercept, min_samples=min_samples
        )
        result[pos] = residuals

    ctx.add_column(target_df, out_col, pd.Series(result, index=df.index))
