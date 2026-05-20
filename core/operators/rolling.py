"""滚动窗口聚合操作

支持两种模式:
1. 普通 rolling: 在所有行上做滚动窗口聚合
2. 变化日 rolling: 只在指定源数据值变化的日子上做聚合，然后填充

使用示例:
    # 普通 rolling (wide 格式)
    action: rolling
    input: factor_values
    window: 20
    min_periods: 10
    agg: mean

    # 变化日 rolling (long 格式，按 group_by 检测变化)
    action: rolling
    input: ind_rank
    window: 8
    min_periods: 4
    agg: min
    group_by: order_book_id    # long 格式下按股票分组
    on: return_on_invested_capital_ttm  # 引用同一 DataFrame 中的列做变化检测
    fill_method: ffill

    # 变化日 rolling (wide 格式，逐列检测)
    action: rolling
    input: ind_rank
    window: 8
    min_periods: 4
    agg: min
    on: roic_ttm_wide         # 引用另一个 wide DataFrame 做变化检测
    fill_method: ffill
"""

import numpy as np
import pandas as pd
from typing import Any, Dict

from . import OpRegistry, _resolve_input


@OpRegistry.register("rolling")
def op_rolling(ctx: Dict, step: Dict, fetcher: Any) -> pd.DataFrame:
    """
    action: rolling
    滚动窗口聚合

    参数:
        input        : 输入变量名（DataFrame）
        window       : 窗口大小（行数）
        min_periods  : 最少有效值数量（默认=window）
        agg          : 聚合函数，可选: min, max, mean, std, sum, median, count
        group_by     : 【long格式】分组列名，如 order_book_id
        on           : 【可选】变化检测源列名（同一DataFrame中的列）或变量名（另一个DataFrame）
        fill_method  : 填充方法，可选: ffill, bfill, none（默认 ffill）
        columns      : 【可选】只对这些列做rolling，None=所有数值列
        output       : 输出变量名
    """
    input_var = step.get("input")
    output_name = step.get("output", "rolled")
    window = step.get("window", 8)
    min_periods = step.get("min_periods", window)
    agg = step.get("agg", "min")
    group_by = step.get("group_by", None)
    on_var = step.get("on", None)
    fill_method = step.get("fill_method", "ffill")
    columns = step.get("columns")

    df = _resolve_input(ctx, input_var)

    # 确定要处理的列
    if columns is not None:
        value_cols = [c for c in columns if c in df.columns]
    else:
        value_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]

    if not value_cols:
        raise ValueError(f"rolling: 找不到可处理的数值列，columns={list(df.columns)}")

    # 验证聚合函数
    valid_aggs = {"min", "max", "mean", "std", "sum", "median", "count"}
    if agg not in valid_aggs:
        raise ValueError(f"rolling: 不支持的 agg={agg}，可选: {valid_aggs}")

    result = df.copy()

    # 判断 on 是同一 DataFrame 的列名，还是另一个 DataFrame 的变量名
    on_col = None
    on_df = None
    if on_var:
        if on_var in df.columns:
            on_col = on_var  # 同一 DataFrame 的列
        elif on_var in ctx and isinstance(ctx[on_var], pd.DataFrame):
            on_df = ctx[on_var]  # 另一个 DataFrame
        else:
            raise ValueError(
                f"rolling: on={on_var} 既不是输入DataFrame的列，也不是上下文中的DataFrame变量"
            )

    if on_col is not None or on_df is not None:
        # ──────────────────────────────────────────
        # 模式A: 变化日 rolling
        # ──────────────────────────────────────────
        if group_by and group_by in df.columns:
            # ── 模式A-1: long 格式，按 group_by 分组检测变化 ──
            sort_cols = [group_by]
            if "date" in df.columns:
                sort_cols.append("date")
            result = result.sort_values(sort_cols).copy()

            for col in value_cols:
                if col not in result.columns:
                    continue

                # 变化检测源
                if on_col:
                    source = result[on_col]
                else:
                    if col not in on_df.columns:
                        continue
                    # 对齐索引（假设 on_df 和 df 有相同的行顺序或可通过索引对齐）
                    source = on_df[col].reindex(result.index)

                # 标记变化日
                change_mask = result.groupby(group_by)[source.name].transform(
                    lambda x: x != x.shift(1)
                )

                # 提取变化日的 col 值（dropna，避免NaN污染rolling窗口）
                change_df = (
                    result.loc[change_mask, [group_by, col]].copy().dropna(subset=[col])
                )

                if len(change_df) == 0:
                    continue

                # 按 group 做 rolling
                rolled = change_df.groupby(group_by)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=min_periods).agg(agg)
                )

                # 先清空该列（用 np.nan 保持 float64 dtype）
                result[col] = np.nan

                # 写回结果（用 rolled 的索引，不是 change_mask，因为 dropna 后索引可能更少）
                result.loc[rolled.index, col] = rolled.values

                # 填充
                if fill_method == "ffill":
                    result[col] = result.groupby(group_by)[col].ffill()
                elif fill_method == "bfill":
                    result[col] = result.groupby(group_by)[col].bfill()

        elif on_df is not None:
            # ── 模式A-2: wide 格式，逐列检测变化 ──
            for col in value_cols:
                if col not in on_df.columns:
                    continue

                # 逐列识别变化日
                on_series = on_df[col]
                changes = on_series[on_series != on_series.shift(1)]
                change_dates = changes.dropna().index

                if len(change_dates) == 0:
                    continue

                # 提取输入数据在这些变化日的值
                report_vals = df.loc[df.index.isin(change_dates), col].dropna()

                if len(report_vals) == 0:
                    continue

                # 在变化日序列上做 rolling
                rolled = report_vals.rolling(
                    window=window, min_periods=min_periods
                ).agg(agg)

                # 先清空该列（用 np.nan 保持 float64 dtype）
                result[col] = np.nan

                # 写回结果（仅变化日有值）
                result.loc[rolled.index, col] = rolled.values

                # 填充到所有行
                if fill_method == "ffill":
                    result[col] = result[col].ffill()
                elif fill_method == "bfill":
                    result[col] = result[col].bfill()
                # "none" 则不填充
        else:
            raise ValueError(
                "rolling: 指定了 on 参数，但既没有 group_by（long格式）也没有提供wide格式的on DataFrame"
            )

    else:
        # ──────────────────────────────────────────
        # 模式B: 普通 rolling
        # ──────────────────────────────────────────
        if group_by and group_by in df.columns:
            # long 格式
            for col in value_cols:
                if col not in result.columns:
                    continue
                result[col] = result.groupby(group_by)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=min_periods).agg(agg)
                )

                if fill_method == "ffill":
                    result[col] = result.groupby(group_by)[col].ffill()
                elif fill_method == "bfill":
                    result[col] = result.groupby(group_by)[col].bfill()
        else:
            # wide 格式: 逐列 rolling
            for col in value_cols:
                if col not in result.columns:
                    continue
                result[col] = (
                    result[col].rolling(window=window, min_periods=min_periods).agg(agg)
                )

                if fill_method == "ffill":
                    result[col] = result[col].ffill()
                elif fill_method == "bfill":
                    result[col] = result[col].bfill()

    ctx[output_name] = result
    return result
