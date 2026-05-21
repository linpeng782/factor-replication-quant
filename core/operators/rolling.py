"""
rolling 算子
============================================================
长表滚动窗口聚合，两种模式：

  模式 A: 普通 rolling
    在每个 group 内对 source_column 做 rolling(window).agg

  模式 B: 变化日 rolling（设置 change_on）
    只在 change_on 列**值变化**的行采样，对 source_column 做 rolling
    然后用 fill_method 填回所有行（典型用途：财报期 ROIC 排名 → 过去 N 期最值）

契约：
  - source_column   : str    必填
  - output_column   : str    必填
  - window          : int    必填
  - min_periods     : int    默认 = window
  - agg             : str    可选: min/max/mean/std/sum/median/count
  - group_by        : str    默认 'order_book_id'（long 表必备）
  - change_on       : str    可选——指定后启用变化日 rolling，引用的列必须在同一 DataFrame
  - fill_method     : str    可选: ffill/bfill/none，默认 ffill；仅 change_on 模式生效
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from . import Context, OpRegistry


_VALID_AGGS = frozenset({"min", "max", "mean", "std", "sum", "median", "count"})


@OpRegistry.register("rolling")
def op_rolling(ctx: Context, step: Dict, fetcher: Any) -> None:
    target_df = step.get("output_dataframe", "data")
    df = ctx.get_df(target_df)

    src = step["source_column"]
    out = step["output_column"]
    window = int(step["window"])
    min_periods = int(step.get("min_periods", window))
    agg = step.get("agg", "min")
    group_by = step.get("group_by", "order_book_id")
    change_on = step.get("change_on")
    fill_method = step.get("fill_method", "ffill")

    if agg not in _VALID_AGGS:
        raise ValueError(f"rolling: agg={agg!r} 不支持；可选 {sorted(_VALID_AGGS)}")
    if group_by not in df.columns:
        raise ValueError(f"rolling: group_by={group_by!r} 不在 DataFrame 列中")

    # 排序保证 rolling 语义稳定（按 group_by + date）
    sort_cols = [group_by] + (["date"] if "date" in df.columns else [])
    df_sorted = df.sort_values(sort_cols).copy()

    if change_on is None:
        # ── 模式 A：普通 rolling ─────────────────────────
        rolled = df_sorted.groupby(group_by)[src].transform(
            lambda x: x.rolling(window=window, min_periods=min_periods).agg(agg)
        )
        result = rolled
    else:
        # ── 模式 B：变化日 rolling ────────────────────────
        if change_on not in df_sorted.columns:
            raise ValueError(
                f"rolling: change_on={change_on!r} 不在 DataFrame；"
                f"已有列 {list(df_sorted.columns)}"
            )

        change_mask = df_sorted.groupby(group_by)[change_on].transform(
            lambda x: x != x.shift(1)
        )
        rolled_full = pd.Series(pd.NA, index=df_sorted.index, dtype="float64")

        change_rows = df_sorted.loc[change_mask, [group_by, src]].dropna(subset=[src])
        if len(change_rows) > 0:
            rolled = change_rows.groupby(group_by)[src].transform(
                lambda x: x.rolling(window=window, min_periods=min_periods).agg(agg)
            )
            rolled_full.loc[rolled.index] = rolled.values

        if fill_method in ("ffill", "bfill"):
            rolled_full = (
                rolled_full.groupby(df_sorted[group_by])
                .transform(getattr(pd.Series, fill_method))
            )
        result = rolled_full

    # 把 result 按原始 index 顺序对齐回去
    out_series = result.reindex(df.index)
    ctx.add_column(target_df, out, out_series)
