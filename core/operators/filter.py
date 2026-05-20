"""
filter 算子
============================================================
按布尔条件过滤行（不增列、不删列）。

契约：
  - condition         : str     —— pandas.query 表达式
  - output_dataframe  : str     —— 默认 "data"；过滤就地生效（覆盖原 DataFrame）
"""

from __future__ import annotations

from typing import Any, Dict

from . import Context, OpRegistry


@OpRegistry.register("filter")
def op_filter(ctx: Context, step: Dict, fetcher: Any) -> None:
    target_df = step.get("output_dataframe", "data")
    df = ctx.get_df(target_df)

    condition = step.get("condition")
    if not condition:
        raise ValueError("filter: condition 必填")

    filtered = df.query(condition).reset_index(drop=True)
    ctx.set_df(target_df, filtered)
