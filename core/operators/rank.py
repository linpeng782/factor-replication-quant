"""
rank 算子
============================================================
分组排名（典型用法：截面排名 / 行业内截面排名）。

契约：
  - source_column   : str    必填
  - output_column   : str    必填
  - group_by        : list   必填（**必须显式包含 'date' 才是截面排名**，
                              不再有"检测到 date 列就自动加入分组"的隐式行为）
  - ascending       : bool   默认 True
  - pct             : bool   默认 False
  - rank_method     : str    默认 'average'（pandas.rank 的 method 参数）
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import Context, OpRegistry


@OpRegistry.register("rank")
def op_rank(ctx: Context, step: Dict, fetcher: Any) -> None:
    target_df = step.get("output_dataframe", "data")
    df = ctx.get_df(target_df)

    src = step["source_column"]
    out = step["output_column"]
    group_by = step.get("group_by")
    if not group_by or not isinstance(group_by, list):
        raise ValueError(
            "rank: group_by 必须为非空 list；想做日度截面排名请显式写 group_by: [date]"
        )

    pct = bool(step.get("pct", False))
    ascending = bool(step.get("ascending", True))
    rank_method = step.get("rank_method", "average")

    series = df.groupby(group_by)[src].rank(
        pct=pct, ascending=ascending, method=rank_method
    )
    ctx.add_column(target_df, out, series)
