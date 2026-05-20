"""
row_aggregate 算子
============================================================
跨多列做行向（axis=1）聚合，把 N 列折叠为 1 列。

典型用途：当因子需要"在已经存在的多个相关列上做 mean / std / min / max" 时，
本算子比硬写公式（如展开成 7 项平方和）更直观。

注意：很多场景可以用代数恒等式（望远镜求和、方差恒等式 Var=E[X²]−E[X]²）
绕开本算子。比如 npf_mrq_sue8 的 std_diff 是用方差恒等式从 sum_sq + mean
反推的，没用 row_aggregate；写起来反而更紧凑。本算子价值在于**列数大、
列名规则、行向聚合直观胜过代数变形**的场景。

契约：
  - source_columns : list[str]    必填，行向聚合的输入列
  - output_column  : str          必填
  - agg            : str          必填: mean / std / min / max / sum / median / count
  - ddof           : int          仅 agg=std/var 时有意义（默认 1，即样本标准差）
  - skipna         : bool         默认 True；True 时跨列任意 NaN 不传播
"""

from __future__ import annotations

from typing import Any, Dict

from . import Context, OpRegistry


_VALID_AGGS = frozenset({"mean", "std", "var", "min", "max", "sum", "median", "count"})


@OpRegistry.register("row_aggregate")
def op_row_aggregate(ctx: Context, step: Dict, fetcher: Any) -> None:
    target_df = step.get("output_dataframe", "data")
    df = ctx.get_df(target_df)

    sources = list(step["source_columns"])
    out = step["output_column"]
    agg = step["agg"]
    ddof = int(step.get("ddof", 1))
    skipna = bool(step.get("skipna", True))

    if agg not in _VALID_AGGS:
        raise ValueError(
            f"row_aggregate: 不支持的 agg={agg!r}；可选 {sorted(_VALID_AGGS)}"
        )

    sub = df.loc[:, sources]

    if agg in ("std", "var"):
        result = getattr(sub, agg)(axis=1, ddof=ddof, skipna=skipna)
    elif agg == "count":
        result = sub.count(axis=1)
    else:
        result = getattr(sub, agg)(axis=1, skipna=skipna)

    ctx.add_column(target_df, out, result)
