"""
merge 算子
============================================================
显式跨 DataFrame 合并。把旧 compute 中"自动 merge_columns"的隐式行为剥离出来，
让 spec 里 merge 行为完全可见。

契约：
  - target_dataframe  : str   必填
  - source_dataframe  : str   必填
  - on                : list  必填——join 键（典型: [order_book_id, date]）
  - columns           : list  必填——要从 source 带过去的列
  - how               : str   可选，默认 'left'
"""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from . import Context, OpRegistry


@OpRegistry.register("merge")
def op_merge(ctx: Context, step: Dict, fetcher: Any) -> None:
    target_name = step["target_dataframe"]
    source_name = step["source_dataframe"]
    on = list(step["on"])
    columns = list(step["columns"])
    how = step.get("how", "left")

    target = ctx.get_df(target_name)
    source = ctx.get_df(source_name)

    needed = on + columns
    missing = [c for c in needed if c not in source.columns]
    if missing:
        raise ValueError(
            f"merge: source DataFrame {source_name!r} 缺少列 {missing}；"
            f"已有 {list(source.columns)}"
        )
    missing_target = [c for c in on if c not in target.columns]
    if missing_target:
        raise ValueError(
            f"merge: target DataFrame {target_name!r} 缺少 join 键 {missing_target}"
        )
    overlap = [c for c in columns if c in target.columns]
    if overlap:
        raise ValueError(
            f"merge: 待合并列 {overlap} 已在 target {target_name!r} 中存在，禁止覆盖"
        )

    merged = target.merge(source[needed], on=on, how=how)
    n_before = len(target)
    n_after = len(merged)
    if n_after != n_before:
        logger.warning(
            f"[merge] target 行数变化: {n_before:,} → {n_after:,}（how={how}, on={on}）"
        )

    # 缺失率统计（只看新增的几列）
    for c in columns:
        miss = merged[c].isna().sum()
        logger.info(
            f"[merge] +{c}: {len(merged) - miss:,}/{len(merged):,} 非空 "
            f"({(len(merged) - miss) / max(len(merged), 1):.2%})"
        )

    ctx.set_df(target_name, merged)
