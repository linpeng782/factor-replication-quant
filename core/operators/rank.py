"""排名操作"""

import pandas as pd
from typing import Any, Dict

from . import OpRegistry, _resolve_input


@OpRegistry.register("rank")
def op_rank(ctx: Dict, step: Dict, fetcher: Any) -> pd.DataFrame:
    """
    action: rank
    排名操作（行业内排名 / 全市场排名）

    参数:
        method          : "rank" 或 "industry_rank"
        rank_column     : 需要排名的列名
        group_column    : 分组列名（industry_rank 时使用）
        pct             : bool, 是否百分比排名（默认 True）
        ascending       : bool, 是否升序（默认 True，即值越小排名越靠前）
        rank_method     : str, pandas rank 的 method 参数（默认 "average"）
                          可选: "average", "min", "max", "first", "dense"
        output_name     : 输出列名前缀/完整名
    """
    method = step.get("method", "rank")
    input_var = step.get("input", step.get("output"))
    output_name = step.get("output", "ranked")
    rank_col = step.get("rank_column", "")
    group_col = step.get("group_column", "")

    # 排名参数（新增）
    pct = step.get("pct", True)
    ascending = step.get("ascending", True)
    rank_method = step.get("rank_method", "average")
    output_col = step.get("output_col", f"{rank_col}_rank")

    df = _resolve_input(ctx, input_var)

    rank_kwargs = {
        "pct": pct,
        "ascending": ascending,
        "method": rank_method,
    }

    if method == "industry_rank":
        if not group_col:
            print("⚠️ industry_rank 需要 group_column，未指定则使用全市场排名")
            df[output_col] = df[rank_col].rank(**rank_kwargs)
        else:
            # 支持 long 格式：如果数据包含 date 列，按 [date, group_col] 分组
            # 确保每个交易日、每个行业内独立排名
            if "date" in df.columns:
                df[output_col] = df.groupby(["date", group_col])[rank_col].rank(**rank_kwargs)
            else:
                df[output_col] = df.groupby(group_col)[rank_col].rank(**rank_kwargs)
    elif method == "rank":
        # 支持 long 格式：如果数据包含 date 列，按 date 分组，每天独立排名
        if "date" in df.columns:
            df[output_col] = df.groupby("date")[rank_col].rank(**rank_kwargs)
        else:
            df[output_col] = df[rank_col].rank(**rank_kwargs)
    else:
        raise ValueError(f"不支持的 rank method: {method}")

    ctx[output_name] = df
    return df
