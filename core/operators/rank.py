"""排名操作"""

import pandas as pd
from typing import Any, Dict

from . import OpRegistry, _resolve_input


@OpRegistry.register("rank")
def op_rank(ctx: Dict, step: Dict, fetcher: Any) -> pd.DataFrame:
    """
    action: rank
    排名操作（行业内排名 / 全市场排名）
    """
    method = step.get("method", "rank")
    input_var = step.get("input", step.get("output"))
    output_name = step.get("output", "ranked")
    rank_col = step.get("rank_column", "")
    group_col = step.get("group_column", "")

    df = _resolve_input(ctx, input_var)

    if method == "industry_rank":
        if not group_col:
            print("⚠️ industry_rank 需要 group_column，未指定则使用全市场排名")
            df[f"{rank_col}_rank"] = df[rank_col].rank(pct=True, ascending=True)
        else:
            df[f"{rank_col}_rank"] = df.groupby(group_col)[rank_col].rank(pct=True, ascending=True)
    elif method == "rank":
        df[f"{rank_col}_rank"] = df[rank_col].rank(pct=True, ascending=True)
    else:
        raise ValueError(f"不支持的 rank method: {method}")

    ctx[output_name] = df
    return df
