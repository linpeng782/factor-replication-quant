"""条件过滤操作"""

import pandas as pd
from typing import Any, Dict

from . import OpRegistry, _resolve_input, _resolve_dataframe_for_expr


@OpRegistry.register("filter")
def op_filter(ctx: Dict, step: Dict, fetcher: Any) -> pd.DataFrame:
    """
    action: filter
    按条件过滤数据
    """
    condition = step.get("condition", "")
    output_name = step.get("output", "filtered")

    input_var = step.get("input")
    if input_var and input_var in ctx:
        df = _resolve_input(ctx, input_var)
    else:
        best_key = None
        for key in reversed(list(ctx.keys())):
            if not key.startswith("_") and isinstance(ctx[key], pd.DataFrame):
                best_key = key
                break
        if best_key:
            df = ctx[best_key].copy()
        else:
            df = _resolve_dataframe_for_expr(ctx, condition)

    filtered = df.query(condition)
    ctx[output_name] = filtered
    return filtered
