"""公式计算操作"""

import pandas as pd
from typing import Any, Dict

from . import OpRegistry, _resolve_dataframe_for_expr


@OpRegistry.register("compute")
def op_compute(ctx: Dict, step: Dict, fetcher: Any) -> pd.DataFrame:
    """
    action: compute
    根据 formula 计算新列
    """
    formula = step.get("formula", "")
    output_name = step.get("output", "computed")

    if "=" in formula:
        lhs, rhs = formula.split("=", 1)
        col_name = lhs.strip()
        expr = rhs.strip()
    else:
        col_name = output_name
        expr = formula

    df = _resolve_dataframe_for_expr(ctx, expr)

    try:
        df[col_name] = pd.eval(expr, local_dict=df, engine="numexpr")
    except Exception as e:
        try:
            df[col_name] = df.eval(expr)
        except Exception:
            raise ValueError(f"公式求值失败: {formula}, 错误: {e}")

    ctx[output_name] = df
    return df
