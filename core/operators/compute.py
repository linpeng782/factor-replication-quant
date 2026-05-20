"""公式计算操作"""

import pandas as pd
from typing import Any, Dict

from . import OpRegistry, _resolve_input, _resolve_dataframe_for_expr


@OpRegistry.register("compute")
def op_compute(ctx: Dict, step: Dict, fetcher: Any) -> pd.DataFrame:
    """
    action: compute
    根据 formula 计算新列
    """
    formula = step.get("formula", "")
    output_name = step.get("output", "computed")
    input_var = step.get("input", None)

    if "=" in formula:
        lhs, rhs = formula.split("=", 1)
        col_name = lhs.strip()
        expr = rhs.strip()
    else:
        col_name = output_name
        expr = formula

    merge_columns = step.get("merge_columns")

    # 优先使用 input 指定的 DataFrame，避免从 ctx 中误选同名列的旧数据
    if input_var and input_var in ctx and isinstance(ctx[input_var], pd.DataFrame):
        df = ctx[input_var].copy()
        # 如果有 merge_columns，从 ctx 中其他 DataFrame 补齐缺失列
        if merge_columns:
            missing = [c for c in merge_columns if c not in df.columns]
            if missing:
                merge_keys = [
                    c
                    for c in ["order_book_id", "date", "quarter", "info_date"]
                    if c in df.columns
                ]
                for key, val in ctx.items():
                    if not isinstance(val, pd.DataFrame) or key == input_var:
                        continue
                    extra = set(missing) & set(val.columns)
                    if extra and merge_keys:
                        cols = list(extra | (set(merge_keys) & set(val.columns)))
                        df = df.merge(
                            val[cols],
                            on=[k for k in merge_keys if k in val.columns],
                            how="left",
                        )
                        missing = [c for c in missing if c not in df.columns]
                    if not missing:
                        break
    else:
        df = _resolve_dataframe_for_expr(ctx, expr, extra_cols=merge_columns)

    try:
        df[col_name] = pd.eval(expr, local_dict=df, engine="numexpr")
    except Exception as e:
        try:
            df[col_name] = df.eval(expr)
        except Exception:
            raise ValueError(f"公式求值失败: {formula}, 错误: {e}")

    ctx[output_name] = df
    return df
