"""
compute 算子
============================================================
按公式计算新列。**严格契约**，无任何静默兜底：
  - source_columns: list[str]（必填）—— 公式中可引用的列，必须全部已存在于目标 DataFrame
  - output_column : str（必填）
  - formula       : str          —— pandas eval 表达式；只能引用 source_columns 中的列名

公式中引用的标识符必须出现在 source_columns 中（除内置数学函数 abs/log/exp/sqrt 外），
否则 raise。这把"研究员拼错列名引擎自动找其他 DataFrame 兜底"的旧坑彻底堵死。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Set

import pandas as pd

from . import Context, OpRegistry


_BUILTINS: Set[str] = {"abs", "log", "exp", "sqrt", "where", "nan", "True", "False", "None"}


@OpRegistry.register("compute")
def op_compute(ctx: Context, step: Dict, fetcher: Any) -> None:
    target_df = step.get("output_dataframe", "data")
    df = ctx.get_df(target_df)

    formula = step.get("formula")
    if not formula:
        raise ValueError("compute: formula 必填")
    out = step["output_column"]
    sources: list = list(step["source_columns"])

    # 校验：formula 引用的所有标识符必须出现在 source_columns 中。
    # 用 lookbehind 排除"紧跟数字/点的标识符"（如 1e-12 中的 e、1.5e10 中的 e10），
    # 避免把科学计数法的尾部误判成跨表标识符。
    tokens = set(re.findall(r"(?<![0-9.])[a-zA-Z_][a-zA-Z0-9_]*", formula)) - _BUILTINS
    extra = tokens - set(sources)
    if extra:
        raise ValueError(
            f"compute: formula 引用了未在 source_columns 声明的标识符 {sorted(extra)}；"
            f"source_columns={sources}"
        )

    # 求值：限定 local_dict 只暴露声明的 source_columns + 安全字面量，杜绝跨表/跨列引用
    local = {c: df[c] for c in sources}
    local["nan"] = float("nan")
    try:
        result = pd.eval(formula, local_dict=local, engine="numexpr")
    except Exception:
        result = pd.eval(formula, local_dict=local, engine="python")

    ctx.add_column(target_df, out, result)
