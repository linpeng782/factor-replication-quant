"""
Operators 注册中心和公共辅助函数
"""

import re
from typing import Dict, Optional

import pandas as pd


class OpRegistry:
    """元操作注册中心"""

    _ops = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func):
            cls._ops[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._ops:
            raise ValueError(f"未知的元操作: {name}，已注册: {list(cls._ops.keys())}")
        return cls._ops[name]


def _resolve_input(ctx: Dict, var_name: Optional[str]) -> pd.DataFrame:
    """从上下文中解析输入变量"""
    if var_name and var_name in ctx:
        val = ctx[var_name]
        if isinstance(val, pd.DataFrame):
            return val.copy()
    for key in reversed(list(ctx.keys())):
        if isinstance(ctx[key], pd.DataFrame):
            return ctx[key].copy()
    raise ValueError(f"无法在上下文中找到输入变量: {var_name}")


def _resolve_dataframe_for_expr(ctx: Dict, expr: str) -> pd.DataFrame:
    """从上下文中找到包含表达式所需列的数据框，支持跨表自动 merge"""
    skip_words = {"abs", "log", "exp", "sqrt", "if", "else", "and", "or", "not", "in", "is", "None", "True", "False"}
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expr))
    needed_cols = tokens - skip_words

    best_df = None
    best_score = -1
    for key, val in ctx.items():
        if isinstance(val, pd.DataFrame):
            score = sum(1 for col in val.columns if col in needed_cols)
            if score > best_score:
                best_score = score
                best_df = val

    if best_df is None:
        raise ValueError(f"无法为表达式找到数据框: {expr}")

    df = best_df.copy()
    existing_cols = set(df.columns)
    missing_cols = needed_cols - existing_cols

    if missing_cols:
        merge_keys = [c for c in ["order_book_id", "date", "quarter", "info_date"] if c in existing_cols]
        if not merge_keys:
            all_keys = None
            for key, val in ctx.items():
                if isinstance(val, pd.DataFrame):
                    cols = set(val.columns)
                    all_keys = cols if all_keys is None else all_keys & cols
            merge_keys = [c for c in ["order_book_id", "date", "quarter", "info_date"] if c in all_keys] if all_keys else []

        if not merge_keys:
            raise ValueError(f"表达式需要列 {missing_cols}，但找不到可用于 merge 的键")

        for key, val in ctx.items():
            if isinstance(val, pd.DataFrame) and val is not best_df:
                extra = missing_cols & set(val.columns)
                if extra:
                    cols_to_merge = list(extra | set(merge_keys))
                    df = df.merge(val[cols_to_merge], on=merge_keys, how="left")
                    missing_cols -= extra
                    if not missing_cols:
                        break

        if missing_cols:
            raise ValueError(f"表达式需要列 {missing_cols}，但在所有 DataFrame 中都找不到")

    return df
