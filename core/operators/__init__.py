"""
算子注册中心 + 极简执行上下文
============================================================

设计契约（与 core/spec_schema.py 一致）：
  - 整条流水线维护一组**命名 DataFrame**（默认主表叫 "data"），由 Context 持有
  - 每个算子读 Context 的某个 DataFrame，**新增列**写回，永不覆盖
  - 上下文不再藏 silent fallback（旧 _resolve_input / _resolve_dataframe_for_expr 全部删除）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd
from loguru import logger


# ── 算子注册表 ──────────────────────────────────────────────


class OpRegistry:
    _ops: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func):
            cls._ops[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        if name not in cls._ops:
            raise ValueError(
                f"未知算子 {name!r}，已注册: {sorted(cls._ops.keys())}"
            )
        return cls._ops[name]


# ── 执行上下文 ──────────────────────────────────────────────


@dataclass
class Context:
    """YOLO 执行期上下文。所有命名 DataFrame 都挂在 dataframes 里。"""

    factor_name: str
    universe: List[str] = field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    start_quarter: Optional[str] = None
    end_quarter: Optional[str] = None
    trade_date: Optional[str] = None
    dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # ── DataFrame 访问 ──

    def has_df(self, name: str) -> bool:
        return name in self.dataframes

    def get_df(self, name: str) -> pd.DataFrame:
        if name not in self.dataframes:
            raise KeyError(
                f"DataFrame {name!r} 不存在；当前已有: {sorted(self.dataframes.keys())}"
            )
        return self.dataframes[name]

    def set_df(self, name: str, df: pd.DataFrame) -> None:
        """整体替换（仅 fetch / merge 用，普通算子不要走这条）"""
        self.dataframes[name] = df

    def add_column(self, df_name: str, col_name: str, series) -> None:
        """在指定 DataFrame 上新增列。冲突立即 raise，永不覆盖。"""
        df = self.get_df(df_name)
        if col_name in df.columns:
            raise ValueError(
                f"列 {col_name!r} 已存在于 DataFrame {df_name!r}，"
                f"算子层禁止覆盖（spec_schema 应该已经在静态阶段拦截）"
            )
        df[col_name] = series

    # ── 调试 ──

    def schema_snapshot(self) -> Dict[str, List[str]]:
        return {name: list(df.columns) for name, df in self.dataframes.items()}

    def log_schema_diff(self, before: Dict[str, List[str]], step_label: str) -> None:
        after = self.schema_snapshot()
        all_dfs = sorted(set(before) | set(after))
        for name in all_dfs:
            old = set(before.get(name, []))
            new = set(after.get(name, []))
            added = sorted(new - old)
            removed = sorted(old - new)
            n_rows = len(self.dataframes[name]) if name in self.dataframes else 0
            if added or removed or name not in before:
                msg = f"  [{step_label}] df={name!r} rows={n_rows:,}"
                if added:
                    msg += f", +cols={added}"
                if removed:
                    msg += f", -cols={removed}"
                logger.info(msg)
