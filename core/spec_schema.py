"""
Spec 静态校验
============================================================
在 YOLO 启动前完成四项强制检查，违反任一条立即 raise，绝不进 fetch：

  1. factor.column 必填，且必须等于某个 step 的 output_column
  2. 每个 source_column 必须能追溯到某个之前的 fetch 字段或 output_column
  3. 同一个 DataFrame 内 output_column 不允许重复（防覆盖隐患）
  4. merge 引用的 source/target_dataframe 必须先被 output_dataframe 定义过

实现思路：
    模拟执行 calculation_steps，建符号表 known_cols[df_name] = {col_name}，
    每步根据 action 检查输入列是否存在、输出列是否冲突，结束后看 factor.column 是否落地。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


SENTINEL_COLS = frozenset({"order_book_id", "date"})

# 不增列的 action（filter 只过滤行）
NO_OUTPUT_ACTIONS = frozenset({"filter"})

# 增列的 action（必须显式声明 output_column）
COLUMN_ADDING_ACTIONS = frozenset({"transform", "compute", "rank", "rolling"})


class SpecError(ValueError):
    """spec yaml 静态校验失败"""


@dataclass
class _SymbolTable:
    """模拟执行过程中各 DataFrame 已知列的符号表"""

    known: Dict[str, Set[str]] = field(default_factory=dict)

    def create(self, df_name: str) -> None:
        if df_name in self.known:
            return
        self.known[df_name] = set(SENTINEL_COLS)

    def has_df(self, df_name: str) -> bool:
        return df_name in self.known

    def cols(self, df_name: str) -> Set[str]:
        return self.known[df_name]

    def add(self, df_name: str, col: str, loc: str) -> None:
        if col in self.known[df_name]:
            raise SpecError(
                f"{loc}: output_column {col!r} 在 DataFrame {df_name!r} 中已存在，"
                f"禁止覆盖（已有列: {sorted(self.known[df_name])}）"
            )
        self.known[df_name].add(col)

    def require(self, df_name: str, col: str, loc: str, role: str = "source_column") -> None:
        if df_name not in self.known:
            raise SpecError(f"{loc}: DataFrame {df_name!r} 未被定义")
        if col not in self.known[df_name]:
            raise SpecError(
                f"{loc}: {role} {col!r} 不在 DataFrame {df_name!r}（已有列: "
                f"{sorted(self.known[df_name])}）"
            )


def validate_spec(spec_yaml: dict) -> None:
    """主入口：校验 spec yaml，违反任一条 raise SpecError"""
    factor = spec_yaml.get("factor")
    if not factor:
        raise SpecError("缺少顶层 factor 块")

    factor_name = factor.get("name")
    factor_column = factor.get("column")

    if not factor_name:
        raise SpecError("factor.name 必填")
    if not factor_column:
        raise SpecError("factor.column 必填（最终因子值取主表 'data' 的这一列）")

    steps = spec_yaml.get("calculation_steps", [])
    if not steps:
        raise SpecError("calculation_steps 不能为空")

    sym = _SymbolTable()

    for i, step in enumerate(steps, 1):
        action = step.get("action")
        if not action:
            raise SpecError(f"step #{i}: 缺少 action 字段")

        loc = f"step #{i} ({step.get('name') or action})"

        if action == "fetch":
            _validate_fetch(step, sym, loc)
        elif action == "merge":
            _validate_merge(step, sym, loc)
        elif action in COLUMN_ADDING_ACTIONS:
            _validate_column_adding(step, sym, loc)
        elif action in NO_OUTPUT_ACTIONS:
            _validate_filter(step, sym, loc)
        else:
            raise SpecError(f"{loc}: 未知 action {action!r}")

    # 最终检查 factor.column 在主表 data 中
    if not sym.has_df("data"):
        raise SpecError(
            "执行完所有 step 后主表 'data' 未被定义；至少要有一个 "
            "output_dataframe='data'（或省略，默认即 data）的 fetch step"
        )
    if factor_column not in sym.cols("data"):
        raise SpecError(
            f"factor.column={factor_column!r} 不在主表 'data' 中（已有列: "
            f"{sorted(sym.cols('data'))}）"
        )


def _validate_fetch(step: dict, sym: _SymbolTable, loc: str) -> None:
    target_df = step.get("output_dataframe", "data")
    sym.create(target_df)

    if "output_column" in step and "output_columns" in step:
        raise SpecError(f"{loc}: output_column 与 output_columns 不可同时指定")

    if "output_columns" in step:
        mapping = step["output_columns"]
        if not isinstance(mapping, dict) or not mapping:
            raise SpecError(f"{loc}: output_columns 必须是非空 dict {{field: col_name}}")
        for col in mapping.values():
            sym.add(target_df, col, loc)
    elif "output_column" in step:
        sym.add(target_df, step["output_column"], loc)
    else:
        raise SpecError(f"{loc}: fetch 必须指定 output_column 或 output_columns")


def _validate_merge(step: dict, sym: _SymbolTable, loc: str) -> None:
    target = step.get("target_dataframe")
    source = step.get("source_dataframe")
    if not target:
        raise SpecError(f"{loc}: merge 必须指定 target_dataframe")
    if not source:
        raise SpecError(f"{loc}: merge 必须指定 source_dataframe")
    if not sym.has_df(target):
        raise SpecError(f"{loc}: target_dataframe {target!r} 未被先前 step 定义")
    if not sym.has_df(source):
        raise SpecError(f"{loc}: source_dataframe {source!r} 未被先前 step 定义")

    on = step.get("on", [])
    if not on:
        raise SpecError(f"{loc}: merge 必须指定 on（join 键列表）")
    for k in on:
        sym.require(target, k, loc, role="merge.on")
        sym.require(source, k, loc, role="merge.on")

    cols = step.get("columns", [])
    if not cols:
        raise SpecError(f"{loc}: merge 必须指定 columns（要从 source 带过去的列）")
    for c in cols:
        sym.require(source, c, loc, role="merge.columns")
        sym.add(target, c, loc)


def _validate_column_adding(step: dict, sym: _SymbolTable, loc: str) -> None:
    target_df = step.get("output_dataframe", "data")
    if not sym.has_df(target_df):
        raise SpecError(
            f"{loc}: output_dataframe {target_df!r} 未被定义；请先 fetch 创建该 DataFrame"
        )

    sources = _collect_source_columns(step, loc)
    for c in sources:
        sym.require(target_df, c, loc)

    out = step.get("output_column")
    if not out:
        raise SpecError(f"{loc}: output_column 必填")
    sym.add(target_df, out, loc)


def _validate_filter(step: dict, sym: _SymbolTable, loc: str) -> None:
    target_df = step.get("output_dataframe", "data")
    if not sym.has_df(target_df):
        raise SpecError(f"{loc}: output_dataframe {target_df!r} 未被定义")
    # filter 的 condition 列引用校验放算子运行时（pandas.query 会自然报错）


def _collect_source_columns(step: dict, loc: str) -> List[str]:
    """规范化收集本步引用的输入列名（给 sym.require 用）"""
    if "source_column" in step and "source_columns" in step:
        raise SpecError(f"{loc}: source_column 与 source_columns 不可同时指定")
    if "source_column" in step:
        return [step["source_column"]]
    if "source_columns" in step:
        cols = step["source_columns"]
        if not isinstance(cols, list) or not cols:
            raise SpecError(f"{loc}: source_columns 必须是非空 list")
        return list(cols)
    raise SpecError(f"{loc}: 必须显式声明 source_column 或 source_columns")
