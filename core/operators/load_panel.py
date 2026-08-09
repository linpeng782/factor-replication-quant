"""
load_panel 算子
============================================================
从本地宽表 parquet（date × order_book_id）加载一列，merge 进 target DataFrame。
target DataFrame 尚不存在时，直接用该面板**创建主表**（按 ctx.universe +
[start_date, end_date] 裁剪），供纯面板型因子（如改进动量）起手。

典型用途：把 Ret20 后复权面板并入 APM 长表，供 cross_section_regress 使用。

Spec 契约：
  panel_config  : str   config 中的属性名（如 'RET20_PANEL_PATH'），或绝对路径
  output_column : str   合并到 target df 的列名
  output_dataframe : str  默认 'data'
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from loguru import logger

import config as cfg
from . import Context, OpRegistry


@lru_cache(maxsize=8)
def _load_panel_cached(path_str: str) -> pd.DataFrame:
    """加载宽表面板并缓存（date × stock，date=DatetimeIndex）。"""
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(
            f"面板文件不存在: {p}\n"
            "APM 使用 Ret20 面板请先运行: PYTHONPATH=. python scripts/build_ret20_panel.py"
        )
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df


def _resolve_path(panel_config: str) -> str:
    """panel_config 可以是 config 属性名（如 'RET20_PANEL_PATH'）或绝对路径。"""
    if hasattr(cfg, panel_config):
        return str(getattr(cfg, panel_config))
    return panel_config


def _slice_to_ctx(long: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    """按 ctx.universe + [start_date, end_date] 裁剪长表（建主表时用）。"""
    if ctx.universe:
        long = long[long["order_book_id"].isin(set(ctx.universe))]
    if ctx.start_date:
        long = long[long["date"] >= pd.Timestamp(ctx.start_date)]
    if ctx.end_date:
        long = long[long["date"] <= pd.Timestamp(ctx.end_date)]
    return long


@OpRegistry.register("load_panel")
def op_load_panel(ctx: Context, step: Dict, fetcher: Any) -> None:
    """加载本地宽表面板列 → merge 到 target df（df 不存在则用面板创建主表）。"""
    target_df_name = step.get("output_dataframe", "data")

    panel_config = step["panel_config"]
    out_col = step["output_column"]
    path_str = _resolve_path(panel_config)

    panel_wide = _load_panel_cached(path_str)

    # 宽表 → 长表：stack，保留 order_book_id 和 date
    # 列轴名各面板不一（order_book_id / stock / 无名）→ 统一重命名，不依赖原名
    long = (
        panel_wide
        .rename_axis(index="date", columns="order_book_id")
        .stack()
        .rename(out_col)
        .reset_index()
    )
    long["date"] = pd.to_datetime(long["date"])

    # 主表还不存在 → 本面板即主表来源，按 universe + 日期区间裁剪后建表
    if not ctx.has_df(target_df_name):
        created = _slice_to_ctx(long, ctx).sort_values(
            ["order_book_id", "date"]
        ).reset_index(drop=True)
        logger.info(
            f"[load_panel] {out_col}: 由 {Path(path_str).name} 创建主表 "
            f"{target_df_name!r}，{len(created):,} 行 × {created['order_book_id'].nunique():,} 股"
        )
        ctx.set_df(target_df_name, created)
        return

    df = ctx.get_df(target_df_name)

    # merge 进 df
    merged = df.merge(long, on=["date", "order_book_id"], how="left")
    merged.index = df.index

    logger.info(
        f"[load_panel] {out_col}: 从 {Path(path_str).name} 合并，"
        f"非空={merged[out_col].notna().sum():,}/{len(merged):,}"
    )
    ctx.add_column(target_df_name, out_col, merged[out_col])
