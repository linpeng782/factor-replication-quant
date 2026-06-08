"""
load_panel 算子
============================================================
从本地宽表 parquet（date × order_book_id）加载一列，merge 进 target DataFrame。

典型用途：把 Ret20 后复权面板并入 APM 长表，供 cross_section_regress 使用。

Spec 契约：
  panel_config  : str   core.config 中的属性名（如 'RET20_PANEL_PATH'），或绝对路径
  output_column : str   合并到 target df 的列名
  output_dataframe : str  默认 'data'
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from loguru import logger

import core.config as cfg
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


@OpRegistry.register("load_panel")
def op_load_panel(ctx: Context, step: Dict, fetcher: Any) -> None:
    """加载本地宽表面板列 → merge 到 target df。"""
    target_df_name = step.get("output_dataframe", "data")
    df = ctx.get_df(target_df_name)

    panel_config = step["panel_config"]
    out_col = step["output_column"]
    path_str = _resolve_path(panel_config)

    panel_wide = _load_panel_cached(path_str)

    # 宽表 → 长表：stack，保留 order_book_id 和 date
    long = (
        panel_wide
        .rename_axis("date")
        .stack()
        .rename(out_col)
        .reset_index()
        .rename(columns={"level_1": "order_book_id"})
    )
    long["date"] = pd.to_datetime(long["date"])

    # merge 进 df
    merged = df.merge(long, on=["date", "order_book_id"], how="left")
    merged.index = df.index

    logger.info(
        f"[load_panel] {out_col}: 从 {Path(path_str).name} 合并，"
        f"非空={merged[out_col].notna().sum():,}/{len(merged):,}"
    )
    ctx.add_column(target_df_name, out_col, merged[out_col])
