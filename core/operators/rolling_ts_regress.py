"""
rolling_ts_regress 算子
============================================================
APM 因子的核心 L3 算子：逐股、每日用过去 window 日（默认 20 日）做滚动时序回归，
输出统计量 stat = mean(δ) / (std(δ)/√n)。

数学（论文 §凤鸣朝阳 L86-135 / 开源系列5 L88-102）：
  在窗口 [t-W+1, t] 内，把两类"半日"收益（各 W 个）合并成 2W 个观测：
    y_i = r_seg1_i（股票"段1"收益，如 ret_overnight）
    X_i = R_seg1_i（指数"段1"收益）
  以及对应的"段2"（如 ret_pm）共 2W 对，做单次 OLS：
    y_i = α + β·X_i + ε_i
  残差差值 δ_t = ε_seg1_t - ε_seg2_t（共 W 个）。
  stat = mean(δ) / (std(δ, ddof=1) / √n_valid)。

实现：矢量化 sliding_window_view（无 Python 循环）+ 闭合式 OLS（无矩阵分解）。

Spec 契约：
  source_column_seg1  : str   股票段1收益列（如 ret_overnight）
  source_column_seg2  : str   股票段2收益列（如 ret_pm）
  index_col_seg1      : str   指数段1收益列名（在 INDEX_SEGMENTS_PATH 文件中）
  index_col_seg2      : str   指数段2收益列名
  window              : int   默认 20
  min_obs             : int   最小有效观测数（默认 window，可设小值允许缺失）
  output_column       : str
  group_by            : str   默认 'order_book_id'
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import numpy as np
import numpy.lib.stride_tricks as nst
import pandas as pd
from loguru import logger

import core.config as cfg
from . import Context, OpRegistry


@lru_cache(maxsize=4)
def _load_index_segments(path_str: str) -> pd.DataFrame:
    """加载指数分段收益宽表（date → 各段收益列），缓存避免重复 IO。"""
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(
            f"指数段收益文件不存在: {p}\n"
            "请先运行: PYTHONPATH=. python data_fetching/index_segments.py --full"
        )
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    return df


def _rolling_stat(
    r1: np.ndarray,  # (T,) 股票段1收益
    r2: np.ndarray,  # (T,) 股票段2收益
    R1: np.ndarray,  # (T,) 指数段1收益
    R2: np.ndarray,  # (T,) 指数段2收益
    W: int,
    min_obs: int,
) -> np.ndarray:
    """矢量化 rolling W-day stat。返回 (T,) 数组（前 W-1 个为 NaN）。"""
    T = len(r1)
    if T < W:
        return np.full(T, np.nan)

    # sliding_window_view: (T-W+1, W)
    r1w = nst.sliding_window_view(r1, W)   # shape (T-W+1, W)
    r2w = nst.sliding_window_view(r2, W)
    R1w = nst.sliding_window_view(R1, W)
    R2w = nst.sliding_window_view(R2, W)

    # 合并 2W 观测: shape (T-W+1, 2W)
    y = np.concatenate([r1w, r2w], axis=1)
    X = np.concatenate([R1w, R2w], axis=1)

    # 有效 mask（均有限）
    valid = np.isfinite(y) & np.isfinite(X)
    n_v = valid.sum(axis=1)  # (T-W+1,) 有效观测数

    # 闭合式 OLS（无矩阵分解，快）：β = (n·ΣXY - ΣX·ΣY) / (n·ΣX² - (ΣX)²)
    Xv = np.where(valid, X, 0.0)
    yv = np.where(valid, y, 0.0)
    n   = n_v.astype(float)
    Sx  = Xv.sum(axis=1)
    Sy  = yv.sum(axis=1)
    Sxx = (Xv * Xv).sum(axis=1)
    Sxy = (Xv * yv).sum(axis=1)
    denom = n * Sxx - Sx * Sx
    beta  = np.where(np.abs(denom) > 1e-12, (n * Sxy - Sx * Sy) / denom, 0.0)
    # alpha = (Sy - beta*Sx) / n（n>0 guard）
    alpha = np.where(n > 0, (Sy - beta * Sx) / n, 0.0)

    # 残差 (T-W+1, 2W)
    eps   = y - alpha[:, None] - beta[:, None] * X
    eps   = np.where(valid, eps, np.nan)

    # δ_t = ε_seg1_t - ε_seg2_t，在窗口内各日对齐（前 W 列=段1，后 W 列=段2）
    delta = eps[:, :W] - eps[:, W:]   # (T-W+1, W)

    # stat = mean(δ) / (std(δ, ddof=1) / √n_valid_delta)
    delta_valid = np.isfinite(delta)
    nd = delta_valid.sum(axis=1).astype(float)
    mu  = np.nansum(delta, axis=1) / np.where(nd > 0, nd, 1.0)
    # var = E[δ²] - μ²（sum of squares）
    sq  = np.nansum(delta ** 2, axis=1)
    var = np.where(nd > 1, (sq - nd * mu ** 2) / (nd - 1), np.nan)
    std = np.sqrt(np.maximum(var, 0.0))

    stat = np.where(
        (nd >= min_obs) & (std > 1e-12),
        mu / (std / np.sqrt(nd)),
        np.nan,
    )
    # 在结果前面补 W-1 个 NaN，对齐原始序列
    return np.concatenate([np.full(W - 1, np.nan), stat])


@OpRegistry.register("rolling_ts_regress")
def op_rolling_ts_regress(ctx: Context, step: Dict, fetcher: Any) -> None:
    """滚动时序回归（APM stat）算子。"""
    target_df_name = step.get("output_dataframe", "data")
    df = ctx.get_df(target_df_name)

    col1 = step["source_column_seg1"]
    col2 = step["source_column_seg2"]
    idx_col1 = step["index_col_seg1"]
    idx_col2 = step["index_col_seg2"]
    W   = int(step.get("window", 20))
    out = step["output_column"]
    grp = step.get("group_by", "order_book_id")
    min_obs = int(step.get("min_obs", max(W // 2, 5)))

    # 加载指数段收益并合并（共享 LRU 缓存，不重复 IO）
    idx_path = str(cfg.INDEX_SEGMENTS_PATH)
    idx_df = _load_index_segments(idx_path)
    if idx_col1 not in idx_df.columns or idx_col2 not in idx_df.columns:
        raise ValueError(
            f"rolling_ts_regress: 指数列 {idx_col1!r}/{idx_col2!r} 不在 {idx_path}\n"
            f"可用列: {list(idx_df.columns)}"
        )

    # 临时把指数列 join 进 df（按 date 对齐）
    df = df.merge(
        idx_df[[idx_col1, idx_col2]].rename_axis("date").reset_index(),
        on="date", how="left",
    )

    # 对每只股票做滚动 OLS
    result = pd.Series(np.nan, index=df.index, dtype="float64")

    for stock, g in df.groupby(grp, sort=False):
        g_sorted = g.sort_values("date")
        r1 = g_sorted[col1].to_numpy(float)
        r2 = g_sorted[col2].to_numpy(float)
        R1 = g_sorted[idx_col1].to_numpy(float)
        R2 = g_sorted[idx_col2].to_numpy(float)
        stat_arr = _rolling_stat(r1, r2, R1, R2, W, min_obs)
        result.loc[g_sorted.index] = stat_arr

    logger.info(
        f"[rolling_ts_regress] {out}: "
        f"非空={result.notna().sum():,}/{len(result):,}"
    )
    ctx.add_column(target_df_name, out, result)
