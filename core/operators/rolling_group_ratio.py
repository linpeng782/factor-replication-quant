"""
rolling_group_ratio 算子
============================================================
滚动窗口内"按 A 列排序分组、看 B 列组内占比"的通用算子（国盛"高/低位放量"系列④）。

对每个 group（默认逐股）在长表上滚动 window 行：
  1. 窗口内按 source_column_sort 从低到高排序，等分为 n_groups 组；
  2. 取 select 指定的组（high=排序最高组 / low=排序最低组）；
  3. 输出 = mean(组内 source_column) / mean(窗口内全部 source_column)。

典型用法（window=20, n_groups=5）：
  - 高位波动占比: sort=close_d,      value=intraday_vol, select=high
  - 高波价格占比: sort=intraday_vol, value=close_d,      select=high

契约：
  - source_column        : str  必填（取占比的 value 列）
  - source_column_sort   : str  必填（排序依据列）
  - output_column        : str  必填
  - window               : int  必填，必须能被 n_groups 整除
  - n_groups             : int  默认 5
  - select               : str  'high' | 'low'，默认 'high'
  - group_by             : str  默认 'order_book_id'

NaN 语义：窗口内 value/sort 任一存在 NaN → 该行输出 NaN（严格 min_periods=window，
与论文"过去 20 个交易日"口径一致；分钟长表天然只含交易日，无需额外对齐）。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import Context, OpRegistry


def _one_group(v: np.ndarray, s: np.ndarray, window: int, k: int, hi: bool) -> np.ndarray:
    """单 group 向量化：v=value, s=sort（等长 1D）→ 输出与 v 等长（前 window-1 行 NaN）。"""
    n = v.shape[0]
    out = np.full(n, np.nan)
    if n < window:
        return out
    vw = sliding_window_view(v, window)            # (n-window+1, window)
    sw = sliding_window_view(s, window)
    ok = np.isfinite(vw).all(axis=1) & np.isfinite(sw).all(axis=1)
    if not ok.any():
        return out
    order = np.argsort(sw[ok], axis=1, kind="stable")   # 低→高
    idx = order[:, -k:] if hi else order[:, :k]
    grp_mean = np.take_along_axis(vw[ok], idx, axis=1).mean(axis=1)
    tot_mean = vw[ok].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = grp_mean / tot_mean
    res = np.full(n - window + 1, np.nan)
    res[ok] = ratio
    out[window - 1:] = res
    return out


@OpRegistry.register("rolling_group_ratio")
def op_rolling_group_ratio(ctx: Context, step: Dict, fetcher: Any) -> None:
    target_df = step.get("output_dataframe", "data")
    df = ctx.get_df(target_df)

    val_col = step["source_column"]
    sort_col = step["source_column_sort"]
    out = step["output_column"]
    window = int(step["window"])
    n_groups = int(step.get("n_groups", 5))
    select = step.get("select", "high")
    group_by = step.get("group_by", "order_book_id")

    if select not in ("high", "low"):
        raise ValueError(f"rolling_group_ratio: select={select!r} 不支持；可选 high/low")
    if window % n_groups != 0:
        raise ValueError(
            f"rolling_group_ratio: window={window} 必须能被 n_groups={n_groups} 整除"
        )
    k = window // n_groups

    sort_cols = [group_by] + (["date"] if "date" in df.columns else [])
    df_sorted = df.sort_values(sort_cols)
    v_all = df_sorted[val_col].to_numpy(dtype=float)
    s_all = df_sorted[sort_col].to_numpy(dtype=float)

    result = np.full(len(df_sorted), np.nan)
    pos = 0
    for _, g in df_sorted.groupby(group_by, sort=False):
        m = len(g)
        result[pos:pos + m] = _one_group(
            v_all[pos:pos + m], s_all[pos:pos + m], window, k, select == "high"
        )
        pos += m

    out_series = pd.Series(result, index=df_sorted.index).reindex(df.index)
    ctx.add_column(target_df, out, out_series)
