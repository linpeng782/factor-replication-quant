"""
rolling_sorted_subset 算子
============================================================
逐股滚动窗口内**按另一列排序切割子集后聚合**：

  窗口 = 最近 window 个交易日；先按 mask_column 剔除无效日（如涨跌停/停牌），
  再把剩余 n_valid 天按 source_column_sort **升序**排序，取
    select='low'  → 排序最小的 k 天（k = floor(n_valid × frac)）
    select='high' → 排序最大的 k 天
  对这 k 天的 source_column 求 agg（sum / mean）。

典型用途：开源证券「长端动量」——160 日内按日振幅切割，取低振幅 70% 交易日的
日收益加总（振幅低的日子过度反应少 → 保留动量信息，剔除反转噪声）。

NaN 语义：source_column / source_column_sort 任一缺失，或 mask_column ≤ 0 的日子
视为无效日（不排序、不计入 n_valid）；n_valid < min_valid（默认 window//2）→ 输出 NaN。

契约：
  - source_column      : str    必填，被聚合的值列
  - source_column_sort : str    必填，排序列
  - output_column      : str    必填
  - window             : int    必填
  - frac               : float  可选，默认 0.7（取窗口有效日的比例）
  - select             : str    可选，'low'（默认）/ 'high'
  - agg                : str    可选，'sum'（默认）/ 'mean'
  - mask_column        : str    可选，有效日标记列（> 0 为有效）
  - min_valid          : int    可选，默认 window // 2
  - group_by           : str    默认 'order_book_id'
  - chunk_stocks       : int    可选，列分块大小（内存/速度权衡，默认 64）

实现：长表 → (date × stock) 宽表 → 逐列块 sliding_window_view + argsort + cumsum，
无 Python 逐股循环。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view

from . import Context, OpRegistry


_VALID_AGGS = frozenset({"sum", "mean"})


@OpRegistry.register("rolling_sorted_subset")
def op_rolling_sorted_subset(ctx: Context, step: Dict, fetcher: Any) -> None:
    target_df = step.get("output_dataframe", "data")
    df = ctx.get_df(target_df)

    src = step["source_column"]
    srt = step["source_column_sort"]
    out = step["output_column"]
    window = int(step["window"])
    frac = float(step.get("frac", 0.7))
    select = step.get("select", "low")
    agg = step.get("agg", "sum")
    mask_col = step.get("mask_column")
    min_valid = int(step.get("min_valid", window // 2))
    group_by = step.get("group_by", "order_book_id")
    chunk = int(step.get("chunk_stocks", 64))

    if agg not in _VALID_AGGS:
        raise ValueError(f"rolling_sorted_subset: agg={agg!r} 不支持；可选 {sorted(_VALID_AGGS)}")
    if select not in ("low", "high"):
        raise ValueError(f"rolling_sorted_subset: select={select!r} 只能是 low / high")
    if not 0 < frac <= 1:
        raise ValueError(f"rolling_sorted_subset: frac 必须在 (0,1]，实际 {frac}")

    # ⚠️ pivot_table 会丢掉「该值全为 NaN」的日期行 → 时间轴塌缩、滚动窗口错位。
    # 必须显式 reindex 回长表的完整 (date × stock) 轴。
    all_dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    all_stocks = sorted(df[group_by].unique())

    def _wide(col: str) -> pd.DataFrame:
        return (df.pivot_table(index="date", columns=group_by, values=col, aggfunc="last")
                  .reindex(index=all_dates, columns=all_stocks))

    wide_v = _wide(src)
    wide_s = _wide(srt)
    v = wide_v.to_numpy(dtype="float32")
    s = wide_s.to_numpy(dtype="float32")
    valid = np.isfinite(v) & np.isfinite(s)
    if mask_col:
        valid &= _wide(mask_col).to_numpy(dtype="float32") > 0

    v_eff = np.where(valid, v, 0.0).astype("float32")
    s_eff = np.where(valid, s, np.inf).astype("float32")   # 无效日排到窗口末尾
    T, N = v.shape

    res = np.full((T, N), np.nan, dtype="float64")
    if T >= window:
        for lo in range(0, N, chunk):
            hi = min(lo + chunk, N)
            sw_s = sliding_window_view(s_eff[:, lo:hi], window, axis=0)      # (T-W+1, C, W)
            sw_v = sliding_window_view(v_eff[:, lo:hi], window, axis=0)
            n_valid = sliding_window_view(valid[:, lo:hi], window, axis=0).sum(-1)

            order = np.argsort(sw_s, axis=-1, kind="stable")
            v_sorted = np.take_along_axis(sw_v, order, axis=-1)
            csum = np.cumsum(v_sorted, axis=-1, dtype="float64")

            k = np.maximum(np.floor(n_valid * frac).astype("int64"), 1)
            k = np.minimum(k, np.maximum(n_valid, 1))
            if select == "low":
                total = np.take_along_axis(csum, (k - 1)[..., None], axis=-1)[..., 0]
            else:
                # 最大的 k 天 = 有效日总和 − 最小的 (n_valid−k) 天
                lower = np.maximum(n_valid - k, 0)
                head = np.take_along_axis(
                    csum, np.maximum(lower - 1, 0)[..., None], axis=-1
                )[..., 0]
                head = np.where(lower > 0, head, 0.0)
                all_valid_sum = np.take_along_axis(
                    csum, np.maximum(n_valid - 1, 0)[..., None], axis=-1
                )[..., 0]
                total = all_valid_sum - head
            block = total / k if agg == "mean" else total
            block[n_valid < min_valid] = np.nan
            res[window - 1:, lo:hi] = block

    row_idx = wide_v.index.get_indexer(df["date"])
    col_idx = wide_v.columns.get_indexer(df[group_by])
    values = np.full(len(df), np.nan)
    ok = (row_idx >= 0) & (col_idx >= 0)
    values[ok] = res[row_idx[ok], col_idx[ok]]

    logger.info(
        f"[rolling_sorted_subset] {out}: window={window} frac={frac} select={select} "
        f"agg={agg} mask={mask_col} min_valid={min_valid} | "
        f"非空={np.isfinite(values).sum():,}/{len(values):,}"
    )
    ctx.add_column(target_df, out, pd.Series(values, index=df.index))
