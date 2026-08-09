"""
rolling_weighted_mean 算子
============================================================
逐股滚动窗口的**归一化加权平均**：

    out_t = Σ_{i=0..W-1} d_i · w_{t-i} · v_{t-i}  /  Σ_{i=0..W-1} d_i · w_{t-i}

  v = source_column（如日收益率），w = weight_column（如日换手率），
  d_i = exp(-i / decay_scale) 为可选的指数时间衰减（i=0 即当日；不设则 d≡1）。

典型用途：华泰改进动量因子
  wgt_return_Nm     = 换手率加权平均日收益（decay_scale 不设）
  exp_wgt_return_Nm = 换手率 × exp(-x_i/(4N)) 加权平均日收益（decay_scale=4N）

NaN 语义：v 或 w 任一缺失（停牌等）→ 该日整体不进分子/分母，也不计入有效天数；
窗口内有效天数 < min_periods（默认 window//2）→ 输出 NaN。

契约：
  - source_column : str    必填，被加权的值列
  - weight_column : str    必填，权重列（须非负）
  - output_column : str    必填
  - window        : int    必填，回看交易日数（含当日）
  - decay_scale   : float  可选，指数衰减尺度（交易日）；不设 = 不衰减
  - min_periods   : int    可选，默认 window // 2
  - group_by      : str    默认 'order_book_id'（仅作合法性校验，实际按宽表逐股）

实现：长表 → (date × stock) 宽表 → W 次移位累加（O(W·T·N) 纯 numpy，无 Python 逐股循环）。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger

from . import Context, OpRegistry


@OpRegistry.register("rolling_weighted_mean")
def op_rolling_weighted_mean(ctx: Context, step: Dict, fetcher: Any) -> None:
    target_df = step.get("output_dataframe", "data")
    df = ctx.get_df(target_df)

    src = step["source_column"]
    wgt = step["weight_column"]
    out = step["output_column"]
    window = int(step["window"])
    min_periods = int(step.get("min_periods", window // 2))
    decay_scale = step.get("decay_scale")
    group_by = step.get("group_by", "order_book_id")

    if window <= 0:
        raise ValueError(f"rolling_weighted_mean: window 必须为正整数，实际 {window}")
    if group_by not in df.columns or "date" not in df.columns:
        raise ValueError(
            f"rolling_weighted_mean: 需要 {group_by!r} + 'date' 列；现有 {list(df.columns)}"
        )

    # ⚠️ pivot_table 会丢掉「该值全为 NaN」的日期行 → 时间轴塌缩、滚动窗口错位。
    # 必须显式 reindex 回长表的完整 (date × stock) 轴。
    all_dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    all_stocks = sorted(df[group_by].unique())

    def _wide(col: str) -> pd.DataFrame:
        return (df.pivot_table(index="date", columns=group_by, values=col, aggfunc="last")
                  .reindex(index=all_dates, columns=all_stocks))

    wide_v = _wide(src)
    wide_w = _wide(wgt)

    v = wide_v.to_numpy(dtype="float64")
    w = wide_w.to_numpy(dtype="float64")
    valid = np.isfinite(v) & np.isfinite(w)
    vw = np.where(valid, v * w, 0.0)
    ww = np.where(valid, w, 0.0)
    cnt = valid.astype("float64")

    # 时间衰减权重：d_i = exp(-i/decay_scale)，i 为距当日的交易日数
    decay = (
        np.ones(window)
        if decay_scale is None
        else np.exp(-np.arange(window) / float(decay_scale))
    )

    T = v.shape[0]
    num = np.zeros_like(vw)
    den = np.zeros_like(ww)
    n_eff = np.zeros_like(cnt)
    for i in range(min(window, T)):
        d = decay[i]
        num[i:] += d * vw[: T - i]
        den[i:] += d * ww[: T - i]
        n_eff[i:] += cnt[: T - i]

    with np.errstate(divide="ignore", invalid="ignore"):
        res = num / den
    res[(n_eff < min_periods) | (den <= 0)] = np.nan

    # 宽表结果按 (date, stock) 对齐回长表原行序
    row_idx = wide_v.index.get_indexer(df["date"])
    col_idx = wide_v.columns.get_indexer(df[group_by])
    values = np.full(len(df), np.nan)
    ok = (row_idx >= 0) & (col_idx >= 0)
    values[ok] = res[row_idx[ok], col_idx[ok]]

    logger.info(
        f"[rolling_weighted_mean] {out}: window={window} decay_scale={decay_scale} "
        f"min_periods={min_periods} | 非空={np.isfinite(values).sum():,}/{len(values):,}"
    )
    ctx.add_column(target_df, out, pd.Series(values, index=df.index))
