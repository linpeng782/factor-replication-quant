"""
分钟数据读时复权 loader（新版 minute/raw/ 按日分片）
============================================================
磁盘只存【原始不复权】分钟（minute/raw/<YYYY-MM-DD>.parquet，全股一日一文件，含 order_book_id）。
后复权在【读时】实时算：价格 × ffill(ex_cum_factor)，量/额不动（与旧 1m_post 口径一致）。

主接口：
  load_adjusted_minute_window(start, end, stocks=None)
    → 读 [start,end] 的日文件集 → 拼接 → 价格后复权 → 返回长表(order_book_id,datetime,OHLCV...)

设计见 docs/minute_incremental_design.md（L1 读时复权）。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import config

PRICE_FIELDS = ["open", "high", "low", "close"]   # 仅价格 × cum_factor；量/额本就是原始


def _date_files(start, end) -> list[Path]:
    """返回 [start,end] 区间内存在的 per-date 文件（按日期升序）。"""
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    out = []
    for p in sorted(config.MINUTE_RAW_DIR.glob("*.parquet")):
        try:
            d = pd.Timestamp(p.stem)
        except ValueError:
            continue
        if s <= d <= e:
            out.append(p)
    return out


@lru_cache(maxsize=8192)
def _cum_factor_series(stock: str) -> pd.Series | None:
    """读单股稀疏 ex_cum_factor（DatetimeIndex 升序）；无则 None。"""
    p = config.MINUTE_EX_FACTORS_DIR / f"{stock}.parquet"
    if not p.exists():
        return None
    ex = pd.read_parquet(p, columns=["ex_cum_factor"])
    ex.index = pd.to_datetime(ex.index)
    return ex["ex_cum_factor"].sort_index()


def _post_adjust(df: pd.DataFrame, stock: str) -> pd.DataFrame:
    """对一只股票的原始分钟做后复权：价格 × ffill(cum_factor by 日)，量/额不动。"""
    df = df.copy()
    cf_ser = _cum_factor_series(stock)
    days = pd.to_datetime(df["datetime"]).dt.normalize()
    if cf_ser is None or cf_ser.empty:
        cf = np.ones(len(df), dtype="float64")
    else:
        cf = cf_ser.reindex(days.values, method="ffill").fillna(1.0).to_numpy()
    for c in PRICE_FIELDS:
        df[c] = (df[c].to_numpy() * cf).astype("float32")
    return df


def load_adjusted_minute_window(start, end, stocks: list[str] | None = None) -> pd.DataFrame:
    """读 [start,end] per-date 原始分钟 → 后复权 → 长表。

    返回列：order_book_id, datetime, open, high, low, close, volume, total_turnover（价格后复权）。
    """
    files = _date_files(start, end)
    if not files:
        return pd.DataFrame()
    parts = []
    for p in files:
        d = pd.read_parquet(p)
        if stocks is not None:
            d = d[d["order_book_id"].isin(stocks)]
        if len(d):
            parts.append(d)
    if not parts:
        return pd.DataFrame()
    allm = pd.concat(parts, ignore_index=True)
    allm["datetime"] = pd.to_datetime(allm["datetime"])
    # 逐股后复权（显式遍历分组，避免 groupby.apply 操作分组列的弃用告警）
    out = [_post_adjust(g, stock) for stock, g in allm.groupby("order_book_id", sort=False)]
    return pd.concat(out, ignore_index=True)
