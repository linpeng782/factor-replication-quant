"""
超额标签构造（路 A：截面等权 demean）
============================================================
  基础收益 = forward_return_20d = vwap[t+21]/vwap[t+1] − 1（已避前视，core 标签产出）
  超额     = r − 当日 pre_mask universe 的【等权均值】     （减市场 = 定义超额）
  标准化   = 对 excess 再套全集 RobustZScore（在 dataset/preprocess 阶段统一处理）

注：demean 用 pre_mask universe（剔 ST/停牌/新股；涨停股保留，是有效信号）。
    与中证全指（000985）超额相关 0.94、IC 完全等价（已实证）。
"""
from __future__ import annotations

import pandas as pd

from core import config


def load_forward_return(horizon: int = 20) -> pd.DataFrame:
    """读 LABELS_DIR/forward_return_{horizon}d.parquet → (T, N) 绝对收益面板（DatetimeIndex）。"""
    path = config.LABELS_DIR / f"forward_return_{horizon}d.parquet"
    if not path.exists():
        raise FileNotFoundError(f"标签不存在：{path}")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def build_excess_label(
    ret: pd.DataFrame,
    pre_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """截面等权 demean → 超额收益面板（与 ret 同 (T, N) 网格）。

    mkt[t] = mean_j( ret[t, j] | pre_mask[t, j] )   # 当日可投 universe 等权市场收益
    excess = ret.sub(mkt, axis=0)                    # 对【全体】股票减去当日市场（保留全网格）

    参数：
        ret      : (T, N) 绝对收益面板
        pre_mask : (T, N) 布尔；True=计入市场均值。None 则用 ret 全部非 NaN。
    """
    if pre_mask is not None:
        pre_mask = pre_mask.reindex(index=ret.index, columns=ret.columns)
        ret_for_mean = ret.where(pre_mask.astype("boolean").fillna(False))
    else:
        ret_for_mean = ret
    mkt = ret_for_mean.mean(axis=1, skipna=True)     # (T,) 当日市场等权收益
    excess = ret.sub(mkt, axis=0)
    return excess
