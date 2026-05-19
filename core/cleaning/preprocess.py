"""
因子预处理：去极值 + 标准化
------------------------------------------------------------
- winsorize_mad    : 向量化 MAD 去极值（横截面）
- standardize_zscore: 向量化 z-score 标准化（横截面）
- prepare_factor   : mask → winsorize → standardize 一站式

所有操作均为横截面（按行），即每个交易日独立处理。
"""

import numpy as np
import pandas as pd
from loguru import logger


# 正态分布下 MAD 到 std 的换算系数
_MAD_SCALE = 1.4826


def winsorize_mad(df: pd.DataFrame, n: float = 3.0) -> pd.DataFrame:
    """
    MAD 横截面去极值（向量化）

    公式：
        med = 横截面中位数
        mad = median(|x - med|) × 1.4826
        clip 到 [med - n·mad, med + n·mad]

    参数:
        df: (T, N) 因子面板
        n : 阈值倍数，默认 3

    返回:
        (T, N) 去极值后的因子面板（保留 NaN 位置）
    """
    arr = df.values.astype(np.float64)
    med = np.nanmedian(arr, axis=1, keepdims=True)  # (T, 1)
    mad = np.nanmedian(np.abs(arr - med), axis=1, keepdims=True)  # (T, 1)
    mad = mad * _MAD_SCALE

    upper = med + n * mad
    lower = med - n * mad

    # mad == 0 的行（常数行）：跳过 clip 避免将所有值压成 median
    zero_mad_rows = (mad == 0).ravel()
    clipped = np.where(np.isnan(arr), arr, np.minimum(np.maximum(arr, lower), upper))
    if zero_mad_rows.any():
        clipped[zero_mad_rows, :] = arr[zero_mad_rows, :]

    return pd.DataFrame(clipped, index=df.index, columns=df.columns)


def standardize_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    z-score 横截面标准化（向量化）

    公式:
        z = (x - mean) / std        (ddof=0, nan-safe)

    std == 0 的行：整行置 NaN（避免除零）

    返回:
        (T, N) 标准化后的因子面板
    """
    arr = df.values.astype(np.float64)
    mean = np.nanmean(arr, axis=1, keepdims=True)
    std = np.nanstd(arr, axis=1, keepdims=True, ddof=0)
    std_safe = np.where(std > 0, std, np.nan)
    z = (arr - mean) / std_safe
    return pd.DataFrame(z, index=df.index, columns=df.columns)


def prepare_factor(
    factor: pd.DataFrame,
    pre_mask: pd.DataFrame,
    post_mask: pd.DataFrame,
    mad_n: float = 3.0,
) -> pd.DataFrame:
    """
    完整清洗链路：pre_mask → winsorize → standardize → post_mask

    实盘时序逻辑:
      1. T 日盘后: 用 pre_mask 过滤 ST / 停牌 / 新股
         （这三类 T 日就已知不可交易，提前剔除避免污染横截面分布）
      2. T 日盘后: 在干净的横截面上做 winsorize_mad + standardize_zscore
         （涨停股仍参与，因为它们是正常市场信号）
      3. T+1 日开盘前: 用 post_mask 过滤涨停
         （涨停股 T+1 日不能买入，最终从可交易集合里剔除）

    参数:
        factor   : (T, N) 原始因子面板
        pre_mask : (T, N) 布尔 mask，True = 参与因子分布
                   = NOT is_st AND NOT is_suspended AND NOT is_new_stock
        post_mask: (T, N) 布尔 mask，True = 可执行下单
                   = NOT is_limit_up
        mad_n    : MAD 去极值阈值倍数，默认 3

    返回:
        (T, N) 清洗后的因子面板（最终只在 pre_mask & post_mask 处有值）
    """
    # 对齐索引和列（以 pre_mask 为基准）
    factor = factor.reindex(index=pre_mask.index, columns=pre_mask.columns)

    # Step 0: 先把 inf 替换成 NaN（YOLO 引擎可能产生 inf）
    factor = factor.replace([np.inf, -np.inf], np.nan)

    n_raw = factor.notna().values.sum()

    # Step 1: 先过滤 ST / 停牌 / 新股，这三类不参与横截面分布计算
    factor = factor.where(pre_mask)
    n_after_pre = factor.notna().values.sum()

    # Step 2: 干净横截面上做 MAD 去极值
    factor = winsorize_mad(factor, n=mad_n)
    # Step 3: 干净横截面上做 z-score 标准化
    factor = standardize_zscore(factor)
    n_after_std = factor.notna().values.sum()

    # Step 4: 最后过滤涨停（涨停股参与了标准化，但最终不下单）
    factor = factor.where(post_mask)
    n_final = factor.notna().values.sum()

    logger.info(
        f"[preprocess] 清洗完成: 原始={n_raw:,} → "
        f"pre_mask 后={n_after_pre:,} → 标准化后={n_after_std:,} → "
        f"post_mask 后={n_final:,}"
    )
    return factor
