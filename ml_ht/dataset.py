"""数据加载 + 截面预处理 + 标签构建 + DataLoader。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from core import config

LONG_TABLE_PATH = config.ML_ROOT / "ht" / "alpha158_long.parquet"
LABEL_PATH = config.LABELS_DIR / "forward_return_20d.parquet"

SPLIT = {
    "train": (None, "2017-11-30"),
    "valid": ("2018-01-01", "2019-12-31"),
    "test": ("2020-01-01", None),
}


def load_long_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """读长表，返回 numpy 数组（释放 DataFrame 节省内存）。

    Returns:
        features:  (n_rows, 158) float32
        labels:    (n_rows,) float32 — 原始 20d 远期收益
        can_train: (n_rows,) bool
        factor_names: list[str]
        index_dates:  (n_rows,) datetime64
        index_stocks: (n_rows,) str
    """
    df = pd.read_parquet(LONG_TABLE_PATH)

    meta_cols = {"has_factor", "can_buy", "has_label", "can_train"}
    factor_names = [c for c in df.columns if c not in meta_cols]

    features = df[factor_names].to_numpy(dtype=np.float32)
    can_train = df["can_train"].to_numpy(dtype=bool)
    index_dates = df.index.get_level_values(0).to_numpy()
    index_stocks = df.index.get_level_values(1).to_numpy()

    # 标签：宽表 stack → MultiIndex Series → reindex 到长表的 (date, stock)
    label_wide = pd.read_parquet(LABEL_PATH)
    label_long = label_wide.stack()  # Series, MultiIndex (datetime, stock_code)
    label_long.index.names = ["date", "stock_code"]
    labels = label_long.reindex(df.index).to_numpy(dtype=np.float32)
    del label_wide, label_long

    del df
    return features, labels, can_train, factor_names, index_dates, index_stocks


def binarize_labels(labels: np.ndarray, dates: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """二分类标签：每天在 can_train 池内取截面中位数，> 中位数 → 1，否则 → 0。

    中位数只在 mask=True 的样本上计算，保证训练池正负样本 50/50。
    """
    binary = np.zeros_like(labels, dtype=np.float32)
    unique_dates = np.unique(dates)
    for d in unique_dates:
        day_mask = mask & (dates == d)
        day_labels = labels[day_mask]
        valid = np.isfinite(day_labels)
        if valid.sum() < 2:
            continue
        median = np.median(day_labels[valid])
        # 中位数应用到当天所有样本（含 non-can_train，虽然它们不会进训练）
        all_today = dates == d
        binary[all_today] = (labels[all_today] > median).astype(np.float32)
    return binary


def cross_sectional_zscore(
    features: np.ndarray, dates: np.ndarray
) -> np.ndarray:
    """截面 zscore：每天独立减均值除标准差。"""
    z = np.empty_like(features)
    unique_dates = np.unique(dates)
    for d in unique_dates:
        mask = dates == d
        day = features[mask]
        mu = day.mean(axis=0, keepdims=True)
        sigma = day.std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        z[mask] = (day - mu) / sigma
    return z


def preprocess(
    features: np.ndarray,
    labels: np.ndarray,
    can_train: np.ndarray,
    dates: np.ndarray,
    stocks: np.ndarray,
) -> dict:
    """完整预处理流水线：MAD 去极值 → zscore + 二分类标签 + 时间切分。

    Returns:
        dict with keys: X_train, y_train, X_valid, y_valid, X_test, y_test,
                        test_dates, test_stocks, feature_names (set by caller)
    """
    # 只对 can_train=True 的样本做 zscore（有效样本决定均值/标准差）
    z = np.zeros_like(features)
    ct_dates = dates[can_train]
    unique_dates = np.unique(ct_dates)

    for d in unique_dates:
        ct_mask = can_train & (dates == d)
        day = features[ct_mask]
        if len(day) < 2:
            continue
        # 1) 中位数去极值（±5×MAD，研报 full.md:385）—— 必须在 zscore 之前，
        #    否则单个离群点会抬高该因子当天 std，压平全市场截面信号
        med = np.median(day, axis=0, keepdims=True)
        mad = np.median(np.abs(day - med), axis=0, keepdims=True)
        mad = np.where(mad < 1e-8, 1.0, mad)
        lo, hi = med - 5.0 * 1.4826 * mad, med + 5.0 * 1.4826 * mad
        day = np.clip(day, lo, hi)
        # 2) 标准化
        mu = day.mean(axis=0, keepdims=True)
        sigma = day.std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        z[ct_mask] = (day - mu) / sigma

    # 二分类标签（中位数只在 can_train 池内计算）
    y = binarize_labels(labels, dates, can_train)

    # 时间切分（train/valid、valid/test 各留 1 个月 embargo）
    date_series = pd.DatetimeIndex(dates)
    train_mask = can_train & (date_series <= "2017-11-30")
    # embargo: 2017-12
    valid_mask = can_train & (date_series >= "2018-01-01") & (date_series <= "2019-11-30")
    # embargo: 2019-12
    test_mask = can_train & (date_series >= "2020-01-01")

    result = {
        "X_train": z[train_mask],
        "y_train": y[train_mask],
        "X_valid": z[valid_mask],
        "y_valid": y[valid_mask],
        "X_test": z[test_mask],
        "y_test": y[test_mask],
        # 原始连续远期收益（rank-IC / 多空价差用，非二分类标签）
        "ret_train": labels[train_mask],
        "ret_valid": labels[valid_mask],
        "ret_test": labels[test_mask],
        "train_dates": dates[train_mask],
        "valid_dates": dates[valid_mask],
        "test_dates": dates[test_mask],
        "test_stocks": stocks[test_mask],
        "n_train": int(train_mask.sum()),
        "n_valid": int(valid_mask.sum()),
        "n_test": int(test_mask.sum()),
    }
    return result


def build_loaders(data: dict, batch_size: int = 8192) -> dict:
    """构建 PyTorch DataLoader。

    StockMLP 是逐样本 pointwise MLP（无 BatchNorm、无截面操作、BCEWithLogitsLoss 独立），
    batch 内容不影响梯度期望方向，只需 shuffle 消除 Adam 动量的 recency bias。
    num_workers=0：数据已在内存，多进程 IPC 开销反而更慢。
    """
    loaders = {}
    for split in ("train", "valid", "test"):
        X = torch.from_numpy(data[f"X_{split}"])
        y = torch.from_numpy(data[f"y_{split}"]).unsqueeze(1)
        ds = TensorDataset(X, y)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            num_workers=0,
            pin_memory=True,
        )
    return loaders
