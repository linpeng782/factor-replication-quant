"""选股评价指标（自包含 numpy 实现，无 scipy 依赖）。

按日截面计算排序类指标，用于训练中监控 + 训练后 test 评估。
预测值 vs 真实远期收益（连续），而非 vs 二分类标签——后者信息量低。
"""
from __future__ import annotations

import numpy as np


def _ordinal_rank(a: np.ndarray) -> np.ndarray:
    """平均秩（处理并列）。"""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    # 并列取平均秩
    a_sorted = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i + 1
        while j < n and a_sorted[j] == a_sorted[i]:
            j += 1
        if j - i > 1:
            avg = (i + j - 1) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 相关 = 秩上的 Pearson。"""
    rx = _ordinal_rank(x)
    ry = _ordinal_rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else np.nan


def daily_rank_ic(
    pred: np.ndarray, ret: np.ndarray, dates: np.ndarray, min_stocks: int = 10
) -> dict:
    """逐日 rank-IC，返回 mean / icir / 正占比 / 日序列。

    pred: 预测值（概率或 logit，单调即可）
    ret:  真实远期收益（连续）
    dates: 与 pred/ret 对齐的日期数组
    """
    ics = []
    for d in np.unique(dates):
        m = dates == d
        p, r = pred[m], ret[m]
        v = np.isfinite(r) & np.isfinite(p)
        if v.sum() < min_stocks:
            continue
        ics.append(_spearman(p[v], r[v]))
    ics = np.array(ics, dtype=np.float64)
    if len(ics) == 0:
        return {"ic_mean": np.nan, "icir": np.nan, "ic_pos_ratio": np.nan, "n_days": 0}
    std = np.nanstd(ics)
    return {
        "ic_mean": float(np.nanmean(ics)),
        "icir": float(np.nanmean(ics) / std) if std > 0 else np.nan,
        "ic_pos_ratio": float(np.nanmean(ics > 0)),
        "n_days": int(len(ics)),
    }


def daily_long_short(
    pred: np.ndarray, ret: np.ndarray, dates: np.ndarray, q: int = 10, min_stocks: int = 20
) -> dict:
    """逐日 top/bottom 分位多空价差（top − bottom 的真实收益均值）。"""
    tops, bots, spreads = [], [], []
    for d in np.unique(dates):
        m = dates == d
        p, r = pred[m], ret[m]
        v = np.isfinite(r) & np.isfinite(p)
        p, r = p[v], r[v]
        if len(p) < max(q * 2, min_stocks):
            continue
        order = np.argsort(p)
        nb = len(p) // q
        bot = r[order[:nb]].mean()
        top = r[order[-nb:]].mean()
        tops.append(top)
        bots.append(bot)
        spreads.append(top - bot)
    if not spreads:
        return {"long_short": np.nan, "top_mean": np.nan, "bottom_mean": np.nan}
    return {
        "long_short": float(np.mean(spreads)),
        "top_mean": float(np.mean(tops)),
        "bottom_mean": float(np.mean(bots)),
    }


def yearly_report(
    pred: np.ndarray, ret: np.ndarray, dates: np.ndarray, q: int = 10
) -> dict:
    """逐年 rank-IC / ICIR / 多空价差，用于检测风格漂移导致的衰减。"""
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    report = {}
    for y in np.unique(years):
        m = years == y
        ic = daily_rank_ic(pred[m], ret[m], dates[m])
        ls = daily_long_short(pred[m], ret[m], dates[m], q=q)
        report[int(y)] = {**ic, **ls}
    return report
