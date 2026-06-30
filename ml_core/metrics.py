"""
ml_core.metrics —— 选股评价指标（合并 ml.evaluate + ml_ht.metrics）
============================================================
numpy 自包含实现（无 scipy 依赖），逐日截面计算排序类指标。
预测值 vs 真实远期收益（连续）。daily_rank_ic / daily_long_short / yearly_report
口径同 ml_ht.metrics；model_ic_panel 提供 ml.evaluate.model_ic 的面板等价入口。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ordinal_rank(a: np.ndarray) -> np.ndarray:
    """平均秩（处理并列）。"""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    a_sorted = a[order]
    i, n = 0, len(a)
    while i < n:
        j = i + 1
        while j < n and a_sorted[j] == a_sorted[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = _ordinal_rank(x); ry = _ordinal_rank(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else np.nan


def daily_rank_ic(pred: np.ndarray, ret: np.ndarray, dates: np.ndarray, min_stocks: int = 10) -> dict:
    """逐日 rank-IC → mean / icir / 正占比 / 天数。"""
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


def daily_long_short(pred: np.ndarray, ret: np.ndarray, dates: np.ndarray, q: int = 10, min_stocks: int = 20) -> dict:
    """逐日 top/bottom 分位多空价差。"""
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
        bots.append(r[order[:nb]].mean())
        tops.append(r[order[-nb:]].mean())
        spreads.append(tops[-1] - bots[-1])
    if not spreads:
        return {"long_short": np.nan, "top_mean": np.nan, "bottom_mean": np.nan}
    return {"long_short": float(np.mean(spreads)), "top_mean": float(np.mean(tops)),
            "bottom_mean": float(np.mean(bots))}


def yearly_report(pred: np.ndarray, ret: np.ndarray, dates: np.ndarray, q: int = 10) -> dict:
    """逐年 rank-IC / ICIR / 多空价差。"""
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    report = {}
    for y in np.unique(years):
        m = years == y
        report[int(y)] = {**daily_rank_ic(pred[m], ret[m], dates[m]),
                          **daily_long_short(pred[m], ret[m], dates[m], q=q)}
    return report


def model_ic_panel(pred_panel: pd.DataFrame, target_panel: pd.DataFrame, min_stocks: int = 30) -> pd.Series:
    """面板入口（对齐 ml.evaluate.model_ic）：逐日截面 Spearman IC(ŷ, 真实) → 日 IC 序列。"""
    ex = target_panel.reindex(index=pred_panel.index, columns=pred_panel.columns)
    ic = {}
    for d in pred_panel.index:
        a, b = pred_panel.loc[d], ex.loc[d]
        m = a.notna() & b.notna()
        if m.sum() >= min_stocks:
            ic[d] = a[m].rank().corr(b[m].rank())
    return pd.Series(ic).sort_index()
