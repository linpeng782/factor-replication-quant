"""
JumpReducer —— paper_33 高频价格跳跃峰/岭/谷（开源微观结构系列㉝）
============================================================
与 PeakRidgeValleyReducer（paper_27 成交量喷发版）平行的孪生 reducer：把"峰岭谷"
方法论从【成交量喷发】换成【价格跳跃】。同时点振幅 σ → 跳跃标签 → 双特征划分
（局域情绪 × 缺口）→ 价峰/价岭/价谷 → 日频 reduce（superset 列）。

★ 本文件已从"独立引擎 op"（读旧 MINUTE_DATA_DIR per-stock 文件）**迁移为 MinuteReducer**，
  与 tide/dazzle/intraday/smartmoney 同走 MinuteAggregateEngine（读 minute/raw 窗口 →
  读时复权 → 按股切表 → reduce → append-only 缓存）。spec 入口 action 不变（paper_33 18 spec 零改动）。

核心定义（paper.md 行号引证）：
  - 振幅（每分钟）= (high - low) / close（paper L163；取标准实践 close）
  - 跳跃 = 同时点过去 std_window 日 1σ 之上（paper L163；σ 窗口类比 paper_27"过去20日同时点"）
  - 局域情绪（paper L165）= 前后1分钟振幅是否 ≥1σ：高高=高涨/低低=低迷/混合=适中
  - 缺口（paper L167）= 跳跃前后1分钟价格区间不重叠：high(t-1)<low(t+1) ∨ high(t+1)<low(t-1)
  - 价峰（L171）= 跳跃 ∧ ¬高涨 ∧ ¬缺口；价岭（L171）= 跳跃 ∧ ¬低迷 ∧ 缺口；价谷（L163）= ¬跳跃
  - 日内 first/last 无完整邻居 → peak/ridge 保守为 False，但仍计入 jump_* self-only 列。

warmup=2×std_window：peakridge_minute_corr_pooled 嵌套两层 rolling（同 prv，见设计 §7.3）。
"""

from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from core.operators.minute_engine import MinuteReducer, register_reducer


# Bump 此版本 → params_hash 变 → 全量缓存失效
# v1 (2026-05-25): 初版；含 paper_33 6 代表因子需要的 superset
# v2 (2026-05-25): 扩 17 因子全集；加 peak_interval_*、ridge_vwap、peak/ridge_turnover_sum、
#                  peakridge_minute_corr_pooled
# (2026-06-07 迁移为 MinuteReducer：数学逐行不变，输入源由旧 per-stock 文件改为引擎喂入的复权窗口)
_FEATURES_SUPERSET_VERSION = "v2"


_SUPERSET_COLUMNS: List[str] = [
    # 峰/岭/谷 时点计数
    "peak_count", "ridge_count", "valley_count",
    # 跳跃自身计数（peak ∪ ridge ∪ 适中跳跃 = 全部 jump 时点）
    "jump_count",
    # 三类成交量 / 成交额聚合（p13 用 peak/ridge_turnover_sum）
    "peak_turnover_sum", "ridge_turnover_sum",
    "valley_volume_sum",
    # 三类 vwap（p12 = valley_vwap / ridge_vwap；p4 用 valley_vwap）
    "ridge_vwap", "valley_vwap",
    # 价岭分钟收益（p3）：当日所有 ridge 分钟的 1-min 同日收益之和
    "ridge_return_sum",
    # 价峰日内时间间隔 5 阶矩 sum（p6/p7/p8 std/skew/kurt）
    "peak_interval_n", "peak_interval_m1", "peak_interval_m2",
    "peak_interval_m3", "peak_interval_m4",
    # 价岭日内时间间隔 5 阶矩 sum（p9/p10/p11 std/skew/kurt）
    "ridge_interval_n", "ridge_interval_m1", "ridge_interval_m2",
    "ridge_interval_m3", "ridge_interval_m4",
    # 跳跃自身 X 阶矩 + 下一分钟 Y 阶矩 + XY（p14/p15/p16）
    "jump_turnover_sum", "jump_turnover_sumsq",
    "jump_next_turnover_sum", "jump_next_turnover_sumsq",
    "jump_xy_sum",
    # 同时点峰岭数相关性（p17）：reducer 内置 std_window 日 Pearson over minute_of_day
    "peakridge_minute_corr_pooled",
    # 日频价格 / 总量
    "daily_high", "daily_low", "daily_close", "daily_vwap",
    "daily_volume", "daily_turnover",
]


# ─────────────────────────── Reducer ───────────────────────────


@register_reducer("minute_pricejump_aggregate")
class JumpReducer(MinuteReducer):
    """价格跳跃峰岭谷归约（paper_33；pjr superset）。

    同时点振幅 σ(std_window 日) 定跳跃 → 双特征(局域情绪×缺口)划分价峰/价岭/价谷 → 日频 superset。
    warmup=2×std_window（peakridge_minute_corr_pooled 嵌套两层 rolling）。
    spec 入口沿用 action `minute_pricejump_aggregate`（paper_33 的 18 个 spec 零改动）。
    """

    version = _FEATURES_SUPERSET_VERSION
    superset_columns = _SUPERSET_COLUMNS

    def __init__(self, cache_key: str, std_window: int = 20, std_threshold: float = 1.0):
        self.cache_key = cache_key
        self.std_window = std_window
        self.std_threshold = std_threshold
        self.params = {"std_window": std_window, "std_threshold": std_threshold}
        self.warmup = 2 * std_window

    @classmethod
    def from_step(cls, step: Dict) -> "JumpReducer":
        return cls(
            step["cache_key"],
            int(step.get("std_window", 20)),
            float(step.get("std_threshold", 1.0)),
        )

    def reduce(self, ob: str, raw: pd.DataFrame) -> pd.DataFrame:
        return _compute_one_stock(ob, raw, self.std_window, self.std_threshold)


# ─────────────────── 归约数学（搬自原 op 的 _compute_one_stock，逐行不变；仅入参 src_path→raw）───────────────────


def _compute_one_stock(
    ob: str,
    raw: pd.DataFrame,
    std_window: int,
    std_threshold: float,
) -> pd.DataFrame:
    """单股 superset 计算。入参 raw：已读时复权的单股长表（datetime + OHLCV，close 后复权）。"""
    raw = raw.copy()

    if "datetime" not in raw.columns:
        raise ValueError(f"{ob}: 缺少 datetime 列；现有 {list(raw.columns)}")

    tz = raw["datetime"].dt.tz
    if tz is not None and str(tz) not in ("Asia/Shanghai", "Asia/Shanghai+08:00"):
        raise ValueError(f"{ob}: 意外的时区 {tz!r}（期望 tz-naive 或 Asia/Shanghai）")

    raw["date"] = raw["datetime"].dt.normalize()
    raw["minute_of_day"] = (
        raw["datetime"].dt.hour * 60 + raw["datetime"].dt.minute - (9 * 60 + 30)
    )
    raw = raw.drop_duplicates(subset=["date", "minute_of_day"], keep="last")

    wide_close = raw.pivot(index="date", columns="minute_of_day", values="close").sort_index()
    if wide_close.empty:
        return pd.DataFrame()

    common_cols = wide_close.columns
    n_dates = len(wide_close.index)

    wide_high = raw.pivot(index="date", columns="minute_of_day", values="high").reindex(
        index=wide_close.index, columns=common_cols
    )
    wide_low = raw.pivot(index="date", columns="minute_of_day", values="low").reindex(
        index=wide_close.index, columns=common_cols
    )
    wide_vol = raw.pivot(index="date", columns="minute_of_day", values="volume").reindex(
        index=wide_close.index, columns=common_cols
    ).fillna(0)
    wide_turnover = raw.pivot(index="date", columns="minute_of_day", values="total_turnover").reindex(
        index=wide_close.index, columns=common_cols
    ).fillna(0)

    # —— 振幅（每分钟）= (high - low) / close ——
    with np.errstate(divide="ignore", invalid="ignore"):
        wide_amp = (wide_high - wide_low) / wide_close.where(wide_close > 0)

    # —— 同时点 σ（每列独立 rolling，shift(1) 防泄漏） ——
    wide_mean = wide_amp.shift(1).rolling(window=std_window, min_periods=std_window).mean()
    wide_std = wide_amp.shift(1).rolling(window=std_window, min_periods=std_window).std()
    wide_threshold = wide_mean + std_threshold * wide_std

    is_jump = (wide_amp > wide_threshold).to_numpy(dtype=bool)
    is_high_amp = (wide_amp >= wide_threshold).to_numpy(dtype=bool)

    H = wide_high.to_numpy(dtype=float)
    L = wide_low.to_numpy(dtype=float)

    prev_high_amp = np.zeros_like(is_high_amp)
    prev_high_amp[:, 1:] = is_high_amp[:, :-1]
    next_high_amp = np.zeros_like(is_high_amp)
    next_high_amp[:, :-1] = is_high_amp[:, 1:]

    has_prev = np.zeros_like(is_jump)
    has_prev[:, 1:] = True
    has_next = np.zeros_like(is_jump)
    has_next[:, :-1] = True
    has_both_neighbors = has_prev & has_next

    is_emotion_high = prev_high_amp & next_high_amp & has_both_neighbors      # 高高
    is_emotion_low = (~prev_high_amp) & (~next_high_amp) & has_both_neighbors  # 低低

    H_prev = np.full_like(H, np.nan); H_prev[:, 1:] = H[:, :-1]
    L_prev = np.full_like(L, np.nan); L_prev[:, 1:] = L[:, :-1]
    H_next = np.full_like(H, np.nan); H_next[:, :-1] = H[:, 1:]
    L_next = np.full_like(L, np.nan); L_next[:, :-1] = L[:, 1:]
    with np.errstate(invalid="ignore"):
        is_gap = ((H_prev < L_next) | (H_next < L_prev)) & has_both_neighbors

    is_peak = is_jump & has_both_neighbors & (~is_emotion_high) & (~is_gap)
    is_ridge = is_jump & has_both_neighbors & (~is_emotion_low) & is_gap
    is_valley = ~is_jump

    out = pd.DataFrame(index=wide_close.index)
    out["peak_count"] = is_peak.sum(axis=1)
    out["ridge_count"] = is_ridge.sum(axis=1)
    out["valley_count"] = is_valley.sum(axis=1)
    out["jump_count"] = is_jump.sum(axis=1)

    V = wide_vol.to_numpy(dtype=float)
    T = wide_turnover.to_numpy(dtype=float)

    out["peak_turnover_sum"] = (T * is_peak).sum(axis=1)
    out["ridge_turnover_sum"] = (T * is_ridge).sum(axis=1)
    out["valley_volume_sum"] = (V * is_valley).sum(axis=1)
    ridge_volume_sum = (V * is_ridge).sum(axis=1)

    C_filled = wide_close.fillna(0).to_numpy(dtype=float)
    valley_pv = (C_filled * V * is_valley).sum(axis=1)
    ridge_pv = (C_filled * V * is_ridge).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["valley_vwap"] = np.where(out["valley_volume_sum"] > 0, valley_pv / out["valley_volume_sum"], np.nan)
        out["ridge_vwap"] = np.where(ridge_volume_sum > 0, ridge_pv / ridge_volume_sum, np.nan)

    minute_cols = np.asarray(common_cols)
    peak_moments = _interval_moments_per_row(is_peak, minute_cols)
    ridge_moments = _interval_moments_per_row(is_ridge, minute_cols)
    out["peak_interval_n"] = peak_moments[:, 0]
    out["peak_interval_m1"] = peak_moments[:, 1]
    out["peak_interval_m2"] = peak_moments[:, 2]
    out["peak_interval_m3"] = peak_moments[:, 3]
    out["peak_interval_m4"] = peak_moments[:, 4]
    out["ridge_interval_n"] = ridge_moments[:, 0]
    out["ridge_interval_m1"] = ridge_moments[:, 1]
    out["ridge_interval_m2"] = ridge_moments[:, 2]
    out["ridge_interval_m3"] = ridge_moments[:, 3]
    out["ridge_interval_m4"] = ridge_moments[:, 4]

    T_next = np.zeros_like(T)
    T_next[:, :-1] = T[:, 1:]
    out["jump_turnover_sum"] = (T * is_jump).sum(axis=1)
    out["jump_turnover_sumsq"] = ((T ** 2) * is_jump).sum(axis=1)
    out["jump_next_turnover_sum"] = (T_next * is_jump).sum(axis=1)
    out["jump_next_turnover_sumsq"] = ((T_next ** 2) * is_jump).sum(axis=1)
    out["jump_xy_sum"] = (T * T_next * is_jump).sum(axis=1)

    C = wide_close.to_numpy(dtype=float)
    returns = np.full_like(C, np.nan)
    returns[:, 1:] = C[:, 1:] / C[:, :-1] - 1.0
    out["ridge_return_sum"] = np.nansum(returns * is_ridge, axis=1)

    peak_int = is_peak.astype(np.int32)
    ridge_int = is_ridge.astype(np.int32)
    peak_pooled = (
        pd.DataFrame(peak_int, index=wide_close.index, columns=common_cols)
        .rolling(window=std_window, min_periods=std_window).sum().shift(1).to_numpy(dtype=float)
    )
    ridge_pooled = (
        pd.DataFrame(ridge_int, index=wide_close.index, columns=common_cols)
        .rolling(window=std_window, min_periods=std_window).sum().shift(1).to_numpy(dtype=float)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        P_centered = peak_pooled - np.nanmean(peak_pooled, axis=1, keepdims=True)
        R_centered = ridge_pooled - np.nanmean(ridge_pooled, axis=1, keepdims=True)
        num = np.nansum(P_centered * R_centered, axis=1)
        den = np.sqrt(np.nansum(P_centered ** 2, axis=1) * np.nansum(R_centered ** 2, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(den > 1e-12, num / den, np.nan)
    out["peakridge_minute_corr_pooled"] = corr

    out["daily_high"] = wide_high.max(axis=1)
    out["daily_low"] = wide_low.min(axis=1)
    out["daily_close"] = wide_close.ffill(axis=1).iloc[:, -1]
    daily_pv = (C_filled * V).sum(axis=1)
    out["daily_volume"] = V.sum(axis=1)
    out["daily_turnover"] = T.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["daily_vwap"] = np.where(out["daily_volume"] > 0, daily_pv / out["daily_volume"], np.nan)

    # —— warmup 日（前 std_window 日）全部特征列置 NaN ——
    warmup_n = min(std_window, n_dates)
    if warmup_n > 0:
        out.iloc[:warmup_n, :] = np.nan

    out = out.reset_index()
    out["order_book_id"] = ob
    return out.loc[:, ["order_book_id", "date"] + _SUPERSET_COLUMNS]


def _interval_moments_per_row(label_mat: np.ndarray, minute_cols: np.ndarray) -> np.ndarray:
    """对每行（一天）计算选中分钟之间的间隔（diff sorted minute_of_day）的 5 阶矩 sum。
    返回 shape=(n_rows, 5)，列序: n, m1=Σ, m2=Σ², m3=Σ³, m4=Σ⁴。"""
    n_rows = label_mat.shape[0]
    out = np.zeros((n_rows, 5), dtype=np.float64)
    for i in range(n_rows):
        row = label_mat[i]
        if not row.any():
            continue
        sel = minute_cols[row]
        if len(sel) < 2:
            continue
        gaps = np.diff(np.sort(sel)).astype(np.float64)
        out[i, 0] = len(gaps)
        out[i, 1] = gaps.sum()
        out[i, 2] = (gaps ** 2).sum()
        out[i, 3] = (gaps ** 3).sum()
        out[i, 4] = (gaps ** 4).sum()
    return out
