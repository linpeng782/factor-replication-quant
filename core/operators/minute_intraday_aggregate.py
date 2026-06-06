"""
minute_intraday_aggregate 算子
============================================================
分钟级因子专用：逐股票流式 load → 同时点 σ → 喷发标签 → 峰/岭/谷分类
→ 日频 reduce（superset 列）→ per-stock parquet 缓存（mtime 失效 + 原子 rename）
→ slice 到 ctx 日期 → 按 spec.features 投影到 ctx.dataframes[output_dataframe]。

复合算子的原因：5457 stocks × 16y × 240 min ≈ 5×10⁹ 行无法物化为 ctx 的单 DataFrame；
拆 fetch+classify+reduce 三步会强迫中间产物落地或全量驻留 ctx，都不可行。

契约：
  - cache_key       : str        必填，与参数 hash 联合定位缓存目录
  - features        : list[str]  必填，从 _SUPERSET_COLUMNS 中挑列（spec 暴露给下游）
  - std_window      : int        默认 20（同时点 σ 窗口，单位：日）
  - std_threshold   : float      默认 1.0（喷发判定阈值，×σ）
  - output_dataframe: str        默认 "data"

缓存路径：
  INTERMEDIATE_CACHE_DIR / f"{cache_key}__h{params_hash}/{order_book_id}.parquet"
其中 params_hash = sha1(json{std_window, std_threshold, _FEATURES_SUPERSET_VERSION})[:10]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from core.operators.minute_engine import MinuteAggregateEngine, MinuteReducer, register_reducer

from . import Context, OpRegistry


# Bump 此版本 → params_hash 变 → 全量缓存失效（用于 superset 列变更）
# v1 → v2 (2026-05-23): 加 4 列 eruption 自身的 X / X² / XY 阶矩，支撑 f17/f18/f19 跟随比例 / 敏感度 / 相关性因子
# v2 → v3 (2026-05-23): 加 ridge_return_sum（f3）+ peakridge_minute_corr_pooled（f20，operator 内置 std_window 日 Pearson）
# v3 → v4 (2026-05-23): **修 vwap unit bug** —— peak/ridge/valley_vwap 改用后复权 close × volume 加权（与 daily_high/low/close 同单位，可跨日比）；加 daily_vwap 同口径
_FEATURES_SUPERSET_VERSION = "v4"


# 算子产出的所有 superset 列（spec.features 必须是其子集）
_SUPERSET_COLUMNS: List[str] = [
    # 三类分钟数
    "peak_count", "ridge_count", "valley_count",
    # 三类成交量 / 成交额聚合
    "peak_volume_sum", "ridge_volume_sum", "valley_volume_sum",
    "peak_turnover_sum", "ridge_turnover_sum", "valley_turnover_sum",
    # 三类 vwap（turnover_sum / volume_sum）
    "peak_vwap", "ridge_vwap", "valley_vwap",
    # 量岭分钟收益（f3）：当日所有 ridge 分钟的 1-min return 之和
    "ridge_return_sum",
    # 峰间隔分布 5 阶矩（pooled 时跨日 rolling sum 后可还原 mean/var/skew/kurt）
    "peak_interval_n", "peak_interval_m1", "peak_interval_m2",
    "peak_interval_m3", "peak_interval_m4",
    # 岭间隔分布 5 阶矩
    "ridge_interval_n", "ridge_interval_m1", "ridge_interval_m2",
    "ridge_interval_m3", "ridge_interval_m4",
    # 喷发自身的 X 阶矩（f17/f18/f19 公用）：X = T[m] where E[m]=True
    "eruption_count",            # N（喷发分钟数 = peak + ridge）
    "eruption_turnover_sum",     # ΣX
    "eruption_turnover_sumsq",   # ΣX²
    # 喷发后下一分钟 Y 阶矩（论文 f17/f18/f19）：Y = T[m+1] where E[m]=True
    "eruption_next_turnover_sum",     # ΣY
    "eruption_next_turnover_sumsq",   # ΣY²
    # XY 交叉项（f18 OLS 斜率、f19 Pearson 相关都需要）
    "eruption_xy_sum",           # ΣXY = Σ T[m]·T[m+1] where E[m]=True
    # 同时点峰岭数相关性（f20）：operator 内置 std_window 日 Pearson over minute_of_day
    # 不同于其他列的"日频原始量 + spec 端 rolling"范式：这一列是 operator 已经做过 std_window 日 pooling 的 corr
    "peakridge_minute_corr_pooled",
    # 日频价格 / 总量（close 是后复权；volume/turnover 是原始量；daily_vwap = sum(close·vol)/sum(vol) 也后复权）
    "daily_high", "daily_low", "daily_close", "daily_vwap",
    "daily_volume", "daily_turnover",
]


# ─────────────────────────── Reducer ───────────────────────────


@register_reducer
class PeakRidgeValleyReducer(MinuteReducer):
    """峰岭谷归约（开源微观结构系列；prv_v3 superset 36 列）。

    同时点 σ(std_window 日) 定喷发 → 峰/岭/谷分类 → 日频 36 列。
    warmup=2×std_window：peakridge_minute_corr_pooled 嵌套两层 rolling（见设计 §7.3）。
    """

    version = _FEATURES_SUPERSET_VERSION
    superset_columns = _SUPERSET_COLUMNS

    def __init__(self, cache_key: str, std_window: int = 20, std_threshold: float = 1.0):
        self.cache_key = cache_key
        self.std_window = std_window
        self.std_threshold = std_threshold
        self.params = {"std_window": std_window, "std_threshold": std_threshold}
        self.warmup = 2 * std_window

    def reduce(self, ob: str, raw: pd.DataFrame) -> pd.DataFrame:
        return _compute_one_stock(ob, raw, self.std_window, self.std_threshold)


@OpRegistry.register("minute_intraday_aggregate")
def op_minute_intraday_aggregate(ctx: Context, step: Dict, fetcher: Any) -> None:
    target_df = step.get("output_dataframe", "data")
    if ctx.has_df(target_df):
        raise ValueError(
            f"minute_intraday_aggregate: DataFrame {target_df!r} 已存在；"
            "本算子负责创建主表，不要在它之前 fetch 同名 DataFrame"
        )
    cache_key = step["cache_key"]
    std_window = int(step.get("std_window", 20))
    std_threshold = float(step.get("std_threshold", 1.0))
    features = list(step["features"])

    invalid = sorted(set(features) - set(_SUPERSET_COLUMNS))
    if invalid:
        raise ValueError(
            f"minute_intraday_aggregate: 未知 feature {invalid}; 可用 superset={_SUPERSET_COLUMNS}"
        )
    if not ctx.universe:
        raise ValueError("minute_intraday_aggregate: ctx.universe 为空（先确定股票池）")

    reducer = PeakRidgeValleyReducer(cache_key, std_window, std_threshold)
    engine = MinuteAggregateEngine(reducer)
    logger.info(
        f"[minute_intraday_aggregate] cache_dir={engine.cache_dir} | universe={len(ctx.universe)} 只 "
        f"| std_window={std_window} std_threshold={std_threshold}"
    )
    engine.refresh_cache(list(ctx.universe))            # ① 增量刷新 superset 缓存

    parts: List[pd.DataFrame] = []                      # ② 消费：读缓存→slice→投影
    for ob in ctx.universe:
        p = engine.cache_dir / f"{ob}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    if not parts:
        raise RuntimeError("minute_intraday_aggregate: 所有股票都无缓存/为空，拒绝产出空主表")

    full = pd.concat(parts, ignore_index=True)
    if ctx.start_date and ctx.end_date:
        start, end = pd.to_datetime(ctx.start_date), pd.to_datetime(ctx.end_date)
        full = full[(full["date"] >= start) & (full["date"] <= end)].copy()
    keep_cols = ["order_book_id", "date"] + features
    full = full.loc[:, keep_cols].sort_values(["order_book_id", "date"]).reset_index(drop=True)
    ctx.set_df(target_df, full)


# ─────────────────────────── 归约数学（搬自原 _compute_one_stock，逐行不变） ───────────────────────────


def _compute_one_stock(
    ob: str,
    raw: pd.DataFrame,
    std_window: int,
    std_threshold: float,
) -> pd.DataFrame:
    """单股 superset 计算。

    入参 raw：**已读时复权**的单股长表（列含 datetime + OHLCV，close 为后复权）。
    数据来源已从"旧 per-stock post 文件"上移到调用方（读 minute/raw 日文件窗口 →
    core.minute_data 复权 → 按股切表），本函数只做 superset 数学（与 v4 旧缓存逐值一致）。
    """
    raw = raw.copy()

    if "datetime" not in raw.columns:
        raise ValueError(f"{ob}: parquet 缺少 datetime 列；现有 {list(raw.columns)}")

    # 时区检查（数据来源是 tz-naive 北京时间）
    tz = raw["datetime"].dt.tz
    if tz is not None and str(tz) not in ("Asia/Shanghai", "Asia/Shanghai+08:00"):
        raise ValueError(f"{ob}: 意外的时区 {tz!r}（期望 tz-naive 或 Asia/Shanghai）")

    raw["date"] = raw["datetime"].dt.normalize()
    # 09:30→0, 09:31→1, ..., 11:30→120, 13:01→211, ..., 15:00→330（中间有 gap，不影响"同时点"语义）
    raw["minute_of_day"] = (
        raw["datetime"].dt.hour * 60 + raw["datetime"].dt.minute - (9 * 60 + 30)
    )

    # 防御：去重（同一 (date, minute_of_day) 取最后一条）
    raw = raw.drop_duplicates(subset=["date", "minute_of_day"], keep="last")

    # —— 宽表化便于"同时点 rolling" + 矢量化标签 ——
    wide_vol = raw.pivot(index="date", columns="minute_of_day", values="volume").sort_index()
    if wide_vol.empty:
        return pd.DataFrame()

    n_dates = len(wide_vol.index)
    if n_dates == 0:
        return pd.DataFrame()

    # 按列对齐其他指标（fillna(0) 用于求和；价格列保留 NaN）
    common_cols = wide_vol.columns
    wide_turnover = (
        raw.pivot(index="date", columns="minute_of_day", values="total_turnover")
        .reindex(index=wide_vol.index, columns=common_cols)
        .fillna(0)
    )
    wide_close = raw.pivot(index="date", columns="minute_of_day", values="close").reindex(
        index=wide_vol.index, columns=common_cols
    )
    wide_high = raw.pivot(index="date", columns="minute_of_day", values="high").reindex(
        index=wide_vol.index, columns=common_cols
    )
    wide_low = raw.pivot(index="date", columns="minute_of_day", values="low").reindex(
        index=wide_vol.index, columns=common_cols
    )

    # —— 同时点 σ（每列独立 rolling，shift(1) 防泄漏） ——
    wide_mean = wide_vol.shift(1).rolling(window=std_window, min_periods=std_window).mean()
    wide_std = wide_vol.shift(1).rolling(window=std_window, min_periods=std_window).std()
    wide_threshold = wide_mean + std_threshold * wide_std

    # 喷发：volume > threshold；NaN 比较 → False（warmup 后会整行 NaN-mask）
    E = (wide_vol > wide_threshold).to_numpy(dtype=bool)

    # 峰：自身喷发 & 前后分钟均不喷发（同一行内的水平邻居）
    prev_E = np.zeros_like(E)
    prev_E[:, 1:] = E[:, :-1]
    next_E = np.zeros_like(E)
    next_E[:, :-1] = E[:, 1:]
    is_peak = E & ~prev_E & ~next_E
    is_ridge = E & ~is_peak
    is_valley = ~E

    V = wide_vol.fillna(0).to_numpy(dtype=float)
    T = wide_turnover.to_numpy(dtype=float)

    out = pd.DataFrame(index=wide_vol.index)

    out["peak_count"] = is_peak.sum(axis=1)
    out["ridge_count"] = is_ridge.sum(axis=1)
    out["valley_count"] = is_valley.sum(axis=1)

    out["peak_volume_sum"] = (V * is_peak).sum(axis=1)
    out["ridge_volume_sum"] = (V * is_ridge).sum(axis=1)
    out["valley_volume_sum"] = (V * is_valley).sum(axis=1)
    out["peak_turnover_sum"] = (T * is_peak).sum(axis=1)
    out["ridge_turnover_sum"] = (T * is_ridge).sum(axis=1)
    out["valley_turnover_sum"] = (T * is_valley).sum(axis=1)

    # vwap 用**后复权 close × volume** 加权（与 daily_high/low/close 单位一致），
    # 不要用 turnover/volume（原始价，单位与 daily_* 不一致，跨日不可比）。
    # _post 数据：close 是后复权，volume/turnover 是原始量。
    C_filled = wide_close.fillna(0).to_numpy(dtype=float)
    peak_pv = (C_filled * V * is_peak).sum(axis=1)
    ridge_pv = (C_filled * V * is_ridge).sum(axis=1)
    valley_pv = (C_filled * V * is_valley).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["peak_vwap"] = np.where(out["peak_volume_sum"] > 0, peak_pv / out["peak_volume_sum"], np.nan)
        out["ridge_vwap"] = np.where(out["ridge_volume_sum"] > 0, ridge_pv / out["ridge_volume_sum"], np.nan)
        out["valley_vwap"] = np.where(out["valley_volume_sum"] > 0, valley_pv / out["valley_volume_sum"], np.nan)

    # —— 峰/岭日内间隔的 5 阶矩 ——
    minute_cols = np.asarray(common_cols)  # 各列对应的 minute_of_day 编号
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

    # —— 喷发自身的 X 阶矩 + 下一分钟 Y 阶矩 + XY（论文 f17/f18/f19）——
    # X = T[m] where E[m]=True；Y = T[m+1] where E[m]=True
    T_next = np.zeros_like(T)
    T_next[:, :-1] = T[:, 1:]
    out["eruption_count"] = E.sum(axis=1)
    out["eruption_turnover_sum"] = (T * E).sum(axis=1)
    out["eruption_turnover_sumsq"] = ((T ** 2) * E).sum(axis=1)
    out["eruption_next_turnover_sum"] = (T_next * E).sum(axis=1)
    out["eruption_next_turnover_sumsq"] = ((T_next ** 2) * E).sum(axis=1)
    out["eruption_xy_sum"] = (T * T_next * E).sum(axis=1)

    # —— ridge_return_sum（f3）：当日所有 ridge 分钟的 1-min 同日收益之和 ——
    # r[m] = close[m]/close[m-1] - 1，跨日不计（每日第一分钟无 prev → 自然 NaN，sum 时跳过）
    C = wide_close.to_numpy(dtype=float)
    returns = np.full_like(C, np.nan)
    returns[:, 1:] = C[:, 1:] / C[:, :-1] - 1.0
    out["ridge_return_sum"] = np.nansum(returns * is_ridge, axis=1)

    # —— peakridge_minute_corr_pooled（f20）：operator 内置 std_window 日 Pearson ——
    # 对每天 t：P_k(t) = sum 过去 std_window 日 minute k 的 peak 计数；R_k(t) 同理
    # 因子 = Pearson(P, R) over k ∈ [0..n_minutes-1]
    peak_int = is_peak.astype(np.int32)
    ridge_int = is_ridge.astype(np.int32)
    peak_pooled = (
        pd.DataFrame(peak_int, index=wide_vol.index, columns=common_cols)
        .rolling(window=std_window, min_periods=std_window)
        .sum()
        .shift(1)
        .to_numpy(dtype=float)
    )
    ridge_pooled = (
        pd.DataFrame(ridge_int, index=wide_vol.index, columns=common_cols)
        .rolling(window=std_window, min_periods=std_window)
        .sum()
        .shift(1)
        .to_numpy(dtype=float)
    )
    # 行向 Pearson：P_k 跨 k 中心化 → ρ = <P̃, R̃> / (||P̃|| · ||R̃||)
    # warmup 日 peak_pooled 整行 NaN → nanmean 触发 "Mean of empty slice"，最终 corr=NaN，无害
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        P_centered = peak_pooled - np.nanmean(peak_pooled, axis=1, keepdims=True)
        R_centered = ridge_pooled - np.nanmean(ridge_pooled, axis=1, keepdims=True)
        num = np.nansum(P_centered * R_centered, axis=1)
        den = np.sqrt(
            np.nansum(P_centered ** 2, axis=1) * np.nansum(R_centered ** 2, axis=1)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(den > 1e-12, num / den, np.nan)
    out["peakridge_minute_corr_pooled"] = corr

    # —— 日频价格/总量（daily_vwap 后复权口径，与 daily_high/low/close 单位一致）——
    out["daily_high"] = wide_high.max(axis=1)
    out["daily_low"] = wide_low.min(axis=1)
    out["daily_close"] = wide_close.ffill(axis=1).iloc[:, -1]
    daily_pv = (C_filled * V).sum(axis=1)
    out["daily_volume"] = V.sum(axis=1)
    out["daily_turnover"] = T.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["daily_vwap"] = np.where(out["daily_volume"] > 0, daily_pv / out["daily_volume"], np.nan)

    # —— warmup 日（前 std_window 日）的全部特征列置 NaN ——
    warmup_n = min(std_window, n_dates)
    if warmup_n > 0:
        out.iloc[:warmup_n, :] = np.nan

    # 收尾：date 出列、注 ob、固定列序
    out = out.reset_index()
    out["order_book_id"] = ob
    return out.loc[:, ["order_book_id", "date"] + _SUPERSET_COLUMNS]


def _interval_moments_per_row(label_mat: np.ndarray, minute_cols: np.ndarray) -> np.ndarray:
    """对每一行（一天）计算选中分钟之间的间隔（diff sorted minute_of_day）的 5 阶矩 sum。
    返回 shape=(n_rows, 5)，列序: n, m1=Σ, m2=Σ², m3=Σ³, m4=Σ⁴。
    """
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
