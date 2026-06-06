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

import hashlib
import json
import multiprocessing as mp
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from core.config import INTERMEDIATE_CACHE_DIR, MINUTE_RAW_DIR
from core.minute_data import load_adjusted_minute_window

from . import Context, OpRegistry

# 全量重算时按交易日分块的块大小（#交易日）；内存紧可调小。
_CHUNK_DAYS = int(os.environ.get("MINUTE_CHUNK_DAYS", "250"))


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
            f"minute_intraday_aggregate: 未知 feature {invalid}; "
            f"可用 superset={_SUPERSET_COLUMNS}"
        )

    if not ctx.universe:
        raise ValueError("minute_intraday_aggregate: ctx.universe 为空（先确定股票池）")

    cache_dir = _resolve_cache_dir(cache_key, std_window, std_threshold)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"[minute_intraday_aggregate] cache_dir={cache_dir} | universe={len(ctx.universe)} 只 | "
        f"std_window={std_window} std_threshold={std_threshold}"
    )

    # ① 增量刷新 per-stock superset 缓存（读 minute/raw 日文件窗口 + 读时复权 + append-only）
    _refresh_superset_cache(cache_dir, list(ctx.universe), std_window, std_threshold)

    # ② 消费：读各股缓存 → slice ctx 日期 → 投影 features
    parts: List[pd.DataFrame] = []
    for ob in ctx.universe:
        p = cache_dir / f"{ob}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    if not parts:
        raise RuntimeError("minute_intraday_aggregate: 所有股票都无缓存/为空，拒绝产出空主表")

    full = pd.concat(parts, ignore_index=True)

    # slice 到 ctx 日期范围
    if ctx.start_date and ctx.end_date:
        start = pd.to_datetime(ctx.start_date)
        end = pd.to_datetime(ctx.end_date)
        full = full[(full["date"] >= start) & (full["date"] <= end)].copy()

    # 投影到 spec 声明的 features 子集
    keep_cols = ["order_book_id", "date"] + features
    full = full.loc[:, keep_cols].sort_values(["order_book_id", "date"]).reset_index(drop=True)

    ctx.set_df(target_df, full)


# ─────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────


def _resolve_cache_dir(cache_key: str, std_window: int, std_threshold: float) -> Path:
    params = {
        "std_window": std_window,
        "std_threshold": std_threshold,
        "version": _FEATURES_SUPERSET_VERSION,
    }
    params_hash = hashlib.sha1(
        json.dumps(params, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    return INTERMEDIATE_CACHE_DIR / f"{cache_key}__h{params_hash}"


def _available_raw_dates() -> List[pd.Timestamp]:
    """minute/raw 下现存的全部交易日（升序）。它本身就是数据可用的"交易日历"。"""
    ds = []
    for p in MINUTE_RAW_DIR.glob("*.parquet"):
        try:
            ds.append(pd.Timestamp(p.stem))
        except ValueError:
            continue
    return sorted(ds)


def _cache_last_date(cache_path: Path) -> pd.Timestamp | None:
    if not cache_path.exists():
        return None
    d = pd.read_parquet(cache_path, columns=["date"])
    return pd.Timestamp(d["date"].max()) if len(d) else None


# ── fork COW：大表按股切成子表挂模块全局，worker 按引用取（不 pickle 大表）──
_WORKER_SUBTABLES: Dict[str, pd.DataFrame] = {}
_WORKER_PARAMS: Tuple[int, float] = (20, 1.0)


def _worker_compute(ob: str):
    """worker 任务：从全局子表取该股（已复权）→ 算 superset。返回 (ob, df|None)。"""
    sub = _WORKER_SUBTABLES.get(ob)
    if sub is None or sub.empty:
        return ob, pd.DataFrame()
    try:
        return ob, _compute_one_stock(ob, sub, *_WORKER_PARAMS)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[minute_intraday_aggregate] {ob} compute 失败: {e}")
        return ob, None


def _write_cache_append(cache_path: Path, new_df: pd.DataFrame) -> None:
    """append-only 写：与现有缓存合并 → dedup(date, keep last) → tmp + os.replace。"""
    if new_df.empty:
        return
    if cache_path.exists():
        old = pd.read_parquet(cache_path)
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df
    merged = (
        merged.drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    tmp = cache_path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp)
    os.replace(tmp, cache_path)


def _refresh_superset_cache(
    cache_dir: Path, universe: List[str], std_window: int, std_threshold: float
) -> None:
    """统一"处理时间窗口"引擎（design §6-L2）：

    全量/增量同一路径——只是窗口不同。按交易日分块迭代，块 k 读
    [块首 - W1 交易日, 块末] 日文件，warmup overlap 用真实数据；每股只保留
    (该股缓存 last, raw_max] 的新日 append（幂等 + 内存有界）。

    **W1 = 2×std_window（关键）**：本算子最深列 peakridge_minute_corr_pooled 是嵌套两层
    rolling——某日 corr 依赖前 std_window 天的峰/岭标签，而每天标签又各需 std_window 天
    σ warmup。故块首目标日要正确，须回看 2×std_window 天真实数据（验证 V2.1 实测发现，
    比 design §5 笼统的 W1=std_window 多一层）。
    """
    warmup = 2 * std_window
    raw_dates = _available_raw_dates()
    if not raw_dates:
        raise RuntimeError(f"minute/raw 为空: {MINUTE_RAW_DIR}")
    raw_max = raw_dates[-1]

    last_per_stock = {ob: _cache_last_date(cache_dir / f"{ob}.parquet") for ob in universe}
    cached = {ob: d for ob, d in last_per_stock.items() if d is not None}

    if not cached:
        # 全新构建：全 universe 从 raw 起点 full build
        need_obs = list(universe)
        start_idx = 0
    else:
        # 增量 append-only：前沿 = 已处理到的最新 raw 日 = max(cache_last)。
        # ⚠️ 必须用 max 不是 min——退市股 cache_last 停在其退市年(实测~6%股在2005~2024)，
        # 用 min 会被拖到 2005 误触发全量重建；退市股在新日无数据，load 自然不产出，无害。
        frontier = max(cached.values())
        start_idx = _bisect_after(raw_dates, frontier)
        need_obs = [ob for ob in cached if cached[ob] < raw_max]
        # 无缓存新股：增量模式不建（避免只写尾部的残缺缓存）；要全史须删空缓存后全量重建。
        new_obs = [ob for ob in last_per_stock if last_per_stock[ob] is None]
        if new_obs:
            logger.warning(
                f"[minute_intraday_aggregate] {len(new_obs)} 只无缓存(新股/未建) 增量模式跳过"
            )

    if not need_obs or start_idx >= len(raw_dates):
        logger.info("[minute_intraday_aggregate] 缓存已最新，无需刷新")
        return

    workers = max(1, int(os.environ.get("MINUTE_WORKERS", "8")))
    ctx_fork = mp.get_context("fork")  # COW 共享子表的前提（Mac 默认 spawn 不行）
    logger.info(
        f"[minute_intraday_aggregate] 刷新缓存: {len(need_obs)}/{len(universe)} 股需更新 | "
        f"目标日 {raw_dates[start_idx].date()}~{raw_max.date()} | "
        f"块={_CHUNK_DAYS}交易日 workers={workers}"
    )

    n_written = 0
    for ci in range(start_idx, len(raw_dates), _CHUNK_DAYS):
        tgt_lo = raw_dates[ci]
        tgt_hi = raw_dates[min(ci + _CHUNK_DAYS, len(raw_dates)) - 1]
        load_lo = raw_dates[max(0, ci - warmup)]  # warmup overlap（真实数据，含嵌套 rolling）
        allm = load_adjusted_minute_window(load_lo, tgt_hi, stocks=need_obs)
        if allm.empty:
            continue
        # 按股切子表 → 挂全局（fork 前），worker COW 取用
        global _WORKER_SUBTABLES, _WORKER_PARAMS
        _WORKER_SUBTABLES = {ob: g for ob, g in allm.groupby("order_book_id", sort=False)}
        _WORKER_PARAMS = (std_window, std_threshold)
        del allm
        obs_here = list(_WORKER_SUBTABLES.keys())
        with ctx_fork.Pool(processes=workers) as pool:
            for ob, df in pool.imap_unordered(_worker_compute, obs_here, chunksize=8):
                if df is None or df.empty:
                    continue
                # 只保留本块目标日 + 严格大于该股缓存 last 的新日（幂等）
                keep = (df["date"] >= tgt_lo) & (df["date"] <= tgt_hi)
                lst = last_per_stock.get(ob)
                if lst is not None:
                    keep &= df["date"] > lst
                df = df[keep]
                if not df.empty:
                    _write_cache_append(cache_dir / f"{ob}.parquet", df)
                    n_written += 1
        _WORKER_SUBTABLES = {}
        logger.info(f"  块 {tgt_lo.date()}~{tgt_hi.date()} 完成（累计写 {n_written} 股·块）")
    logger.info(f"[minute_intraday_aggregate] 刷新完成: 写入 {n_written} 股·块")


def _bisect_after(sorted_dates: List[pd.Timestamp], d: pd.Timestamp) -> int:
    """返回第一个 > d 的下标（即从 d 之后开始算新日）。"""
    lo, hi = 0, len(sorted_dates)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_dates[mid] <= d:
            lo = mid + 1
        else:
            hi = mid
    return lo


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
