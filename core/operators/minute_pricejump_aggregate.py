"""
minute_pricejump_aggregate 算子（paper_33：高频价格跳跃峰、岭、谷）
============================================================
与 minute_intraday_aggregate（paper_27 成交量喷发版）平行：逐股票流式
load → 同时点振幅 σ → 跳跃标签 → 双特征划分（局域情绪 × 缺口）→ 价峰/价岭/价谷
→ 日频 reduce → per-stock parquet 缓存（mtime 失效 + 原子 rename）→ slice
到 ctx 日期 → 按 spec.features 投影到 ctx.dataframes[output_dataframe]。

核心定义（paper.md 行号引证）：
  - 振幅（每分钟）= (high - low) / close（paper L163 未明说 close 还是 prev_close，
    取标准实践 close；单股冒烟测试时与 paper 图 2 对照核验）
  - 跳跃 = 同时点过去 std_window 日 1σ 之上（paper L163；σ 窗口 paper 未明说，
    类比 paper_27 的"过去 20 日同时点"）
  - 局域情绪（paper L165）= 前后 1 分钟振幅是否 ≥1σ：高高=高涨/低低=低迷/混合=适中
  - 缺口（paper L167）= 跳跃前后 1 分钟价格区间不重叠：
      gap = high(t-1) < low(t+1)  ∨  high(t+1) < low(t-1)
  - 价峰（paper L171）= 跳跃 ∧ ¬高涨 ∧ ¬缺口
  - 价岭（paper L171）= 跳跃 ∧ ¬低迷 ∧ 缺口
  - 价谷（paper L163）= ¬跳跃（振幅 < 1σ）
  - 日内 first / last 分钟无完整邻居 → peak/ridge 分类为 False（保守），
    但仍计入 jump_count / jump_turnover 等 self-only 列。

契约（与 minute_intraday_aggregate 完全一致）：
  - cache_key       : str        必填，与参数 hash 联合定位缓存目录
  - features        : list[str]  必填，从 _SUPERSET_COLUMNS 中挑列
  - std_window      : int        默认 20
  - std_threshold   : float      默认 1.0
  - output_dataframe: str        默认 "data"

缓存路径：
  INTERMEDIATE_CACHE_DIR / f"{cache_key}__h{params_hash}/{order_book_id}.parquet"
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from core.config import INTERMEDIATE_CACHE_DIR, MINUTE_DATA_DIR

from . import Context, OpRegistry


# Bump 此版本 → params_hash 变 → 全量缓存失效
# v1 (2026-05-25): 初版；含 paper_33 6 代表因子需要的 superset
_FEATURES_SUPERSET_VERSION = "v1"


_SUPERSET_COLUMNS: List[str] = [
    # 峰/岭/谷 时点计数
    "peak_count", "ridge_count", "valley_count",
    # 跳跃自身计数（peak ∪ ridge ∪ 适中跳跃 = 全部 jump 时点）
    "jump_count",
    # 价岭分钟收益（p3）：当日所有 ridge 分钟的 1-min 同日收益之和
    "ridge_return_sum",
    # 价岭日内时间间隔的 5 阶矩 sum（p10 skew 在 spec 端用 m1/m2/m3/n 还原）
    "ridge_interval_n", "ridge_interval_m1", "ridge_interval_m2",
    "ridge_interval_m3", "ridge_interval_m4",
    # 价谷成交量加权均价（p4 与 daily_vwap 做比；p5 与 [low,high,prev_close] 做分位）
    "valley_volume_sum", "valley_vwap",
    # 跳跃自身 X 阶矩 + 下一分钟 Y 阶矩 + XY（p16 价格跳跃成交额相关性）
    "jump_turnover_sum", "jump_turnover_sumsq",
    "jump_next_turnover_sum", "jump_next_turnover_sumsq",
    "jump_xy_sum",
    # 日频价格 / 总量
    "daily_high", "daily_low", "daily_close", "daily_vwap",
    "daily_volume", "daily_turnover",
]


@OpRegistry.register("minute_pricejump_aggregate")
def op_minute_pricejump_aggregate(ctx: Context, step: Dict, fetcher: Any) -> None:
    target_df = step.get("output_dataframe", "data")
    if ctx.has_df(target_df):
        raise ValueError(
            f"minute_pricejump_aggregate: DataFrame {target_df!r} 已存在；"
            "本算子负责创建主表，不要在它之前 fetch 同名 DataFrame"
        )

    cache_key = step["cache_key"]
    std_window = int(step.get("std_window", 20))
    std_threshold = float(step.get("std_threshold", 1.0))
    features = list(step["features"])

    invalid = sorted(set(features) - set(_SUPERSET_COLUMNS))
    if invalid:
        raise ValueError(
            f"minute_pricejump_aggregate: 未知 feature {invalid}; "
            f"可用 superset={_SUPERSET_COLUMNS}"
        )

    if not ctx.universe:
        raise ValueError("minute_pricejump_aggregate: ctx.universe 为空（先确定股票池）")

    cache_dir = _resolve_cache_dir(cache_key, std_window, std_threshold)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"[minute_pricejump_aggregate] cache_dir={cache_dir} | universe={len(ctx.universe)} 只 | "
        f"std_window={std_window} std_threshold={std_threshold}"
    )

    workers = max(1, int(os.environ.get("MINUTE_WORKERS", "8")))
    n_hits = 0
    n_miss = 0
    n_skip = 0
    parts: List[pd.DataFrame] = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_process_stock, ob, std_window, std_threshold, cache_dir): ob
            for ob in ctx.universe
        }
        for fut in as_completed(futures):
            ob = futures[fut]
            try:
                df_stock, status = fut.result()
            except Exception as e:
                logger.warning(f"[minute_pricejump_aggregate] {ob} 失败: {e}")
                n_skip += 1
                continue
            if df_stock is None or df_stock.empty:
                n_skip += 1
                continue
            parts.append(df_stock)
            if status == "hit":
                n_hits += 1
            else:
                n_miss += 1

    logger.info(
        f"[minute_pricejump_aggregate] 完成: cache hit={n_hits} miss={n_miss} skip={n_skip}"
    )

    if not parts:
        raise RuntimeError("minute_pricejump_aggregate: 所有股票都失败/为空")

    full = pd.concat(parts, ignore_index=True)

    if ctx.start_date and ctx.end_date:
        start = pd.to_datetime(ctx.start_date)
        end = pd.to_datetime(ctx.end_date)
        full = full[(full["date"] >= start) & (full["date"] <= end)].copy()

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


def _process_stock(
    ob: str,
    std_window: int,
    std_threshold: float,
    cache_dir: Path,
) -> Tuple[pd.DataFrame, str]:
    src_path = MINUTE_DATA_DIR / f"{ob}.parquet"
    cache_path = cache_dir / f"{ob}.parquet"

    if not src_path.exists():
        return pd.DataFrame(), "miss"

    if cache_path.exists() and cache_path.stat().st_mtime >= src_path.stat().st_mtime:
        df = pd.read_parquet(cache_path)
        return df, "hit"

    df = _compute_one_stock(ob, src_path, std_window, std_threshold)
    if df.empty:
        return df, "miss"

    tmp_path = cache_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path)
    os.replace(tmp_path, cache_path)
    return df, "miss"


def _compute_one_stock(
    ob: str,
    src_path: Path,
    std_window: int,
    std_threshold: float,
) -> pd.DataFrame:
    raw = pd.read_parquet(src_path)

    if "datetime" not in raw.columns:
        raise ValueError(f"{ob}: parquet 缺少 datetime 列；现有 {list(raw.columns)}")

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
    # close == 0 / NaN → 振幅 NaN（warmup 后端到端被 NaN-mask 处理）
    with np.errstate(divide="ignore", invalid="ignore"):
        wide_amp = (wide_high - wide_low) / wide_close.where(wide_close > 0)

    # —— 同时点 σ（每列独立 rolling，shift(1) 防泄漏） ——
    wide_mean = wide_amp.shift(1).rolling(window=std_window, min_periods=std_window).mean()
    wide_std = wide_amp.shift(1).rolling(window=std_window, min_periods=std_window).std()
    wide_threshold = wide_mean + std_threshold * wide_std

    # 跳跃 mask（NaN 比较 → False；warmup 期 wide_threshold NaN → 全 False，最后整行 NaN-mask）
    is_jump = (wide_amp > wide_threshold).to_numpy(dtype=bool)

    # —— 双特征划分 ——
    # 局域情绪：前后 1 分钟振幅是否 ≥1σ（即 is_jump_high_amp）
    # 这里用同一阈值定义"高振幅" = 振幅 ≥ 1σ → 与跳跃判定一致
    is_high_amp = (wide_amp >= wide_threshold).to_numpy(dtype=bool)

    H = wide_high.to_numpy(dtype=float)
    L = wide_low.to_numpy(dtype=float)
    n_rows, n_min = is_jump.shape

    # 邻居 mask：前/后 1 分钟（边界默认 False）
    prev_high_amp = np.zeros_like(is_high_amp)
    prev_high_amp[:, 1:] = is_high_amp[:, :-1]
    next_high_amp = np.zeros_like(is_high_amp)
    next_high_amp[:, :-1] = is_high_amp[:, 1:]

    # 邻居存在性（边界处 prev/next 不存在；用于排除"无完整邻居"的跳跃）
    has_prev = np.zeros_like(is_jump)
    has_prev[:, 1:] = True
    has_next = np.zeros_like(is_jump)
    has_next[:, :-1] = True
    has_both_neighbors = has_prev & has_next

    # 局域情绪（仅当 has_both_neighbors 时才有定义）：
    is_emotion_high = prev_high_amp & next_high_amp & has_both_neighbors      # 高高
    is_emotion_low = (~prev_high_amp) & (~next_high_amp) & has_both_neighbors  # 低低

    # 缺口判定（仅当 has_both_neighbors 时才有定义）：
    # gap = high(t-1) < low(t+1)  OR  high(t+1) < low(t-1)
    H_prev = np.full_like(H, np.nan)
    H_prev[:, 1:] = H[:, :-1]
    L_prev = np.full_like(L, np.nan)
    L_prev[:, 1:] = L[:, :-1]
    H_next = np.full_like(H, np.nan)
    H_next[:, :-1] = H[:, 1:]
    L_next = np.full_like(L, np.nan)
    L_next[:, :-1] = L[:, 1:]
    with np.errstate(invalid="ignore"):
        is_gap = ((H_prev < L_next) | (H_next < L_prev)) & has_both_neighbors

    # 价峰 = 跳跃 ∧ ¬高涨 ∧ ¬缺口（要求双邻居存在）
    is_peak = is_jump & has_both_neighbors & (~is_emotion_high) & (~is_gap)
    # 价岭 = 跳跃 ∧ ¬低迷 ∧ 缺口
    is_ridge = is_jump & has_both_neighbors & (~is_emotion_low) & is_gap
    # 价谷 = ¬跳跃
    is_valley = ~is_jump

    out = pd.DataFrame(index=wide_close.index)

    out["peak_count"] = is_peak.sum(axis=1)
    out["ridge_count"] = is_ridge.sum(axis=1)
    out["valley_count"] = is_valley.sum(axis=1)
    out["jump_count"] = is_jump.sum(axis=1)

    V = wide_vol.to_numpy(dtype=float)
    T = wide_turnover.to_numpy(dtype=float)

    out["valley_volume_sum"] = (V * is_valley).sum(axis=1)
    C_filled = wide_close.fillna(0).to_numpy(dtype=float)
    valley_pv = (C_filled * V * is_valley).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["valley_vwap"] = np.where(out["valley_volume_sum"] > 0, valley_pv / out["valley_volume_sum"], np.nan)

    # —— 价岭日内时间间隔 5 阶矩 ——
    minute_cols = np.asarray(common_cols)
    ridge_moments = _interval_moments_per_row(is_ridge, minute_cols)
    out["ridge_interval_n"] = ridge_moments[:, 0]
    out["ridge_interval_m1"] = ridge_moments[:, 1]
    out["ridge_interval_m2"] = ridge_moments[:, 2]
    out["ridge_interval_m3"] = ridge_moments[:, 3]
    out["ridge_interval_m4"] = ridge_moments[:, 4]

    # —— 跳跃自身 X / 下一分钟 Y / XY（p16 相关性需要） ——
    T_next = np.zeros_like(T)
    T_next[:, :-1] = T[:, 1:]
    out["jump_turnover_sum"] = (T * is_jump).sum(axis=1)
    out["jump_turnover_sumsq"] = ((T ** 2) * is_jump).sum(axis=1)
    out["jump_next_turnover_sum"] = (T_next * is_jump).sum(axis=1)
    out["jump_next_turnover_sumsq"] = ((T_next ** 2) * is_jump).sum(axis=1)
    out["jump_xy_sum"] = (T * T_next * is_jump).sum(axis=1)

    # —— 价岭分钟收益 = 当日 ridge 分钟的 1-min 同日 close return 之和 ——
    C = wide_close.to_numpy(dtype=float)
    returns = np.full_like(C, np.nan)
    returns[:, 1:] = C[:, 1:] / C[:, :-1] - 1.0
    out["ridge_return_sum"] = np.nansum(returns * is_ridge, axis=1)

    # —— 日频价格 / 总量 ——
    out["daily_high"] = wide_high.max(axis=1)
    out["daily_low"] = wide_low.min(axis=1)
    out["daily_close"] = wide_close.ffill(axis=1).iloc[:, -1]
    daily_pv = (C_filled * V).sum(axis=1)
    out["daily_volume"] = V.sum(axis=1)
    out["daily_turnover"] = T.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["daily_vwap"] = np.where(out["daily_volume"] > 0, daily_pv / out["daily_volume"], np.nan)

    # —— warmup 日（前 std_window 日）所有列置 NaN ——
    warmup_n = min(std_window, n_dates)
    if warmup_n > 0:
        out.iloc[:warmup_n, :] = np.nan

    out = out.reset_index()
    out["order_book_id"] = ob
    return out.loc[:, ["order_book_id", "date"] + _SUPERSET_COLUMNS]


def _interval_moments_per_row(label_mat: np.ndarray, minute_cols: np.ndarray) -> np.ndarray:
    """对每行（一天）计算选中分钟之间的间隔（diff sorted minute_of_day）的 5 阶矩 sum。
    返回 shape=(n_rows, 5)，列序: n, m1=Σ, m2=Σ², m3=Σ³, m4=Σ⁴。
    与 minute_intraday_aggregate 的同名函数完全一致。
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
