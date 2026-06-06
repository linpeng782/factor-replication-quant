"""
通用分钟聚合引擎 + Reducer 接口 + 注册表
============================================================
把"分钟→日频"归约拆成两块（见 docs/hf_factor_factory_design.md §4）：
  - MinuteAggregateEngine：**不变的驱动**——窗口读取 / 读时复权 / 按交易日分块 /
    fork-COW 进程池 / append-only per-stock 缓存 / 增量前沿 / warmup-overlap。写一次，全工厂共用。
  - MinuteReducer：**可变的归约口径**——每篇研报只写它，声明 cache_key/列/warmup/params + reduce()。

身份（鲁棒性核心）：superset 缓存目录 = `cache_key + sha1({**params, version})`。
一个 reducer 实例 ↔ 一份 superset 1:1；改 params/version → 新目录，旧的不动（治"陈旧缓存"）。
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import pandas as pd
from loguru import logger

from core.config import INTERMEDIATE_CACHE_DIR, MINUTE_RAW_DIR
from core.minute_data import load_adjusted_minute_window

from . import Context, OpRegistry

_CHUNK_DAYS = int(os.environ.get("MINUTE_CHUNK_DAYS", "250"))


# ─────────────────────────── Reducer 接口 ───────────────────────────


class MinuteReducer(ABC):
    """一种 minute→日频归约口径。子类声明属性 + 实现 reduce() + from_step()。"""

    action: str = ""             # spec 的 action 名（= 该 reducer 的入口；通用 op 据此路由）
    cache_key: str = ""          # 该归约口径 / superset 的命名（= 研报家族）
    version: str = "v1"          # 归约逻辑/列 变更 → bump → 强制重算
    superset_columns: List[str] = []   # 产出的日频特征列（factor 的 features 须 ⊆ 它）
    warmup: int = 0              # 跨日回看（交易日）；嵌套 rolling 要算够（见设计 §7.3）
    granularity: str = "minute"  # half_day | hourly | minute（引擎按需读，预留）
    params: Dict = {}            # 影响归约结果的参数（进 cache hash）

    @abstractmethod
    def reduce(self, ob: str, raw: pd.DataFrame) -> pd.DataFrame:
        """单股【已复权】分钟长表 → 日频行（order_book_id, date + superset_columns）。"""

    @classmethod
    def from_step(cls, step: Dict) -> "MinuteReducer":
        """从 spec step 构建 reducer 实例。子类覆写以解析自己的参数。"""
        return cls(cache_key=step["cache_key"])

    def cache_dir(self) -> Path:
        # ⚠️ 哈希口径保持与旧 _resolve_cache_dir 一致（{**params, version}），以复用既有 golden 缓存。
        payload = {**self.params, "version": self.version}
        h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
        return INTERMEDIATE_CACHE_DIR / f"{self.cache_key}__h{h}"


# 中央注册表：action 名 → reducer 类。register_reducer 同时把【唯一的通用 op】挂到该 action，
# 于是新研报只写 reducer（声明 action+from_step），spec 用该 action，无需任何薄壳 op。
REDUCER_BY_ACTION: Dict[str, type] = {}


def register_reducer(action: str):
    """装饰器：@register_reducer("minute_xxx") —— 注册 reducer 到该 action，并把通用 op 挂上。"""

    def deco(cls: type) -> type:
        if action in REDUCER_BY_ACTION:
            raise ValueError(f"action 撞车: {action!r} 已被 {REDUCER_BY_ACTION[action].__name__} 占用")
        cls.action = action
        REDUCER_BY_ACTION[action] = cls
        OpRegistry.register(action)(op_minute_aggregate)   # 同一个通用 op 挂到此 action
        return cls

    return deco


def op_minute_aggregate(ctx: Context, step: Dict, fetcher) -> None:
    """唯一的分钟聚合算子（所有 reducer 共用）：按 action 路由 reducer → 引擎刷新 → 投影 features。"""
    target_df = step.get("output_dataframe", "data")
    if ctx.has_df(target_df):
        raise ValueError(
            f"minute_aggregate: DataFrame {target_df!r} 已存在；本算子负责创建主表"
        )
    action = step["action"]
    if action not in REDUCER_BY_ACTION:
        raise ValueError(f"minute_aggregate: 未知 action {action!r}; 已注册 {sorted(REDUCER_BY_ACTION)}")
    if not ctx.universe:
        raise ValueError("minute_aggregate: ctx.universe 为空（先确定股票池）")

    reducer = REDUCER_BY_ACTION[action].from_step(step)
    features = list(step["features"])
    invalid = sorted(set(features) - set(reducer.superset_columns))
    if invalid:
        raise ValueError(
            f"minute_aggregate({action}): 未知 feature {invalid}; 可用 superset={reducer.superset_columns}"
        )

    engine = MinuteAggregateEngine(reducer)
    logger.info(
        f"[minute_aggregate] action={action} cache_dir={engine.cache_dir.name} | "
        f"universe={len(ctx.universe)} 只 | warmup={reducer.warmup}"
    )
    engine.refresh_cache(list(ctx.universe))           # ① 增量刷新 superset 缓存

    parts: List[pd.DataFrame] = []                     # ② 消费：读缓存→slice→投影
    for ob in ctx.universe:
        p = engine.cache_dir / f"{ob}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    if not parts:
        raise RuntimeError(f"minute_aggregate({action}): 所有股票都无缓存/为空，拒绝产出空主表")

    full = pd.concat(parts, ignore_index=True)
    if ctx.start_date and ctx.end_date:
        start, end = pd.to_datetime(ctx.start_date), pd.to_datetime(ctx.end_date)
        full = full[(full["date"] >= start) & (full["date"] <= end)].copy()
    keep_cols = ["order_book_id", "date"] + features
    full = full.loc[:, keep_cols].sort_values(["order_book_id", "date"]).reset_index(drop=True)
    ctx.set_df(target_df, full)


# ─────────────────────── fork-COW 进程池 worker ───────────────────────
# 大表按股切子表挂模块全局，worker 按引用取（COW，不 pickle 大表）。fork 前设好全局。
_WORKER_SUBTABLES: Dict[str, pd.DataFrame] = {}
_WORKER_REDUCER: MinuteReducer | None = None


def _worker(ob: str):
    sub = _WORKER_SUBTABLES.get(ob)
    if sub is None or sub.empty:
        return ob, pd.DataFrame()
    try:
        return ob, _WORKER_REDUCER.reduce(ob, sub)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[minute_engine] {ob} reduce 失败: {e}")
        return ob, None


# ───────────────────────────── 引擎 ─────────────────────────────


class MinuteAggregateEngine:
    """通用驱动：给定一个 Reducer，把 minute/raw 增量归约到 per-stock superset 缓存。"""

    def __init__(self, reducer: MinuteReducer):
        self.reducer = reducer
        self.cache_dir = reducer.cache_dir()

    # —— 数据可用交易日历（= minute/raw 现存日文件，离线、无需 rqdatac）——
    @staticmethod
    def _available_raw_dates() -> List[pd.Timestamp]:
        ds = []
        for p in MINUTE_RAW_DIR.glob("*.parquet"):
            try:
                ds.append(pd.Timestamp(p.stem))
            except ValueError:
                continue
        return sorted(ds)

    @staticmethod
    def _cache_last_date(cache_path: Path) -> pd.Timestamp | None:
        if not cache_path.exists():
            return None
        d = pd.read_parquet(cache_path, columns=["date"])
        return pd.Timestamp(d["date"].max()) if len(d) else None

    @staticmethod
    def _bisect_after(sorted_dates: List[pd.Timestamp], d: pd.Timestamp) -> int:
        lo, hi = 0, len(sorted_dates)
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_dates[mid] <= d:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @staticmethod
    def _write_cache_append(cache_path: Path, new_df: pd.DataFrame) -> None:
        """append-only：合并现有缓存 → dedup(date, keep last) → tmp + os.replace。"""
        if new_df.empty:
            return
        if cache_path.exists():
            merged = pd.concat([pd.read_parquet(cache_path), new_df], ignore_index=True)
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

    def refresh_cache(self, universe: List[str]) -> None:
        """统一"处理时间窗口"引擎（设计 §6-L2 / §7）：全量/增量同一路径。

        按交易日分块迭代，块首回看 `reducer.warmup` 交易日真实数据 warmup；
        增量前沿 = max(cache_last)（⚠️ 非 min：退市股早 cache_last 会拖垮 min 误触发全量）；
        无缓存新股增量模式跳过（避免残缺尾部）；每股只 append 严格大于其 cache_last 的新日。
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        raw_dates = self._available_raw_dates()
        if not raw_dates:
            raise RuntimeError(f"minute/raw 为空: {MINUTE_RAW_DIR}")
        raw_max = raw_dates[-1]

        last_per_stock = {ob: self._cache_last_date(self.cache_dir / f"{ob}.parquet") for ob in universe}
        cached = {ob: d for ob, d in last_per_stock.items() if d is not None}

        if not cached:
            need_obs = list(universe)
            start_idx = 0
        else:
            frontier = max(cached.values())
            start_idx = self._bisect_after(raw_dates, frontier)
            need_obs = [ob for ob in cached if cached[ob] < raw_max]
            new_obs = [ob for ob in last_per_stock if last_per_stock[ob] is None]
            if new_obs:
                logger.warning(
                    f"[minute_engine] {len(new_obs)} 只无缓存(新股/未建) 增量模式跳过"
                )

        if not need_obs or start_idx >= len(raw_dates):
            logger.info(f"[minute_engine] {self.reducer.cache_key} 缓存已最新，无需刷新")
            return

        warmup = self.reducer.warmup
        workers = max(1, int(os.environ.get("MINUTE_WORKERS", "8")))
        ctx_fork = mp.get_context("fork")
        logger.info(
            f"[minute_engine] 刷新 {self.reducer.cache_key}: {len(need_obs)}/{len(universe)} 股 | "
            f"目标日 {raw_dates[start_idx].date()}~{raw_max.date()} | "
            f"warmup={warmup} 块={_CHUNK_DAYS} workers={workers}"
        )

        global _WORKER_SUBTABLES, _WORKER_REDUCER
        n_written = 0
        for ci in range(start_idx, len(raw_dates), _CHUNK_DAYS):
            tgt_lo = raw_dates[ci]
            tgt_hi = raw_dates[min(ci + _CHUNK_DAYS, len(raw_dates)) - 1]
            load_lo = raw_dates[max(0, ci - warmup)]  # warmup overlap（真实数据，含嵌套 rolling）
            allm = load_adjusted_minute_window(load_lo, tgt_hi, stocks=need_obs)
            if allm.empty:
                continue
            _WORKER_SUBTABLES = {ob: g for ob, g in allm.groupby("order_book_id", sort=False)}
            _WORKER_REDUCER = self.reducer
            del allm
            obs_here = list(_WORKER_SUBTABLES.keys())
            with ctx_fork.Pool(processes=workers) as pool:
                for ob, df in pool.imap_unordered(_worker, obs_here, chunksize=8):
                    if df is None or df.empty:
                        continue
                    keep = (df["date"] >= tgt_lo) & (df["date"] <= tgt_hi)
                    lst = last_per_stock.get(ob)
                    if lst is not None:
                        keep &= df["date"] > lst
                    df = df[keep]
                    if not df.empty:
                        self._write_cache_append(self.cache_dir / f"{ob}.parquet", df)
                        n_written += 1
            _WORKER_SUBTABLES = {}
            logger.info(f"  块 {tgt_lo.date()}~{tgt_hi.date()} 完成（累计写 {n_written} 股·块）")
        logger.info(f"[minute_engine] {self.reducer.cache_key} 刷新完成: 写入 {n_written} 股·块")
