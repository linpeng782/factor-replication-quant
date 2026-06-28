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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import pandas as pd
from loguru import logger

from core.config import INTERMEDIATE_CACHE_DIR, MINUTE_RAW_DIR
from core.minute_data import load_adjusted_minute_window

from . import Context, OpRegistry

_CHUNK_DAYS = int(os.environ.get("MINUTE_CHUNK_DAYS", "250"))


# ── 首次出现映射（持久化，纯 raw 派生；设计 §13.2 pass2 简化）──────────────
# "每只股在 minute/raw 首次出现的交易日"。pass2 用它替代两个魔法数：
#   · 僵尸股（all_instruments(CS) 里永无 raw 的远古退市标的）不在映射 → 天然过滤（替代旧"近10日"启发式）
#   · 新股/补洞从【真实首现日】回填，无 504 上限 → 深坑老股自愈（旧 504 版留 [IPO, T-504] 永久洞）
# 一次性全量建（首个增量日触发，并行扫 ~数十秒），之后每天只增量扫新日（O(1)）。可随时删文件重建。
_FIRST_APPEARANCE_PATH = MINUTE_RAW_DIR.parent / "minute_first_appearance.parquet"
_SCANNED_SENTINEL = "__SCANNED_THROUGH__"


def _scan_day_obs(path_str: str) -> list:
    """读单个 raw 日文件的 order_book_id 列（单列，便宜）→ 去重列表。ProcessPool worker。"""
    return pd.read_parquet(path_str, columns=["order_book_id"])["order_book_id"].unique().tolist()


def first_appearance_map(raw_dates: List[pd.Timestamp]) -> Dict[str, pd.Timestamp]:
    """加载 / 增量更新 持久化首现映射，返回 {order_book_id: first_raw_date}。

    首建（无映射文件）→ 并行全扫 raw；之后只扫 scanned_through 之后的新日（通常 1~数日，串行）。
    哨兵行 _SCANNED_SENTINEL 记录"已扫到哪天"，区分"扫过但当天无新股"与"还没扫"。
    """
    m: Dict[str, pd.Timestamp] = {}
    scanned: pd.Timestamp | None = None
    if _FIRST_APPEARANCE_PATH.exists():
        df = pd.read_parquet(_FIRST_APPEARANCE_PATH)
        if _SCANNED_SENTINEL in df.index:
            scanned = pd.Timestamp(df.loc[_SCANNED_SENTINEL, "first_date"])
            df = df.drop(index=_SCANNED_SENTINEL)
        m = {ob: pd.Timestamp(d) for ob, d in df["first_date"].items()}

    to_scan = [d for d in raw_dates if scanned is None or d > scanned]
    if not to_scan:
        return m

    first_build = scanned is None
    if first_build:
        workers = max(1, int(os.environ.get("MINUTE_WORKERS", "8")))
        logger.info(
            f"[minute_engine] 首次建立 raw 首现映射（一次性，{len(to_scan)} 日，{workers} 进程并行扫）…"
        )
        paths = [str(MINUTE_RAW_DIR / f"{d.date()}.parquet") for d in to_scan]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            per_day = list(ex.map(_scan_day_obs, paths))   # 保序
    else:
        per_day = [
            _scan_day_obs(str(MINUTE_RAW_DIR / f"{d.date()}.parquet"))
            if (MINUTE_RAW_DIR / f"{d.date()}.parquet").exists() else []
            for d in to_scan
        ]
    for d, obs in zip(to_scan, per_day):           # 按日期升序 → 记录最早首现
        for ob in obs:
            if ob not in m:
                m[ob] = d

    rows = dict(m)
    rows[_SCANNED_SENTINEL] = raw_dates[-1]
    out = pd.DataFrame({"first_date": pd.Series(rows)})
    out.index.name = "order_book_id"
    _FIRST_APPEARANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _FIRST_APPEARANCE_PATH.with_suffix(".parquet.tmp")
    out.to_parquet(tmp)
    os.replace(tmp, _FIRST_APPEARANCE_PATH)         # 原子写
    logger.info(
        f"[minute_engine] 首现映射已更新: {len(m)} 只股, 扫描至 {raw_dates[-1].date()}"
        + ("（一次性全量建完）" if first_build else f"（增量 +{len(to_scan)} 日）")
    )
    return m


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

    def _run_build_pass(
        self,
        need_obs: List[str],
        start_idx: int,
        raw_dates: List[pd.Timestamp],
        last_per_stock: Dict,
        warmup: int,
        workers: int,
        ctx_fork,
        flush_chunks: int,
    ) -> int:
        """通用建库/增量循环（主增量 pass + 新股 pass 共用）。返回写入的股票数。

        last_per_stock[ob] is None  → 不加日期下界过滤（新股从头建）
        last_per_stock[ob] = date   → 只 append 严格大于 date 的新行（增量 append）
        """
        global _WORKER_SUBTABLES, _WORKER_REDUCER
        pending: Dict[str, List[pd.DataFrame]] = {}

        def _flush() -> int:
            for ob, dfs in pending.items():
                self._write_cache_append(
                    self.cache_dir / f"{ob}.parquet", pd.concat(dfs, ignore_index=True)
                )
            cnt = len(pending)
            pending.clear()
            return cnt

        n_written = 0
        chunk_idx = 0
        for ci in range(start_idx, len(raw_dates), _CHUNK_DAYS):
            tgt_lo = raw_dates[ci]
            tgt_hi = raw_dates[min(ci + _CHUNK_DAYS, len(raw_dates)) - 1]
            load_lo = raw_dates[max(0, ci - warmup)]
            allm = load_adjusted_minute_window(load_lo, tgt_hi, stocks=need_obs)
            chunk_idx += 1
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
                        pending.setdefault(ob, []).append(df)
            _WORKER_SUBTABLES = {}
            logger.info(f"  块 {tgt_lo.date()}~{tgt_hi.date()} 完成（待 flush {len(pending)} 股）")
            if chunk_idx % flush_chunks == 0:
                n_written += _flush()
        n_written += _flush()
        return n_written

    def refresh_cache(self, universe: List[str]) -> None:
        """统一"处理时间窗口"引擎（设计 §6-L2 / §7）：全量/增量 + 新股自动建库。

        pass 1  主增量/全量
          · 无缓存 → 全量 start_idx=0（首次建库）
          · 有缓存 → 增量 start_idx = 活跃落后股最早自身前沿之后（**不是**全局 max(cache_last)：
            前沿异构时全局 max 会让 start_idx 跳过所有人 → 落后股永不更新、无法自愈，见下）

        pass 2  新股 / 补洞建库（仅增量模式触发，§13.2 简化）
          · 用持久化"首现映射"识别 universe 中无缓存但有 raw 数据的股
          · 从【真实首现日】回填（无 504 上限）→ 深坑老股自愈
          · 僵尸股（永无 raw）不在映射 → 天然过滤；建完后次日起走正常增量
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        raw_dates = self._available_raw_dates()
        if not raw_dates:
            raise RuntimeError(f"minute/raw 为空: {MINUTE_RAW_DIR}")
        raw_max = raw_dates[-1]

        last_per_stock = {ob: self._cache_last_date(self.cache_dir / f"{ob}.parquet") for ob in universe}
        cached = {ob: d for ob, d in last_per_stock.items() if d is not None}
        new_obs = [ob for ob in last_per_stock if last_per_stock[ob] is None]

        if not cached:
            # 全量模式：universe 全部从头建，new_obs 已包含在内
            need_obs = list(universe)
            start_idx = 0
            new_obs = []   # 全量已覆盖，pass 2 不再重复
        else:
            # 落后股 = 缓存末日 < raw_max。但 cached 里混着早退市、末日停远古的僵尸股
            # （永远落后却无新数据）。start_idx 选取的两个错误极端：
            #   · 全局 max 前沿 → 任意领先子集（如部分刷新/中断遗留）就让 start_idx 越过末日、
            #     pass1 整段跳过 → 其余落后股永不更新、无法自愈（异构前沿锁死 bug）。
            #   · 全局 min 前沿 → 僵尸股把起点拖到远古 → 全量重扫，灾难。
            # 取舍：只 heal「最新交易日 raw 仍出现」的活跃股，start_idx 跟这些活跃落后股的
            # **最早自身前沿**。逐股过滤（_run_build_pass 的 df.date > lst）保证各只只 append 自己缺的日；
            # 健康日更（全股均匀在 T-1）时 min 前沿=T-1 → start_idx=T，行为与旧逻辑等价。无魔法数。
            latest_obs = set(_scan_day_obs(str(MINUTE_RAW_DIR / f"{raw_max.date()}.parquet")))
            need_obs = [ob for ob in cached if cached[ob] < raw_max and ob in latest_obs]
            start_idx = (
                self._bisect_after(raw_dates, min(cached[ob] for ob in need_obs))
                if need_obs else len(raw_dates)
            )

        warmup = self.reducer.warmup
        workers = max(1, int(os.environ.get("MINUTE_WORKERS", "8")))
        ctx_fork = mp.get_context("fork")
        flush_chunks = max(1, int(os.environ.get("MINUTE_FLUSH_CHUNKS", "25")))

        # ── pass 1：主增量 / 全量 ──────────────────────────────────────
        if need_obs and start_idx < len(raw_dates):
            logger.info(
                f"[minute_engine] pass1 {self.reducer.cache_key}: {len(need_obs)}/{len(universe)} 股 | "
                f"目标日 {raw_dates[start_idx].date()}~{raw_max.date()} | "
                f"warmup={warmup} 块={_CHUNK_DAYS} workers={workers}"
            )
            n1 = self._run_build_pass(
                need_obs, start_idx, raw_dates, last_per_stock,
                warmup, workers, ctx_fork, flush_chunks,
            )
            logger.info(f"[minute_engine] pass1 完成: {n1} 股写入")
        else:
            logger.info(f"[minute_engine] {self.reducer.cache_key} 主缓存已最新，pass1 跳过")

        # ── pass 2：新股 / 补洞建库（§13.2 简化：持久化首现映射，去 504/10日 两魔法数）─────
        # · 僵尸股（all_instruments(CS) 里永无 raw 的远古退市标的）不在映射 → 天然过滤
        # · active（真 IPO / 缓存被清的老股 / 退市但有过数据的股）从【真实首现日】回填，
        #   无 504 上限 → 深坑老股完全自愈（旧 504 版会留 [IPO, T-504] 永久洞）。
        # · ns_start 取所有 active 的最早首现日 → 与全量建库对该股的结果逐值一致（warmup 同源）。
        if new_obs:
            fa = first_appearance_map(raw_dates)
            active = {ob: fa[ob] for ob in new_obs if ob in fa}
            n_zombie = len(new_obs) - len(active)
            if not active:
                logger.info(
                    f"[minute_engine] pass2 跳过: {len(new_obs)} 只无缓存股均无 raw 数据（僵尸）"
                )
            else:
                earliest = min(active.values())
                ns_start = max(0, self._bisect_after(raw_dates, earliest) - 1)  # 含首现日本身
                logger.info(
                    f"[minute_engine] pass2 新股/补洞建库: {len(active)} 只"
                    f"（new_obs={len(new_obs)}，僵尸过滤 {n_zombie}）| 最早首现 {earliest.date()} → "
                    f"扫 {raw_dates[ns_start].date()}~{raw_max.date()} 共 {len(raw_dates)-ns_start} 日"
                    f"（从首现回填，无 504 上限）"
                )
                n2 = self._run_build_pass(
                    list(active), ns_start, raw_dates, last_per_stock,
                    warmup, workers, ctx_fork, flush_chunks,
                )
                logger.info(f"[minute_engine] pass2 完成: {n2} 只入库 ✅")
