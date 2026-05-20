"""
YOLO 执行引擎
============================================================
读取 spec yaml → 静态校验 → 按 calculation_steps 顺序调度算子 → 主表
按 factor.column pivot 成 (T, N) 宽表落盘。

ctx 模型：core.operators.Context
spec 校验：core.spec_schema.validate_spec
"""

from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from loguru import logger

warnings.filterwarnings("ignore")

from core.config import RAW_FACTOR_DIR
from core.spec_schema import validate_spec

from .operators import Context, OpRegistry

# 触发算子注册
from .operators import fetch  # noqa: F401
from .operators import compute  # noqa: F401
from .operators import filter as _filter_mod  # noqa: F401
from .operators import rank  # noqa: F401
from .operators import transform  # noqa: F401
from .operators import rolling  # noqa: F401
from .operators import merge  # noqa: F401


# ── 多线程 get_factor 工人函数 ─────────────────────────────
# rqdatac.init() 已在父进程一次性完成；多线程共享同一会话。
# rqdatac.get_factor 是网络 IO 阻塞调用，等待期间释放 GIL，多线程并发是安全且高效的。


def _thread_get_factor(rq, batch_idx, batch, fields, start_date, end_date):
    """单批 fetch；fields 是 list[str]，rqdatac 一次返回多列。
    返回 (batch_idx, df, elapsed_s, err_or_None)
    """
    t0 = time.time()
    try:
        df = rq.get_factor(batch, fields, start_date=start_date, end_date=end_date)
        return batch_idx, df, time.time() - t0, None
    except Exception as e:
        return batch_idx, None, time.time() - t0, str(e)


def _fmt_fields(fields):
    return fields[0] if len(fields) == 1 else f"[{', '.join(fields)}]"


def _fetch_factor_sequential(rq, batches, fields, start_date, end_date):
    n_batches = len(batches)
    label = _fmt_fields(fields)
    logger.info(f"[fetcher] start fields={label} | {n_batches} 批 | sequential")
    t_start = time.time()
    dfs = []
    for i, batch in enumerate(batches, 1):
        _, df, elapsed, err = _thread_get_factor(rq, i, batch, fields, start_date, end_date)
        if err is not None:
            logger.warning(f"[fetcher] {i}/{n_batches} 失败 ({elapsed:.1f}s): {err}")
            continue
        if df is not None and len(df) > 0:
            dfs.append(df)
        logger.info(f"[fetcher] {i}/{n_batches} ✓ {elapsed:.1f}s")
    logger.info(f"[fetcher] done fields={label} in {time.time() - t_start:.1f}s")
    if not dfs:
        raise ValueError(f"get_factor 全部批次失败: fields={label}")
    return pd.concat(dfs)


def _fetch_factor_parallel(rq, batches, fields, start_date, end_date, workers):
    n_batches = len(batches)
    label = _fmt_fields(fields)
    logger.info(f"[fetcher] start fields={label} | {n_batches} 批 × {workers} 线程")
    t_start = time.time()
    dfs = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(_thread_get_factor, rq, i + 1, b, fields, start_date, end_date)
            for i, b in enumerate(batches)
        ]
        for fut in as_completed(futures):
            batch_idx, df, elapsed, err = fut.result()
            done += 1
            if err is not None:
                logger.warning(f"[fetcher] {batch_idx}/{n_batches} 失败 ({elapsed:.1f}s): {err}")
                continue
            if df is not None and len(df) > 0:
                dfs.append(df)
            logger.info(f"[fetcher] {done}/{n_batches} ✓ {elapsed:.1f}s")
    logger.info(f"[fetcher] done fields={label} in {time.time() - t_start:.1f}s")
    if not dfs:
        raise ValueError(f"get_factor 全部批次失败: fields={label}")
    return pd.concat(dfs)


# ── 米筐数据获取层（保持现有 API，算子层调用） ─────────────


class DataFetcher:
    """封装 rqdatac 的数据获取入口"""

    def __init__(self):
        self._rq = None
        self._inited = False

    def _init_rq(self):
        if self._inited:
            return
        try:
            import rqdatac as rq
            self._rq = rq
            try:
                rq.init()
                self._inited = True
            except Exception as e:
                logger.warning(f"rqdatac.init() 失败: {e}")
        except ImportError:
            raise ImportError("使用 YOLO 引擎需要安装 rqdatac")

    def fetch_pit(
        self,
        order_book_ids: List[str],
        fields: List[str],
        start_quarter: str,
        end_quarter: str,
        statements: str = "latest",
    ) -> pd.DataFrame:
        self._init_rq()
        return self._rq.get_pit_financials_ex(
            order_book_ids=order_book_ids,
            fields=fields,
            start_quarter=start_quarter,
            end_quarter=end_quarter,
            statements=statements,
        )

    def get_index_components(self, index_code: str, date: str) -> List[str]:
        self._init_rq()
        return self._rq.index_components(index_code, date=date)

    def get_factor(
        self,
        order_book_ids: List[str],
        fields,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        batch_size: int = 500,
    ):
        """单日 (date) 或区间 (start_date+end_date) 获取因子值。
        fields 可以是 str 或 list[str]——多字段单次返回一个 DataFrame，比顺序拉每个字段快。
        股票数 > batch_size 时自动分批；线程数由环境变量 FETCHER_WORKERS 控制（默认 12）。
        """
        self._init_rq()
        if isinstance(fields, str):
            fields = [fields]

        if date:
            return self._rq.get_factor(order_book_ids, fields, date=date)

        if len(order_book_ids) <= batch_size:
            return self._rq.get_factor(
                order_book_ids, fields, start_date=start_date, end_date=end_date
            )

        n_batches = (len(order_book_ids) + batch_size - 1) // batch_size
        batches = [
            order_book_ids[i : i + batch_size]
            for i in range(0, len(order_book_ids), batch_size)
        ]
        workers = max(1, int(os.environ.get("FETCHER_WORKERS", "12")))
        workers = min(workers, n_batches)

        if workers <= 1:
            return _fetch_factor_sequential(self._rq, batches, fields, start_date, end_date)

        return _fetch_factor_parallel(self._rq, batches, fields, start_date, end_date, workers)

    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        self._init_rq()
        return self._rq.get_trading_dates(start_date, end_date)

    def all_instruments(self, type_: str = "CS") -> pd.DataFrame:
        self._init_rq()
        return self._rq.all_instruments(type=type_)


# ── 股票池构建 ─────────────────────────────────────────────


def build_universe(universe_cfg: dict, trade_date: str, fetcher: DataFetcher) -> List[str]:
    primary = universe_cfg.get("primary_index", "000906.XSHG")
    if primary == "ALL":
        return fetcher.all_instruments(type_="CS")["order_book_id"].tolist()
    return fetcher.get_index_components(primary, trade_date)


# ── 引擎主体 ───────────────────────────────────────────────


class YoloEngine:
    """读 spec yaml，静态校验后按 calculation_steps 顺序执行"""

    def __init__(self, fetcher: Optional[DataFetcher] = None):
        self.fetcher = fetcher or DataFetcher()

    def run(
        self,
        spec_yaml: dict,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        trade_date: Optional[str] = None,
        ctx: Optional[Context] = None,
    ) -> pd.DataFrame:
        # 1. 静态校验（违反任一条立即 raise）
        validate_spec(spec_yaml)

        factor_name = spec_yaml["factor"]["name"]
        factor_column = spec_yaml["factor"]["column"]
        logger.info(f"[engine] 启动 YOLO 执行: {factor_name}")

        # 2. 构建/接管上下文
        if ctx is None:
            ctx = Context(factor_name=factor_name)
            ctx.start_date = start_date
            ctx.end_date = end_date
            ctx.trade_date = trade_date
            if start_date and end_date:
                ctx.start_quarter = f"{start_date[:4]}q1"
                ctx.end_quarter = f"{end_date[:4]}q4"

            # 股票池
            universe_cfg = spec_yaml.get("universe", {})
            pool_date = trade_date or start_date
            if pool_date:
                ctx.universe = build_universe(universe_cfg, pool_date, self.fetcher)
                logger.info(f"[engine] 股票池: {len(ctx.universe)} 只")

        # 3. 顺序执行 steps
        for i, step in enumerate(spec_yaml.get("calculation_steps", []), 1):
            action = step["action"]
            label = f"step#{i} {step.get('name') or action}"
            logger.info(f"[engine] ▶ {label} [action={action}]")

            before = ctx.schema_snapshot()
            op_func = OpRegistry.get(action)
            op_func(ctx, step, self.fetcher)
            ctx.log_schema_diff(before, step_label=f"step#{i}")

        # 4. 主表 → 宽表落盘
        if not ctx.has_df("data"):
            raise RuntimeError("执行完所有 step 后主表 'data' 仍不存在")
        data = ctx.get_df("data")
        if factor_column not in data.columns:
            raise RuntimeError(
                f"factor.column={factor_column!r} 不在主表 'data' 中"
                f"（已有列: {list(data.columns)}）"
            )
        for k in ("order_book_id", "date"):
            if k not in data.columns:
                raise RuntimeError(f"主表 'data' 缺少索引列 {k!r}")

        wide = data.pivot(index="date", columns="order_book_id", values=factor_column)
        wide.index = pd.to_datetime(wide.index)

        RAW_FACTOR_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RAW_FACTOR_DIR / f"{factor_name}.parquet"
        wide.to_parquet(out_path)
        logger.info(
            f"[engine] ✅ 写入 {out_path}, shape={wide.shape}, "
            f"非空={wide.notna().values.sum():,}"
        )
        return wide


# ── 便捷入口 ───────────────────────────────────────────────


def run_factor(
    factor_name: str,
    spec_yaml: Optional[dict] = None,
    **kwargs,
) -> pd.DataFrame:
    if spec_yaml is None:
        spec_path = Path(__file__).parent.parent / "specs" / factor_name / "spec.yaml"
        with open(spec_path, "r", encoding="utf-8") as f:
            spec_yaml = yaml.safe_load(f)
    return YoloEngine().run(spec_yaml, **kwargs)
