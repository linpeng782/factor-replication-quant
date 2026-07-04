"""
A 股基本面 PIT 基础数据（dquant 源）全量回填 + 增量日更
============================================================
与 fundamentals.py（rqdatac 源）完全并行隔离，互不干扰。

形态（每个字段一个 parquet，date×stock 宽表）：
  market-data/fundamentals-dquant/<field>.parquet
  index   : DatetimeIndex（交易日）
  columns : order_book_id

dquant get_factor 查询机制：
  - 内部恒以 order_book_ids=None 查全市场，按月并行查询最优
  - 长区间会 500 Internal Server Error，必须按月分片

模式:
  python data_fetching/fundamentals_dquant.py              # 增量日更（默认）
  python data_fetching/fundamentals_dquant.py --full        # 全量 bootstrap（2006-01 ~ 最新）
  python data_fetching/fundamentals_dquant.py --from 20260101 --to 20260131  # 指定区间
  python data_fetching/fundamentals_dquant.py --workers 16  # 调节并行（默认 16）
"""
from __future__ import annotations

import argparse
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dquant  # noqa: F401  (确保 fork 子进程能继承)
from data_fetching.dquant_source import get_trading_days, latest_trading_date
from core.yolo_engine import incremental_append

# ==================== 字段集（与 fundamentals.py 完全一致）====================
FIELDS = [
    "net_profit_mrq_0", "net_profit_mrq_1", "net_profit_mrq_2", "net_profit_mrq_3",
    "net_profit_mrq_4", "net_profit_mrq_5", "net_profit_mrq_6", "net_profit_mrq_7",
    "net_profit_mrq_8",
    "net_profit_ttm_0",
    "total_equity_mrq_0", "total_equity_mrq_1", "total_equity_mrq_4",
    "cash_flow_from_operating_activities_ttm_0",
    "return_on_invested_capital_ttm",
    "market_cap_3", "pb_ratio_lf", "pe_ratio_ttm",
]

FULL_START = "2005-01-04"       # dquant 基本面数据最早日期（2005 已补齐）
NUM_WORKERS = 16                # 按月并行进程数

# ==================== 路径 ====================
OUT_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/market-data/fundamentals-dquant")


def _panel_path(field: str) -> Path:
    return OUT_DIR / f"{field}.parquet"


# ==================== 交易日 / 起点 ====================
def _latest_ready_date() -> str:
    """终点 = 最近已收盘且就绪的交易日（17:00 前退到前一交易日）。"""
    return latest_trading_date()


def _infer_start() -> str:
    """增量起点 = 各字段面板磁盘最大日期 +1 交易日；本地空或字段缺失则 FULL_START。"""
    maxes = []
    for f in FIELDS:
        p = _panel_path(f)
        if p.exists():
            idx = pd.to_datetime(pd.read_parquet(p, columns=[]).index)
            if len(idx):
                maxes.append(idx.max())
    if len(maxes) < len(FIELDS):
        return FULL_START
    from dquant import data as ddata
    next_d = ddata.get_next_trading_date(min(maxes))
    return pd.Timestamp(next_d).strftime("%Y-%m-%d")


# ==================== 按月并行取数 ====================
def _month_list(start: str, end: str) -> list[tuple[str, str]]:
    """生成 [start, end] 内的 (月初, 月末) 日期对列表。"""
    s = pd.Timestamp(start); e = pd.Timestamp(end)
    out = []
    cur = s.replace(day=1)
    while cur <= e:
        me = (cur + pd.offsets.MonthEnd(1)).normalize()
        ms = max(cur, s).strftime("%Y%m%d")
        me_s = min(me, e).strftime("%Y%m%d")
        if pd.Timestamp(ms) <= pd.Timestamp(me_s):
            out.append((ms, me_s))
        cur += pd.offsets.MonthBegin(1)
    return out


def _fetch_month(args: tuple[str, str]) -> pd.DataFrame:
    """子进程：单月全市场 get_factor → long [order_book_id, trade_date, *FIELDS]。

    含 3 次重试 + 指数退避（dquant 服务端偶发 Connection broken / IncompleteRead）。
    """
    import warnings; warnings.filterwarnings("ignore")
    from dquant import data as ddata
    s, e = args
    last_err = None
    for attempt in range(3):
        try:
            df = ddata.get_factor(None, s, e)
            break
        except Exception as exc:
            last_err = exc
            if attempt < 2:
                time.sleep(5 * (attempt + 1))   # 5s, 10s 退避
                continue
            logger.error(f"  {s}~{e} 3 次重试全失败: {type(exc).__name__}: {exc}")
            raise
    if df is None or len(df) == 0:
        return pd.DataFrame()
    keep = ["order_book_id", "trade_date"] + [f for f in FIELDS if f in df.columns]
    df = df[keep].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _fetch_range(start: str, end: str, workers: int) -> pd.DataFrame:
    """按月并行拉取 [start, end] 全市场基本面 → 聚合 long 表。

    单月失败不崩全局，最后报告失败月份（可重跑 --from/--to 补齐）。
    """
    months = _month_list(start, end)
    if not months:
        return pd.DataFrame()
    logger.info(f"  按月并行拉取 {len(months)} 个月 ({months[0][0]}~{months[-1][1]}) | {workers} 进程")
    ctx = mp.get_context("fork")
    parts = []
    failed = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(_fetch_month, m): m for m in months}
        done = 0
        for f in as_completed(futs):
            m = futs[f]
            try:
                df = f.result()
                if not df.empty:
                    parts.append(df)
            except Exception as exc:
                logger.error(f"  {m[0]}~{m[1]} 失败（已重试3次）: {type(exc).__name__}")
                failed.append(m)
            done += 1
            if done % 20 == 0:
                logger.info(f"    进度 {done}/{len(months)} 用时{time.time()-t0:.0f}s")
    if failed:
        logger.warning(f"  ⚠️ {len(failed)} 个月份失败: {[m[0] for m in failed]}，需重跑补齐")
    if not parts:
        return pd.DataFrame()
    long = pd.concat(parts, ignore_index=True)
    logger.info(f"  取数完成：{len(long):,} 行 用时{time.time()-t0:.0f}s")
    return long


# ==================== 逐字段写盘 ====================
def _write_fields(long: pd.DataFrame, full: bool) -> dict:
    """逐字段 unstack → date×stock 宽表 → incremental_append。"""
    stats = {"ok": 0, "skip": 0}
    for field in FIELDS:
        if field not in long.columns:
            logger.warning(f"  字段 {field} 不在返回中，跳过")
            stats["skip"] += 1
            continue
        wide = long.pivot(index="trade_date", columns="order_book_id", values=field)
        wide.index = pd.to_datetime(wide.index)
        path = _panel_path(field)
        last = None
        if not full and path.exists():
            idx = pd.to_datetime(pd.read_parquet(path, columns=[]).index)
            last = idx.max() if len(idx) else None
        incremental_append(path, wide, last, rebuild=full)
        stats["ok"] += 1
    return stats


# ==================== 主流程 ====================
def build(full: bool = False, start_override: str = None, end_override: str = None,
          workers: int = NUM_WORKERS) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    end = end_override or _latest_ready_date()
    if full:
        start = FULL_START
    elif start_override:
        start = start_override
    else:
        start = _infer_start()

    if pd.Timestamp(start) > pd.Timestamp(end):
        logger.success(f"已最新（磁盘到 {start}），无需更新")
        return

    logger.info(
        f"{'全量 bootstrap' if full else '增量'} {start} ~ {end} | "
        f"{len(FIELDS)} 字段 | {workers} 进程"
    )

    long = _fetch_range(start, end, workers)
    if long.empty:
        logger.warning("fetch 无数据（非交易日或数据未就绪）")
        return

    stats = _write_fields(long, full=full)
    logger.success(
        f"✅ 基本面 PIT（dquant）{'bootstrap' if full else '增量'}完成："
        f"{stats['ok']} 字段写入（skip={stats['skip']}）→ 末日 {end}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="基本面 PIT 基础数据（dquant 源）全量回填 + 增量日更"
    )
    ap.add_argument("--full", action="store_true", help="全量 bootstrap（2005-01 ~ 最新）")
    ap.add_argument("--from", dest="from_date", type=str, help="指定起始日期 YYYYMMDD")
    ap.add_argument("--to", dest="to_date", type=str, help="指定结束日期 YYYYMMDD")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS, help=f"并行进程数（默认 {NUM_WORKERS}）")
    args = ap.parse_args()

    logger.info(f"输出目录: {OUT_DIR} | 模式: {'全量 bootstrap' if args.full else '增量日更'}")
    build(
        full=args.full,
        start_override=args.from_date,
        end_override=args.to_date,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
