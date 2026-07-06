"""
中信一级行业分类（dquant 源）全量回填 + 增量日更
====================================================
与 industry.py（rqdatac 源）完全并行隔离，互不干扰。

形态（宽表，date×stock，值=行业名称字符串）：
  market-data/industry-dquant/industry_panel_zx_dquant.parquet
  index   : DatetimeIndex（交易日）
  columns : order_book_id

dquant get_industry 查询机制：
  - source="rq" 走 ClickHouse（中信 2019，与米筐一致）
  - api_level=1 只取一级行业
  - 按月并行查询最优（与 fundamentals_dquant.py 同策略）

模式:
  python data_fetching/industry_dquant.py              # 增量日更（默认）
  python data_fetching/industry_dquant.py --full        # 全量 bootstrap（2006-01 ~ 最新）
  python data_fetching/industry_dquant.py --from 20260101 --to 20260131
  python data_fetching/industry_dquant.py --workers 16
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
from data_fetching.dquant_source import latest_trading_date
from core.yolo_engine import incremental_append

# ==================== 路径 / 常量 ====================
OUT_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/market-data/industry-dquant")
PANEL_PATH = OUT_DIR / "industry_panel_zx_dquant.parquet"

FULL_START = "2006-01-04"       # dquant 行业数据最早日期
NUM_WORKERS = 16


# ==================== 交易日 / 起点 ====================
def _latest_ready_date() -> str:
    return latest_trading_date()


def _infer_start() -> str:
    """增量起点 = 面板磁盘最大日期 +1 交易日；本地空则 FULL_START。"""
    if not PANEL_PATH.exists():
        return FULL_START
    idx = pd.to_datetime(pd.read_parquet(PANEL_PATH, columns=[]).index)
    if len(idx) == 0:
        return FULL_START
    from dquant import data as ddata
    next_d = ddata.get_next_trading_date(idx.max())
    return pd.Timestamp(next_d).strftime("%Y-%m-%d")


# ==================== 按月并行取数 ====================
def _month_list(start: str, end: str) -> list[tuple[str, str]]:
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
    """子进程：单月中信一级行业 → long [order_book_id, trade_date, first_industry_name]。"""
    import warnings; warnings.filterwarnings("ignore")
    from dquant import data as ddata
    s, e = args
    df = ddata.get_industry(source="rq", api_level=1, start_date=s, end_date=e)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    # 只保留有 first_industry_name 的行（去掉 chain/sector/style 那些 NaN 行）
    df = df[df["first_industry_name"].notna()].copy()
    df = df[["order_book_id", "trade_date", "first_industry_name"]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _fetch_range(start: str, end: str, workers: int) -> pd.DataFrame:
    months = _month_list(start, end)
    if not months:
        return pd.DataFrame()
    logger.info(f"  按月并行拉取 {len(months)} 个月 ({months[0][0]}~{months[-1][1]}) | {workers} 进程")
    ctx = mp.get_context("fork")
    parts = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(_fetch_month, m): m for m in months}
        done = 0
        for f in as_completed(futs):
            df = f.result()
            if not df.empty:
                parts.append(df)
            done += 1
            if done % 40 == 0:
                logger.info(f"    进度 {done}/{len(months)} 用时{time.time()-t0:.0f}s")
    if not parts:
        return pd.DataFrame()
    long = pd.concat(parts, ignore_index=True)
    logger.info(f"  取数完成：{len(long):,} 行 用时{time.time()-t0:.0f}s")
    return long


# ==================== 写盘 ====================
def _write_panel(long: pd.DataFrame, full: bool) -> None:
    """long → date×stock 宽表 → incremental_append。"""
    long = long.drop_duplicates(subset=["trade_date", "order_book_id"], keep="last")
    wide = long.pivot(index="trade_date", columns="order_book_id", values="first_industry_name")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()

    last = None
    if not full and PANEL_PATH.exists():
        idx = pd.to_datetime(pd.read_parquet(PANEL_PATH, columns=[]).index)
        last = idx.max() if len(idx) else None

    # incremental_append 对字符串面板也适用（列并集纳新股 + dedup + 原子写）
    combined = incremental_append(PANEL_PATH, wide, last, rebuild=full)
    logger.info(
        f"  行业面板写入：{combined.shape} | "
        f"{combined.index.min().date()}~{combined.index.max().date()} | "
        f"行业数={combined.stack().nunique()}"
    )


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
        f"中信一级行业 | {workers} 进程"
    )

    long = _fetch_range(start, end, workers)
    if long.empty:
        logger.warning("fetch 无数据（非交易日或数据未就绪）")
        return

    _write_panel(long, full=full)
    logger.success(f"✅ 行业面板（dquant）{'bootstrap' if full else '增量'}完成 → 末日 {end}")


def main():
    ap = argparse.ArgumentParser(
        description="中信一级行业分类（dquant 源）全量回填 + 增量日更"
    )
    ap.add_argument("--full", action="store_true", help="全量 bootstrap（2006-01 ~ 最新）")
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
