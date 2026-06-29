"""
A 股原始 OHLCV(dquant 源)全量回填 + 增量日更
================================================
两阶段:
  Stage A: 按日取数 → per-day/<YYYY-MM-DD>.parquet (长表)
  Stage B: per-day 聚合 → per-stock/<股>.parquet (逐股宽表)
设计见 docs/alpha158_dquant_migration_progress.md · T2。

输出目录:
  /nfs/ofs-prediction/peterzhenglinpeng/market-data/daily_dquant/
    per-day/                 单日 parquet 长表
    stock-ohlcv-dquant/      逐股 parquet (index=date, 列=open/high/low/close/volume/total_turnover, float32)

口径 (T0 已验证):
  - 价格未复权, source="rq" 与 rq stock-ohlcv bit-exact
  - 股池 = 全 A 当日 type=="CS", 含退市股历史 (T0.1), 含新股 (T0.2)
  - 列: open/high/low/close/volume/total_turnover(不含 limit)

模式:
  python raw_ohlcv_dquant.py              # 增量: last+1 ~ 最新交易日
  python raw_ohlcv_dquant.py --full       # 全量: 2005-01-04 ~ 最新交易日
  python raw_ohlcv_dquant.py --from 20200101 --to 20201231  # 指定区间
  python raw_ohlcv_dquant.py --only-per-day   # 只跑 Stage A
  python raw_ohlcv_dquant.py --only-per-stock # 只跑 Stage B
  python raw_ohlcv_dquant.py --workers 100    # 调节并行(默认 64)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dquant  # noqa: F401  (确保 fork 子进程能继承)
from data_fetching.dquant_source import get_ohlcv_for_day, get_trading_days, latest_trading_date

# ==================== 常量 / 路径 ====================

FULL_START = "2005-01-04"
NUM_WORKERS = 64   # fork 进程数(128 核机器留余量给 IO)

OUT_BASE = Path("/nfs/ofs-prediction/peterzhenglinpeng/market-data/daily_dquant")
PER_DAY_DIR = OUT_BASE / "per-day"
PER_STOCK_DIR = OUT_BASE / "stock-ohlcv-dquant"

OHLCV_COLS = ["open", "high", "low", "close", "volume", "total_turnover"]


# ==================== Stage A: 按日取数 → per-day parquet ====================

def _process_one_day(date_str: str) -> dict:
    """fork 子进程入口: 单日 OHLCV 长表 → 落 per-day parquet。"""
    import time as _t
    t0 = _t.time()
    result = {"date": date_str, "status": "error", "rows": 0, "elapsed": 0.0, "error": ""}

    out_path = PER_DAY_DIR / f"{date_str}.parquet"
    if out_path.exists():
        try:
            existing = pd.read_parquet(out_path, columns=["order_book_id"])
            result["status"] = "skip"
            result["rows"] = len(existing)
            result["elapsed"] = _t.time() - t0
            return result
        except Exception:
            pass  # 文件坏了, 重拉

    try:
        df = get_ohlcv_for_day(date_str)
        if len(df) == 0:
            result["status"] = "empty"
            result["elapsed"] = _t.time() - t0
            return result
        df.to_parquet(out_path, index=False)
        result["status"] = "ok"
        result["rows"] = len(df)
        result["elapsed"] = _t.time() - t0
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        result["elapsed"] = _t.time() - t0
    return result


def run_stage_a(start: str, end: str, workers: int = NUM_WORKERS) -> None:
    """Stage A: 按日 fork-pool 落 per-day parquet。"""
    logger.info(f"[Stage A] 按日取数 {start} ~ {end} ({workers} 进程)")
    PER_DAY_DIR.mkdir(parents=True, exist_ok=True)

    days = get_trading_days(start, end)
    if not days:
        logger.warning(f"无交易日 ({start}~{end})")
        return

    logger.info(f"共 {len(days)} 个交易日")
    t0 = time.time()
    ok = skip = empty = err = 0
    total_rows = 0
    done = 0

    mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
        futs = {ex.submit(_process_one_day, d): d for d in days}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                logger.error(f"{d} 子进程异常: {e}")
                r = {"status": "error", "rows": 0}
            if r["status"] == "ok":
                ok += 1; total_rows += r["rows"]
            elif r["status"] == "skip":
                skip += 1
            elif r["status"] == "empty":
                empty += 1
            else:
                err += 1
                if r.get("error"):
                    logger.warning(f"{d} 失败: {r['error']}")
            done += 1
            if done % 500 == 0 or done == len(days):
                el = time.time() - t0
                speed = done / el if el else 0
                eta = (len(days) - done) / speed / 3600 if speed else 0
                logger.info(
                    f"[A] {done}/{len(days)} ({done*100/len(days):.1f}%) "
                    f"ok={ok} skip={skip} empty={empty} err={err} "
                    f"speed={speed:.2f}d/s ETA={eta:.2f}h"
                )

    logger.success(f"[Stage A] 完成: ok={ok} skip={skip} empty={empty} err={err} "
                   f"行={total_rows:,} 耗时={time.time()-t0:.0f}s")


# ==================== Stage B: per-day → per-stock 转换 ====================

def _read_per_days_lazy(days: list[str]) -> pd.DataFrame:
    """惰性concat per-day parquet。"""
    parts = []
    for d in days:
        p = PER_DAY_DIR / f"{d}.parquet"
        if not p.exists():
            continue
        try:
            parts.append(pd.read_parquet(p))
        except Exception as e:
            logger.warning(f"跳过坏文件 {p.name}: {e}")
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _write_one_stock(args) -> tuple[str, str]:
    """fork 子进程入口: 合并长表中某只股票的 Series → 写 per-stock parquet (append+dedup)。"""
    code, long_subset = args
    try:
        s = long_subset.drop(columns=["order_book_id"]).set_index("date").sort_index()
        s = s.astype(np.float64)  # 与现 rq stock-ohlcv 口径一致(float64, bit-exact 可对齐)
        out_path = PER_STOCK_DIR / f"{code}.parquet"
        if out_path.exists():
            old = pd.read_parquet(out_path)
            old.index = pd.to_datetime(old.index)
            comb = pd.concat([old, s])
            comb = comb[~comb.index.duplicated(keep="last")].sort_index()
            tmp = out_path.with_suffix(".parquet.tmp")
            comb.to_parquet(tmp)
            os.replace(tmp, out_path)
            return code, "updated"
        else:
            tmp = out_path.with_suffix(".parquet.tmp")
            s.to_parquet(tmp)
            os.replace(tmp, out_path)
            return code, "new"
    except Exception as e:
        return code, f"error: {e}"


def run_stage_b(workers: int = NUM_WORKERS) -> None:
    """Stage B: 读全部 per-day → groupby(order_book_id) 并行落 per-stock parquet。"""
    logger.info(f"[Stage B] per-day → per-stock ({workers} 进程)")
    PER_STOCK_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(PER_DAY_DIR.glob("*.parquet"))
    if not files:
        logger.error("per-day 目录无文件, 先跑 Stage A")
        return
    logger.info(f"读 {len(files)} 个 per-day parquet...")
    t0 = time.time()
    long_df = _read_per_days_lazy([f.stem for f in files])
    logger.info(f"合并长表: {len(long_df):,} 行, {long_df['order_book_id'].nunique():,} 股, 耗时 {time.time()-t0:.0f}s")

    # 按 order_book_id 切分
    logger.info("按股切分并行写盘...")
    t1 = time.time()
    groups = [(code, g.copy()) for code, g in long_df.groupby("order_book_id")]
    del long_df

    ok_new = ok_update = err = 0
    done = 0
    mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
        futs = {ex.submit(_write_one_stock, g): g[0] for g in groups}
        for fut in as_completed(futs):
            code, status = fut.result()
            if status == "new":
                ok_new += 1
            elif status == "updated":
                ok_update += 1
            else:
                err += 1
                logger.warning(f"{code} 写盘失败: {status}")
            done += 1
            if done % 1000 == 0 or done == len(groups):
                el = time.time() - t1
                logger.info(f"[B] {done}/{len(groups)} ({done*100/len(groups):.1f}%) "
                            f"new={ok_new} upd={ok_update} err={err} ({el:.0f}s)")

    logger.success(f"[Stage B] 完成: new={ok_new} upd={ok_update} err={err} "
                   f"耗时={time.time()-t0:.0f}s")


# ==================== 增量起点推断 ====================

def _infer_start(full: bool, from_date: str | None) -> str:
    if full or from_date:
        return from_date or FULL_START
    # 增量: per-day 目录最大日期 +1(交易日)
    if not PER_DAY_DIR.exists() or not any(PER_DAY_DIR.glob("*.parquet")):
        logger.info("无 per-day 基线, 自动全量")
        return FULL_START
    existing = sorted([f.stem for f in PER_DAY_DIR.glob("*.parquet")])
    last = existing[-1]
    # 用 dquant 取下一交易日
    days = get_trading_days(last, latest_trading_date())
    if len(days) <= 1:
        logger.success(f"已最新 (per-day 到 {last})")
        return last  # 触发"空区间"
    return days[1]  # 跳过 last 本身


# ==================== 主入口 ====================

def main():
    ap = argparse.ArgumentParser(description="A 股原始 OHLCV(dquant)全量/增量")
    ap.add_argument("--full", action="store_true", help="全量回填(2005-01-04~最新)")
    ap.add_argument("--from", dest="from_date", default=None, help="起点 YYYY-MM-DD(覆盖默认起点)")
    ap.add_argument("--to", dest="to_date", default=None, help="终点 YYYY-MM-DD(默认最新交易日)")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS, help=f"fork 进程数(默认 {NUM_WORKERS})")
    ap.add_argument("--only-per-day", action="store_true", help="只跑 Stage A")
    ap.add_argument("--only-per-stock", action="store_true", help="只跑 Stage B")
    args = ap.parse_args()

    logger.info(f"输出: {OUT_BASE}")
    end = args.to_date or latest_trading_date()
    start = _infer_start(args.full, args.from_date)

    if start > end:
        logger.success(f"已最新 ({start} ≥ {end})")
        if not args.only_per_stock:
            return
    else:
        logger.info(f"区间: {start} ~ {end}")

    if not args.only_per_stock:
        run_stage_a(start, end, args.workers)
    if not args.only_per_day:
        run_stage_b(args.workers)

    logger.success("✅ raw_ohlcv_dquant 完成")


if __name__ == "__main__":
    main()