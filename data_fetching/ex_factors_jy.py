"""
A 股复权因子(dquant jy 源)全量回填 + 增量日更
================================================
T3 设计(fork pool 按日比对去重 → 稀疏表):

  问题: dquant get_adj_factor(trade_date=D) 返回"D 日全市场累计因子快照",
        每天 5500 行; 但 alpha158 的 adjusted_panels.py 需要稀疏序列
        (像 rq ex_cum_factor 那样只在除权日有行 + reindex+ffill)。

  解: fork pool 64 workers, 每 worker 处理一日: 调 dquant 拿当日全市场快照。
      父进程收集结果后,**逐日按时间顺序比对**: 与 prev_factor[code] 不同 →
      落一行 (date, code, ex_cum_factor), 同时更新 prev_factor。

      首日(2005-01-04, 或基准日)无 prev → 全市场落一次快照作基线(因这之前
      可能有除权事件, 无法回溯——rq 数据也面临同样问题, 也是 2005 起点起记)。

  输出:
    per-day-change/<YYYY-MM-DD>.parquet   稀疏变化记录(逐日, 只含当日变化的股)
    stock-ex-factors-jy/<股>.parquet     逐股稀疏宽表
      index: ex_date (DatetimeIndex, name='ex_date')
      列: ex_cum_factor (= adjfactor), float64

  口径选择: jy 主 + ricequant 兜底(覆盖互补, 数值一致到 ~1e-6)。
  T0.3 已证 jy adjfactor = 累计值, 与 rq ex_cum_factor 完全等价, **无需 cumprod**。

模式:
  python ex_factors_jy.py --full            # 全量 2005-01-04~最新
  python ex_factors_jy.py                   # 增量: last+1~最新
  python ex_factors_jy.py --from 20200101 --to 20201231
  python ex_factors_jy.py --workers 100
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

import dquant  # noqa: F401 (fork 子进程继承)
from data_fetching.dquant_source import (
    get_adj_factor_for_day,
    get_cs_codes_for_day,
    get_trading_days,
    latest_trading_date,
)

# ==================== 常量 / 路径 ====================

FULL_START = "2005-01-04"
NUM_WORKERS = 64

OUT_BASE = Path("/nfs/ofs-prediction/peterzhenglinpeng/market-data/daily-dquant")
PER_DAY_DIR = OUT_BASE / "per-day-change"     # 逐日"变化记录"
PER_STOCK_DIR = OUT_BASE / "stock-ex-factors-jy"   # 逐股稀疏宽表


# ==================== Stage A: fork pool 拿全市场快照 → 落 per-day-change ====================

def _snapshot_one_day(date_str: str) -> dict:
    """fork 子进程: 拉 D 日全市场 adjfactor 快照, 返回变化记录(已过滤掉无变化)。

    不做比对(子进程不知道 prev), 直接返回当日快照全量, 由父进程聚合做 diff。
    """
    import time as _t
    t0 = _t.time()
    result = {"date": date_str, "status": "error", "data": None, "elapsed": 0.0, "error": ""}

    code = date_str  # 简单标记; 实际返回 date
    out_path = PER_DAY_DIR / f"{date_str}.parquet"

    # 断点续传
    if out_path.exists():
        try:
            existing = pd.read_parquet(out_path)
            result["status"] = "skip"
            result["data"] = existing  # 直接带回去供 diff
            result["elapsed"] = _t.time() - t0
            return result
        except Exception:
            pass

    try:
        codes = get_cs_codes_for_day(date_str)
        if not codes:
            result["status"] = "empty"
            result["elapsed"] = _t.time() - t0
            return result
        adj = get_adj_factor_for_day(codes, date_str)
        if adj is None or len(adj) == 0:
            result["status"] = "empty"
            result["elapsed"] = _t.time() - t0
            return result
        # 整理成 (date, order_book_id, adjfactor)
        out = pd.DataFrame({
            "date": pd.Timestamp(date_str),
            "order_book_id": adj["order_book_id"].values,
            "adjfactor": adj["adjfactor"].astype(np.float64).values,
        })
        out.to_parquet(out_path, index=False)
        result["status"] = "ok"
        result["data"] = out
        result["elapsed"] = _t.time() - t0
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        result["elapsed"] = _t.time() - t0
    return result


def run_stage_a(start: str, end: str, workers: int = NUM_WORKERS) -> None:
    """Stage A: 按日 fork-pool 拉全市场 adjfactor 快照 → 落 per-day-change(全量)。"""
    logger.info(f"[Stage A] adj factor 快照 {start} ~ {end} ({workers} 进程)")
    PER_DAY_DIR.mkdir(parents=True, exist_ok=True)

    days = get_trading_days(start, end)
    if not days:
        logger.warning("无交易日")
        return
    logger.info(f"共 {len(days)} 个交易日")

    t0 = time.time()
    ok = skip = empty = err = 0
    total_rows = 0
    done = 0

    mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
        futs = {ex.submit(_snapshot_one_day, d): d for d in days}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                logger.error(f"{d} child err: {e}")
                r = {"status": "error"}
            if r["status"] == "ok":
                ok += 1
                if r.get("data") is not None:
                    total_rows += len(r["data"])
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
                eta = (len(days) - done) / speed / 60 if speed else 0
                logger.info(
                    f"[A] {done}/{len(days)} ({done*100/len(days):.1f}%) "
                    f"ok={ok} skip={skip} empty={empty} err={err} "
                    f"speed={speed:.2f}d/s ETA={eta:.1f}min"
                )

    logger.success(f"[Stage A] 完成: ok={ok} skip={skip} empty={empty} err={err} "
                   f"快照总行={total_rows:,}  耗时={time.time()-t0:.0f}s")


# ==================== Stage B: 逐日 diff → 稀疏 per-stock 落盘 ====================

def _write_one_stock_sparse(args) -> tuple[str, str]:
    """fork 子进程: 把某只股票的所有"变化日子"序成稀疏 parquet。

    输入: (code, list_of_changes_df) 历史所有变化子的 DataFrame
    输出: (code, status)  → 'new' or 'error: ...'
    """
    code, long_subset = args
    try:
        s = long_subset.set_index("date").sort_index()
        s = s[["adjfactor"]].rename(columns={"adjfactor": "ex_cum_factor"})
        s.index.name = "ex_date"
        s = s.astype(np.float64)
        # dedup
        s = s[~s.index.duplicated(keep="last")]

        out_path = PER_STOCK_DIR / f"{code}.parquet"
        # 增量 append 模式: 如果存在则 concat + dedup
        if out_path.exists():
            old = pd.read_parquet(out_path)
            old.index = pd.to_datetime(old.index)
            comb = pd.concat([old, s])
            comb = comb[~comb.index.duplicated(keep="last")].sort_index()
        else:
            comb = s

        tmp = out_path.with_suffix(".parquet.tmp")
        comb.to_parquet(tmp)
        os.replace(tmp, out_path)
        return code, "ok"
    except Exception as e:
        return code, f"error: {e}"


def run_stage_b(workers: int = NUM_WORKERS) -> None:
    """Stage B: 读所有 per-day-change → 按 diff 降稀疏化 → 逐股并行写 per-stock 稀疏宽表。

    父进程串行做 diff(逐日按时间序排序后的"累计因子变化检测"), 子进程并行写盘。
    """
    logger.info(f"[Stage B] 逐日 diff 降稀疏 + 按股并行写 ({workers} 进程)")
    PER_STOCK_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(PER_DAY_DIR.glob("*.parquet"))
    if not files:
        logger.error("per-day-change 目录无文件, 先跑 Stage A")
        return

    # 读全部 per-day 快照(全市场每日) → 按日期排序
    logger.info(f"读 {len(files)} 个 per-day 快照...")
    t0 = time.time()
    snapshots = []
    for f in files:
        try:
            snapshots.append(pd.read_parquet(f))
        except Exception as e:
            logger.warning(f"跳过坏文件 {f.name}: {e}")
    if not snapshots:
        logger.error("无可用快照")
        return
    all_snap = pd.concat(snapshots, ignore_index=True)
    all_snap = all_snap.sort_values(["date", "order_book_id"]).reset_index(drop=True)
    logger.info(f"合并快照: {len(all_snap):,} 行, {all_snap['date'].nunique()} 日, "
                f"{all_snap['order_book_id'].nunique()} 股, 耗时 {time.time()-t0:.0f}s")

# 逐日按股 向量化 diff → 落每个股的变化点(含基线)
    # 一次性把整张长表 pivot 成宽表, 用 numpy 算 baseline+变化 mask, stack 收 changes
    logger.info("透视快照为 (date × stock) 宽表后向量化 diff...")
    t1 = time.time()
    all_snap["date"] = pd.to_datetime(all_snap["date"])
    piv = all_snap.pivot(index="date", columns="order_book_id", values="adjfactor")
    # piv 形状 (5215 日, 5511 股), dtype float64, 按日期和股票列排好
    logger.info(f"  透视完成 shape={piv.shape}, 耗时 {time.time()-t1:.0f}s")

    logger.info("向量化 diff (baseline + change)...")
    t2 = time.time()
    vals = piv.to_numpy()                       # (T, S) float64, NaN for missing
    not_nan = ~np.isnan(vals)                   # (T, S) bool

    # 基线 mask: 每列沿 axis=0 第一次非 NaN 位置
    # 实现: not_nan[t] & not_nan[t-1]==False(前一位置是NaN或不存在)
    prev_nan = np.ones_like(not_nan, dtype=bool)
    prev_nan[1:] = ~not_nan[:-1]                # 每列 t 行的"前一位置是 NaN"
    baseline_mask = not_nan & prev_nan           # 当前非NaN且前一是NaN = 首次非NaN

    # 变化 mask: 当前非NaN, 且与 **ffill 前一位置**(即真正的前一非NaN值, 排除当前) 差异 > 1e-9
    # 关键: ff 不能是 piv.ffill() 本身(它当位置有值时填自己), 要 shift(1) 后再 ffill
    piv_prev = piv.shift(1).ffill()             # shift 先往下挪一格(排除当前), 再 ffill
    pv = piv_prev.to_numpy()
    diff = np.abs(vals - pv)                    # 当前 vs 真正前一非NaN值
    change_mask = not_nan & (diff > 1e-9)

    keep_mask = baseline_mask | change_mask     # (T, S) bool
    logger.info(f"  baseline={baseline_mask.sum()}  change={change_mask.sum()}  "
                f"keep={keep_mask.sum()}  ({time.time()-t2:.1f}s)")

    # stack 取 keep 位置的原值
    piv_keep = piv.where(keep_mask)
    keep_long = piv_keep.stack().dropna()        # 新版 stack 会保留 NaN 行, 必须显式 dropna(否则逐股表被全日历 NaN 淹没)
    changes_df = keep_long.reset_index()
    if list(changes_df.columns) == [0, 'date', 'order_book_id']:
        changes_df = changes_df.drop(columns=[0])
    changes_df.columns = ["date", "order_book_id", "adjfactor"]
    logger.info(f"  向量化 diff 完成: {len(changes_df):,} 个事件(基线+变化), "
                f"总耗时={time.time()-t2:.0f}s")

    if len(changes_df) == 0:
        logger.warning("无变化事件, 跳过 Stage B")
        return

    # 按股切分 → fork pool 并行写
    logger.info(f"按股切分并行写盘... ({changes_df['order_book_id'].nunique()} 股)")
    t2 = time.time()
    groups = [(code, g.copy()) for code, g in changes_df.groupby("order_book_id")]
    del changes_df, all_snap, snapshots, keep_long, piv, piv_prev, vals, not_nan

    ok = err = 0
    done = 0
    mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
        futs = {ex.submit(_write_one_stock_sparse, g): g[0] for g in groups}
        for fut in as_completed(futs):
            code, status = fut.result()
            if status == "ok":
                ok += 1
            else:
                err += 1
                logger.warning(f"{code} 写盘失败: {status}")
            done += 1
            if done % 1000 == 0 or done == len(groups):
                el = time.time() - t2
                logger.info(f"[B] {done}/{len(groups)} ({done*100/len(groups):.1f}%) "
                            f"ok={ok} err={err} ({el:.0f}s)")

    logger.success(f"[Stage B] 完成: 写 {ok} 股, err={err}, 耗时={time.time()-t0:.0f}s")


# ==================== 增量起点推断 ====================

def _infer_start(full: bool, from_date: str | None) -> str:
    if full or from_date:
        return from_date or FULL_START
    if not PER_DAY_DIR.exists() or not any(PER_DAY_DIR.glob("*.parquet")):
        logger.info("无 per-day-change 基线, 自动全量")
        return FULL_START
    existing = sorted([f.stem for f in PER_DAY_DIR.glob("*.parquet")])
    last = existing[-1]
    days = get_trading_days(last, latest_trading_date())
    if len(days) <= 1:
        logger.success(f"已最新 (per-day-change 到 {last})")
        return last
    return days[1]


# ==================== 主入口 ====================

def main():
    ap = argparse.ArgumentParser(description="A 股复权因子(dquant jy)全量/增量")
    ap.add_argument("--full", action="store_true", help="全量回填(2005-01-04~最新)")
    ap.add_argument("--from", dest="from_date", default=None, help="起点 YYYY-MM-DD")
    ap.add_argument("--to", dest="to_date", default=None, help="终点 YYYY-MM-DD")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS, help=f"fork 进程数(默认 {NUM_WORKERS})")
    ap.add_argument("--only-stage-a", action="store_true", help="只跑 Stage A(拉快照)")
    ap.add_argument("--only-stage-b", action="store_true", help="只跑 Stage B(diff+写盘)")
    args = ap.parse_args()

    logger.info(f"输出: {OUT_BASE}")
    end = args.to_date or latest_trading_date()
    start = _infer_start(args.full, args.from_date)

    if start > end:
        logger.success(f"已最新 ({start} ≥ {end})")
        if not args.only_stage_b:
            return
    else:
        logger.info(f"区间: {start} ~ {end}")

    if not args.only_stage_b:
        run_stage_a(start, end, args.workers)
    if not args.only_stage_a:
        run_stage_b(args.workers)

    logger.success("✅ ex_factors_jy 完成")


if __name__ == "__main__":
    main()