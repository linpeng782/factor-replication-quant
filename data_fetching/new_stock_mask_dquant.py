"""
新股掩码 new_stock_mask(dquant 源)增量日更
================================================
输出对齐 combo_mask_long.parquet 的 (order_book_id, datetime) 长表，列：
  order_book_id, datetime, is_new_stock(bool)
「新股」定义：上市后 NEW_STOCK_THRESHOLD(252) 个交易日内。

设计（与 raw_ohlcv_dquant.py 同项目风格；100% dquant，不依赖 rqdatac）：
  - 模板 = config.COMBO_MASK_PATH(MASK_BACKEND=dquant → cache_dir_dquant/combo_mask_long.parquet)，
    提供交易日历(cal) + 全股池(stocks)。故运行前需 combo_mask 已由 stock-data-fetching 日更流水线刷新到最新交易日。
  - listed_date 来自 dquant all_instruments(None, D, D) 的 listed_date 列，
    per-stock 常量：一旦某日见过某股即得其 listed_date。断点缓存于 .listed_dates_dquant.parquet(长表)。
  - **增量口径**：只按天拉「缓存已覆盖末日之后 且 ≥ DQUANT_DATA_START(2005) 的交易日」。
    dquant 无 2005 前数据(拉空不写缓存 → 老实现每次白试 ~1200 天)，故加 2005 下限彻底杜绝空拉；
    典型日更只拉 1 天。历史 listed_date 全部走缓存，绝不重拉。
  - 对齐长表(compute_split_dates + build_aligned_long)每次全量重算(1700 万行纯向量化 merge，几秒)，
    不涉及任何数据拉取——只是「组装最终文件」。

模式：
  python -m data_fetching.new_stock_mask_dquant           # 增量日更（默认）
  python -m data_fetching.new_stock_mask_dquant --full    # 重建 listed_date 缓存（罕见，走全历史 all_instruments）
  python -m data_fetching.new_stock_mask_dquant --workers 100
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dquant  # noqa: F401  (确保 fork 子进程继承连接)
import config
from data_fetching.dquant_source import latest_trading_date

# ==================== 常量 / 路径 ====================

NEW_STOCK_THRESHOLD = 252              # 上市后多少交易日内算「新股」
DQUANT_DATA_START = "2005-01-04"       # dquant all_instruments 最早有效日；早于此拉空 → 不再尝试
NUM_WORKERS = 64

COMBO_MASK_PATH = config.COMBO_MASK_PATH        # 模板（日历 + 股池），MASK_BACKEND=dquant → cache_dir_dquant/
OUTPUT_PATH = config.NEW_STOCK_MASK_PATH        # 输出
LISTED_CACHE_PATH = OUTPUT_PATH.parent / ".listed_dates_dquant.parquet"  # listed_date 断点缓存
LOG_DIR = PROJECT_ROOT / "data_fetching" / "logs"


# ==================== combo_mask 模板 ====================

def load_combo_template() -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.Index]:
    """读 combo_mask，返回 (长表[oid,datetime], 交易日历, 股池)。"""
    logger.info(f"加载 combo_mask 模板: {COMBO_MASK_PATH}")
    df = pd.read_parquet(COMBO_MASK_PATH, columns=["order_book_id", "datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    if df.empty:
        raise ValueError("combo_mask 为空")
    cal = pd.DatetimeIndex(sorted(df["datetime"].unique()))
    stocks = pd.Index(sorted(df["order_book_id"].unique()), name="order_book_id")
    logger.info(
        f"  日历 {cal[0].date()} ~ {cal[-1].date()}（{len(cal)} 交易日）| "
        f"股票 {len(stocks)} | 样本 {len(df):,}"
    )
    return df, cal, stocks


# ==================== listed_date 拉取（增量） ====================

def _pull_one_day(date_str: str) -> list[dict]:
    """fork 子进程：单日 all_instruments 取 CS 股 listed_date 长表行。"""
    import pandas as pd
    from dquant import data as ddata

    inst = ddata.all_instruments(None, date_str, date_str)
    if inst is None or inst.empty or "listed_date" not in inst.columns:
        return []
    cs = inst[inst["type"] == "CS"]
    ld = pd.to_datetime(cs.set_index("order_book_id")["listed_date"], errors="coerce")
    return [
        {"trade_date": date_str, "order_book_id": oid, "listed_date": d}
        for oid, d in ld.items()
    ]


def fetch_listed_dates(
    cal: pd.DatetimeIndex, workers: int, full: bool
) -> pd.Series:
    """增量聚合 {order_book_id: listed_date}。

    只拉「缓存未覆盖 且 ≥ DQUANT_DATA_START」的交易日（full=True 时忽略缓存重拉全历史）。
    返回 index=order_book_id 的 listed_date Series。
    """
    LISTED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    agg: dict[str, pd.Timestamp] = {}
    done_dates: set[str] = set()
    if LISTED_CACHE_PATH.exists() and not full:
        cached = pd.read_parquet(LISTED_CACHE_PATH)
        cached["listed_date"] = pd.to_datetime(cached["listed_date"], errors="coerce")
        for oid, d in zip(cached["order_book_id"], cached["listed_date"]):
            if oid not in agg and pd.notna(d):
                agg[oid] = d
        done_dates = set(pd.to_datetime(cached["trade_date"]).dt.strftime("%Y-%m-%d"))
        logger.info(f"缓存命中 {len(agg)} 只；已覆盖 {len(done_dates)} 个交易日")

    floor = pd.Timestamp(DQUANT_DATA_START)
    pending = [
        d for d in cal.strftime("%Y-%m-%d")
        if d not in done_dates and pd.Timestamp(d) >= floor
    ]
    logger.info(f"待拉交易日: {len(pending)}（≥{DQUANT_DATA_START} 且未缓存）")

    if not pending:
        logger.success("listed_date 缓存已最新，无需拉取")
        return pd.Series(agg, name="listed_date")

    t0 = time.time()
    new_rows: list[dict] = []
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futures = {ex.submit(_pull_one_day, d): d for d in pending}
        done = 0
        for fut in as_completed(futures):
            ds = futures[fut]
            try:
                rows = fut.result()
                new_rows.extend(rows)
                for r in rows:
                    oid, d = r["order_book_id"], r["listed_date"]
                    if oid not in agg and pd.notna(d):
                        agg[oid] = d
            except Exception as e:
                logger.warning(f"{ds} 拉取失败: {type(e).__name__}: {e}")
            done += 1
            if done % 200 == 0 or done == len(pending):
                sp = done / (time.time() - t0 + 1e-9)
                logger.info(f"  listed_date 进度 {done}/{len(pending)} ({sp:.1f}d/s)")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if LISTED_CACHE_PATH.exists() and not full:
            new_df = pd.concat([pd.read_parquet(LISTED_CACHE_PATH), new_df], ignore_index=True)
        new_df.to_parquet(LISTED_CACHE_PATH, index=False)
        logger.info(f"缓存写入: 累计 {len(new_df):,} 行 → {LISTED_CACHE_PATH.name}")

    return pd.Series(agg, name="listed_date")


# ==================== split_date / 对齐长表（纯向量化） ====================

def compute_split_dates(
    listed_dates: pd.Series, cal: pd.DatetimeIndex, threshold: int
) -> pd.Series:
    """向量化算 split_date（新股期结束日 = 上市后 threshold 个交易日）。"""
    cal_arr = cal.values
    listed_arr = listed_dates.values
    T = len(cal)

    listed_pos = cal_arr.searchsorted(listed_arr, side="left")
    split_pos = listed_pos + threshold

    # 上市早于日历起点：按 252/365 近似把已过交易日折算掉
    early = listed_arr < cal_arr[0]
    if early.any():
        gap = (cal_arr[0] - listed_arr[early]).astype("timedelta64[D]").astype(int)
        split_pos[early] = np.maximum(threshold - (gap * 252 // 365), 0)
        listed_pos[early] = 0

    invalid = pd.isna(listed_dates).values
    if invalid.any():
        listed_pos[invalid] = 0
        split_pos[invalid] = 0
        logger.warning(f"  {int(invalid.sum())} 只 listed_date 缺失 → 标记为非新股")

    split_arr = cal_arr[np.minimum(split_pos, T - 1)].copy()
    over = split_pos >= T
    if over.any():                       # 新股期尚未结束 → 推到日历末日之后
        split_arr[over] = cal_arr[-1] + np.timedelta64(1, "D")
    return pd.Series(split_arr, index=listed_dates.index, name="split_date")


def build_aligned_long(
    combo_df: pd.DataFrame, listed_dates: pd.Series, split_dates: pd.Series
) -> pd.DataFrame:
    """combo_mask 长表左连 (listed_date, split_date) → is_new_stock。"""
    meta = pd.DataFrame({
        "order_book_id": listed_dates.index,
        "listed_date": listed_dates.values,
        "split_date": split_dates.values,
    })
    m = combo_df.merge(meta, on="order_book_id", how="left")
    is_new = (m["datetime"] >= m["listed_date"]) & (m["datetime"] < m["split_date"])
    m["is_new_stock"] = is_new.fillna(False).astype(bool)
    return m[["order_book_id", "datetime", "is_new_stock"]]


# ==================== 与上一版自身对比（真实增量口径） ====================

def compare_with_prev(df_new: pd.DataFrame, prev_path: Path) -> None:
    if not prev_path.exists():
        logger.info("无上一版，跳过对比")
        return
    df_old = pd.read_parquet(prev_path)
    df_old["datetime"] = pd.to_datetime(df_old["datetime"])
    max_old = df_old["datetime"].max()
    logger.info(f"上一版末日 {max_old.date()} | 新版末日 {df_new['datetime'].max().date()}")

    ov = df_new[df_new["datetime"] <= max_old]
    merged = ov.merge(df_old, on=["order_book_id", "datetime"], suffixes=("_new", "_old"), how="outer")
    diff = (merged["is_new_stock_new"].fillna(False) != merged["is_new_stock_old"].fillna(False)).sum()
    logger.info(f"  重叠区 {len(merged):,} 样本，不一致 {int(diff):,}（应为 0，历史冻结）")

    only = df_new[df_new["datetime"] > max_old]
    if len(only):
        logger.success(
            f"  真实新增: {only['datetime'].min().date()} ~ {only['datetime'].max().date()}"
            f"（{len(only):,} 样本，is_new_stock=True {int(only['is_new_stock'].sum()):,}）"
        )


# ==================== 主流程 ====================

def build(workers: int = NUM_WORKERS, full: bool = False) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"new_stock_mask_dquant_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger.remove()
    fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    logger.add(sys.stdout, level="INFO", format=fmt)
    logger.add(log_file, level="DEBUG", format=fmt)

    logger.info("=" * 60)
    logger.info(f"new_stock_mask 增量日更（dquant）| 模式={'全量重建缓存' if full else '增量'}")
    logger.info(f"输出: {OUTPUT_PATH}")
    logger.info(f"日志: {log_file}")
    logger.info("=" * 60)
    t0 = time.time()

    # 1. combo_mask 模板（含日历末日 = 目标末日）
    combo_df, cal, stocks = load_combo_template()

    # 幂等守门：new_stock 已追上 combo_mask 末日则直接退出（缓存不动）
    if OUTPUT_PATH.exists() and not full:
        cur_max = pd.to_datetime(
            pd.read_parquet(OUTPUT_PATH, columns=["datetime"])["datetime"]
        ).max()
        if cur_max >= cal[-1]:
            logger.success(f"已最新（new_stock 末日 {cur_max.date()} ≥ combo_mask 末日 {cal[-1].date()}），跳过")
            return
        logger.info(f"new_stock 现末日 {cur_max.date()} < combo_mask 末日 {cal[-1].date()} → 更新")

    # 2. 增量拉 listed_date
    logger.info("Step 1/4: 增量拉 listed_date...")
    agg = fetch_listed_dates(cal, workers, full)
    listed_dates = agg.reindex(stocks)
    n_missing = int(listed_dates.isna().sum())
    logger.info(f"  listed_date 有效 {len(stocks) - n_missing}/{len(stocks)}，缺失 {n_missing}")

    # 3. split_date + 对齐长表
    logger.info("Step 2/4: 计算 split_date...")
    split_dates = compute_split_dates(listed_dates, cal, NEW_STOCK_THRESHOLD)
    logger.info("Step 3/4: 对齐 combo_mask 长表...")
    df_long = build_aligned_long(combo_df, listed_dates, split_dates)
    n_true = int(df_long["is_new_stock"].sum())
    logger.info(
        f"  is_new_stock=True {n_true:,} ({n_true / len(df_long) * 100:.4f}%) | "
        f"日期 {df_long['datetime'].min().date()} ~ {df_long['datetime'].max().date()}"
    )

    # 4. 保存（先对比上一版，再覆盖）
    logger.info("Step 4/4: 对比上一版 + 保存...")
    compare_with_prev(df_long, OUTPUT_PATH)
    df_long.to_parquet(OUTPUT_PATH, index=False)
    logger.success(f"✅ 保存 {OUTPUT_PATH} | {len(df_long):,} 行 | 耗时 {(time.time() - t0) / 60:.1f} 分钟")


def main() -> None:
    ap = argparse.ArgumentParser(description="new_stock_mask 增量日更（dquant）")
    ap.add_argument("--full", action="store_true", help="重建 listed_date 缓存（走全历史，罕见）")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS, help=f"并行进程数（默认 {NUM_WORKERS}）")
    args = ap.parse_args()
    build(workers=args.workers, full=args.full)


if __name__ == "__main__":
    main()
