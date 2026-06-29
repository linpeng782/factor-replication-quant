"""
A 股【原始(不复权)】1 分钟 bar 按日分片构建 / 日更 —— dquant 版（rq 的并行后端）
============================================================
本脚本是 data_fetching/minute_ohlcv.py（rqdatac 源）的 dquant 平行实现。
数据源 dquant `get_price(frequency="1m", source="rq")` 走 ClickHouse rq_price，
与 rqdatac 1m 实测【逐值 bit 相同】（同 datetime 约定 09:31~15:00 共 240 bar、同量/额）。

输出 <DATA_ROOT>/market-data/minute-dquant/raw/<YYYY-MM-DD>.parquet（全市场一日一文件）：
  列   : order_book_id, datetime, open, high, low, close, volume, total_turnover
         （与 rq raw 列名/列序/dtype 完全一致：价格 float32，量/额 float64）
  口径 : 原始不复权；后复权在【读时】实时算（core.minute_data，dquant 端用 jy 复权因子）。
  区间 : 2005-01-01 ~ 最新交易日

隔离：写到平行的 minute-dquant/ 树，绝不触碰 rq 的 minute/raw/（防 cron 写冲突 + 可对齐）。
与 config.MINUTE_RAW_DIR(MINUTE_DATA_BACKEND=dquant) 指向同一目录。

「按日抓取」设计（全量 / 增量同一路径）：
  - 逐交易日 D：all_instruments(CS) → get_price(1m, source="rq", D) → 原子写 <D>.parquet。
  - 幂等可续传：已存在的日文件默认跳过（--full 强制重写）；崩溃重跑只补缺日。
  - fork 进程池并行：父进程先 touch dquant（get_trade_dates）→ fork 子进程继承连接，
    不重复初始化（与 stock-data-fetching/build/build_full_*_dquant.py 同款 fork 方案）。

用法：
  python minute_ohlcv_dquant.py                 # 增量日更（补缺日）
  python minute_ohlcv_dquant.py --full          # 全量（2005~今；已存在日文件也重写）
  python minute_ohlcv_dquant.py --start 2024-01-01 --end 2024-01-31   # 指定区间（回填/测试）
  python minute_ohlcv_dquant.py --workers 64    # fork 并行进程数（128 核机建议 48~64）
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger

FULL_START = "2005-01-01"
# rq raw 列序/dtype 的唯一真相（与 data_fetching/minute_ohlcv.py 对齐）
FIELDS = ["open", "high", "low", "close", "volume", "total_turnover"]
PRICE_FIELDS = ["open", "high", "low", "close"]
COLS = ["order_book_id", "datetime"] + FIELDS


def _root() -> Path:
    return Path(
        os.environ.get("FACTOR_REPL_DATA_ROOT", "/nfs/ofs-prediction/peterzhenglinpeng")
    )


def _raw_dir() -> Path:
    """dquant 分钟原始目录（平行于 rq 的 minute/raw/，绝不复用同一目录）。"""
    return _root() / "market-data" / "minute-dquant" / "raw"


# ─────────────────────────── 单交易日 worker（fork 子进程入口）───────────────────────────


def fetch_day_df(day_str: str):
    """拉单交易日全市场 dquant 1m → 规整成 rq raw 同构长表（纯函数，无 IO）。

    返回 DataFrame（列/列序/dtype 与 rq minute/raw 完全一致）；当日无数据返回 None。
    对齐脚本与生产写盘共用此函数，保证"验证口径 == 落盘口径"。
    """
    from dquant import data as ddata

    dd = day_str.replace("-", "")
    inst = ddata.all_instruments(None, day_str, day_str)
    if inst is None or inst.empty:
        return None
    codes = inst[inst["type"] == "CS"]["order_book_id"].tolist()
    if not codes:
        return None
    df = ddata.get_price(
        order_book_ids=codes, start_date=dd, end_date=dd,
        frequency="1m", source="rq",
    )
    if df is None or df.empty:
        return None

    # 规整成 rq raw 同构：amount→total_turnover，丢 deals，列序/dtype 对齐
    df = df.rename(columns={"amount": "total_turnover"})
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{day_str}: dquant 1m 缺列 {missing}（现有 {list(df.columns)}）")
    df = df[COLS].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in PRICE_FIELDS:
        df[c] = df[c].astype("float32")           # 价格 float32（与 rq raw 一致）
    df["volume"] = df["volume"].astype("float64")
    df["total_turnover"] = df["total_turnover"].astype("float64")
    return df.sort_values(["order_book_id", "datetime"]).reset_index(drop=True)


def _fetch_write_day(day_str: str, raw_dir_str: str, full: bool) -> tuple:
    """拉单交易日 → 原子写。返回 (date, status, rows)。

    fork 子进程入口：父进程已 import dquant 并 touch 过连接，子进程继承，不重复 init。
    """
    raw_dir = Path(raw_dir_str)
    out_path = raw_dir / f"{day_str}.parquet"
    if out_path.exists() and not full:
        return (day_str, "skip", 0)

    try:
        df = fetch_day_df(day_str)
    except Exception as e:  # noqa: BLE001
        return (day_str, f"err:{type(e).__name__}:{e}", 0)
    if df is None or df.empty:
        return (day_str, "empty", 0)

    raw_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=str(raw_dir))
    os.close(fd)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)
    return (day_str, "ok", len(df))


# ─────────────────────────── 调度 ───────────────────────────


def _trading_days(start: str, end: str) -> list[str]:
    """dquant 交易日历（SH），返回 'YYYY-MM-DD' 升序列表。父进程调用 → touch dquant 连接。"""
    from dquant import data as ddata

    dates = ddata.get_trade_dates(start.replace("-", ""), end.replace("-", ""), market="SH")
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]


def build(full: bool, start: str | None, end: str | None, workers: int,
          raw_dir: Path | None = None) -> None:
    raw_dir = raw_dir or _raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)

    start = start or FULL_START
    end = end or (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    days = _trading_days(start, end)   # 父进程先触达 dquant（fork 前）
    if not full:
        days = [d for d in days if not (raw_dir / f"{d}.parquet").exists()]
    if not days:
        logger.success(f"已最新，无需更新 | 输出 {raw_dir}")
        return

    logger.info(
        f"{'全量' if full else '增量/区间'} {start}~{end} | 待拉 {len(days)} 交易日 | "
        f"workers={workers} | 输出 {raw_dir}"
    )

    stats = {"ok": 0, "skip": 0, "empty": 0, "err": 0}
    total_rows = 0
    t0 = time.time()
    raw_dir_str = str(raw_dir)
    mp_ctx = multiprocessing.get_context("fork")   # fork：子进程继承父 dquant 连接
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
        futs = {ex.submit(_fetch_write_day, d, raw_dir_str, full): d for d in days}
        for i, fut in enumerate(as_completed(futs), 1):
            day, status, rows = fut.result()
            if status == "ok":
                stats["ok"] += 1
                total_rows += rows
            elif status in ("skip", "empty"):
                stats[status] += 1
            else:
                stats["err"] += 1
                logger.warning(f"{day} 失败: {status}")
            if i % 100 == 0 or i == len(days):
                el = time.time() - t0
                rate = i / el if el else 0
                logger.info(
                    f"  [{i}/{len(days)}] ok={stats['ok']} skip={stats['skip']} "
                    f"空={stats['empty']} err={stats['err']} | {rate:.1f}日/s "
                    f"ETA~{(len(days)-i)/rate/60 if rate else 0:.0f}min"
                )
    logger.success(f"完成：{stats} 行数={total_rows:,} | 末日 {days[-1]} → {raw_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="A股原始(不复权)1分钟bar 按日分片 构建/日更 —— dquant 版"
    )
    ap.add_argument("--full", action="store_true", help="全量重下（默认增量补缺日）")
    ap.add_argument("--start", default=None, help="起始日 YYYY-MM-DD（默认 2005-01-01）")
    ap.add_argument("--end", default=None, help="结束日 YYYY-MM-DD（默认昨天）")
    ap.add_argument("--workers", type=int, default=48, help="fork 并行进程数（默认 48）")
    ap.add_argument("--raw-dir", type=Path, default=None, help="输出目录(测试用)")
    a = ap.parse_args()
    build(full=a.full, start=a.start, end=a.end, workers=a.workers, raw_dir=a.raw_dir)


if __name__ == "__main__":
    main()
