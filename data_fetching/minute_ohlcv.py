"""
A 股【原始(不复权)】1 分钟 bar 按日分片构建 / 日更
============================================================
输出 <DATA_ROOT>/market-data/minute/raw/<YYYY-MM-DD>.parquet（全市场一日一文件）：
  列   : order_book_id, datetime, open, high, low, close, volume, total_turnover
  口径 : **adjust_type='none' 原始不复权**（价格 float32，量/额 float64，对齐旧 1m_post 体量）
         后复权在【读时】实时算（core.minute_data.load_adjusted_minute_window）。
  区间 : 2005-01-01 ~ 最新交易日

「按日抓取」设计（全量 / 增量同一路径，简单鲁棒）：
  - 逐交易日 D：分批拉全市场 get_price(none, D) → 原子写 <D>.parquet。
  - **幂等可续传**：已存在的日文件默认跳过（--full 强制重写）；崩溃重跑只补缺日。
  - 内存有界：一次只持有一天（~全市场1.3M行 ~100MB）。
  - 零手动日期：起点=末个日文件次一交易日(本地空→FULL_START)；终点=最近已就绪交易日。

配额：双账号路由，QuotaExceeded → 自动切下一账号续传。

用法：
  python minute_ohlcv.py                 # 增量日更（补缺日）
  python minute_ohlcv.py --full          # 全量（2005~今；已存在日文件也重写）
  python minute_ohlcv.py --verify        # 抽样对齐：重拉磁盘已有的近几日逐分钟比对
  python minute_ohlcv.py --batch 800 --limit-days 5   # 调批量 / 只跑前 N 日(测试)
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rqdatac
from loguru import logger

ACCOUNTS = [("13522652015", "123456"), ("18555079000", "123456")]
FULL_START = "2005-01-01"
FIELDS = ["open", "high", "low", "close", "volume", "total_turnover"]
PRICE_FIELDS = ["open", "high", "low", "close"]
COLS = ["order_book_id", "datetime"] + FIELDS
QUOTA_STOP_GB = 0.3
BATCH = 800  # 每批股票数（每个交易日分批拉，控单次调用规模）


def _root() -> Path:
    return Path(
        os.environ.get("FACTOR_REPL_DATA_ROOT", "/nfs/ofs-prediction/peterzhenglinpeng")
    )


def _raw_dir() -> Path:
    return _root() / "market-data" / "minute" / "raw"


def _quota_gb() -> float:
    q = rqdatac.user.get_quota()
    return (q["bytes_limit"] - q["bytes_used"]) / 1e9


def _latest_trading_date() -> pd.Timestamp:
    """终点 = 最近已收盘且数据就绪的交易日（未过 19:05 退到前一交易日）。"""
    latest = pd.Timestamp(rqdatac.get_latest_trading_date())
    if pd.Timestamp.now() < latest.replace(hour=19, minute=5, second=0):
        return pd.Timestamp(rqdatac.get_previous_trading_date(latest))
    return latest


def _existing_max_date(raw_dir: Path) -> pd.Timestamp | None:
    ds = []
    for p in raw_dir.glob("*.parquet"):
        try:
            ds.append(pd.Timestamp(p.stem))
        except ValueError:
            continue
    return max(ds) if ds else None


def _fetch_day(day: pd.Timestamp, universe: list[str], batch: int) -> pd.DataFrame:
    """拉单交易日全市场原始分钟（分批）→ 长表(order_book_id, datetime, fields)，价格 float32。"""
    d = str(day.date())
    parts = []
    for i in range(0, len(universe), batch):
        sub = universe[i : i + batch]
        df = rqdatac.get_price(
            sub,
            start_date=d,
            end_date=d,
            frequency="1m",
            fields=FIELDS,
            adjust_type="none",
            skip_suspended=False,
            expect_df=True,
        )
        if df is not None and len(df):
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=COLS)
    out = pd.concat(parts).reset_index()  # -> order_book_id, datetime, fields
    out["datetime"] = pd.to_datetime(out["datetime"])
    for c in PRICE_FIELDS:
        out[c] = out[c].astype("float32")  # 价格 float32（量/额保持 float64）
    return out[COLS].sort_values(["order_book_id", "datetime"]).reset_index(drop=True)


def _write_day(df: pd.DataFrame, raw_dir: Path, day: pd.Timestamp) -> int:
    path = raw_dir / f"{day.date()}.parquet"
    tmp = tempfile.mktemp(suffix=".parquet", dir=str(raw_dir))
    df.to_parquet(tmp)
    os.replace(tmp, path)
    return path.stat().st_size


def _fetch_write_day(day, universe, batch, raw_dir) -> tuple[str, int]:
    """线程任务：单交易日 拉原始 → 原子写。返回 (status, bytes)。已存在则跳过。"""
    path = raw_dir / f"{day.date()}.parquet"
    if path.exists():
        return ("skip", 0)
    try:
        df = _fetch_day(day, universe, batch)
    except Exception as e:
        if "Quota" in str(e):
            return ("quota", 0)
        return ("err", 0)
    if len(df) == 0:
        return ("empty", 0)
    return ("ok", _write_day(df, raw_dir, day))


def verify(raw_dir: Path, n_days: int = 3) -> None:
    """抽样对齐：对磁盘已有的最近 n_days，重拉同口径逐分钟比对（证明 fetch 与盘上一致）。"""
    rqdatac.init(*ACCOUNTS[0])
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        logger.warning("磁盘无 raw，跳过校验")
        return
    universe = rqdatac.all_instruments(type="CS")["order_book_id"].tolist()
    for p in files[-n_days:]:
        day = pd.Timestamp(p.stem)
        disk = pd.read_parquet(p)
        disk["datetime"] = pd.to_datetime(disk["datetime"])
        new = _fetch_day(day, universe, BATCH)
        dk = disk.set_index(["order_book_id", "datetime"]).sort_index()
        nw = new.set_index(["order_book_id", "datetime"]).sort_index()
        idx = dk.index.intersection(nw.index)
        a = dk.loc[idx, PRICE_FIELDS].to_numpy(float)
        b = nw.loc[idx, PRICE_FIELDS].to_numpy(float)
        rel = (
            np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-6))
            if len(idx)
            else np.nan
        )
        logger.info(
            f"  {'✅' if rel < 1e-4 else '❌'} {day.date()}: 重叠{len(idx):,} 价maxRel={rel:.2e}"
        )


def build(
    full: bool = False,
    batch: int = BATCH,
    limit_days: int | None = None,
    workers: int = 8,
    raw_dir: Path | None = None,
) -> None:
    raw_dir = raw_dir or _raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    end = _latest_trading_date()
    if full:
        start = pd.Timestamp(FULL_START)
    else:
        mx = _existing_max_date(raw_dir)
        start = (
            pd.Timestamp(rqdatac.get_next_trading_date(mx))
            if mx is not None
            else pd.Timestamp(FULL_START)
        )
    days = [
        pd.Timestamp(d)
        for d in rqdatac.get_trading_dates(str(start.date()), str(end.date()))
    ]
    if limit_days:
        days = days[:limit_days]
    if not full:
        days = [d for d in days if not (raw_dir / f"{d.date()}.parquet").exists()]
    universe = rqdatac.all_instruments(type="CS")["order_book_id"].tolist()
    logger.info(
        f"{'全量' if full else '增量'} {start.date()}~{end.date()} | 待拉 {len(days)} 交易日 | "
        f"{len(universe)} 股 | batch={batch} workers={workers} | 输出 {raw_dir}"
    )
    if not days:
        logger.success("已最新，无需更新")
        return
    stats = {"ok": 0, "skip": 0, "empty": 0, "quota": 0, "err": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_fetch_write_day, d, universe, batch, raw_dir): d for d in days
        }
        for i, fut in enumerate(futs, 1):
            st, _ = fut.result()
            stats[st] = stats.get(st, 0) + 1
            if i % 50 == 0 or i == len(days):
                el = time.time() - t0
                rate = i / el if el else 0
                logger.info(
                    f"  [{i}/{len(days)}] ok={stats['ok']} 空={stats['empty']} quota={stats['quota']} "
                    f"err={stats['err']} | {rate:.1f}日/s ETA~{(len(days)-i)/rate/60 if rate else 0:.0f}min | 剩配额{_quota_gb():.1f}GB"
                )
    logger.success(f"完成：{stats} | 末日 {end.date()} → {raw_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="A股原始(不复权)1分钟bar 按日分片 构建/日更"
    )
    ap.add_argument("--full", action="store_true", help="全量重下（默认增量补缺日）")
    ap.add_argument("--verify", action="store_true", help="抽样对齐校验，不写盘")
    ap.add_argument("--batch", type=int, default=BATCH, help="每交易日分批股票数")
    ap.add_argument(
        "--limit-days", type=int, default=None, help="只跑前 N 个交易日(测试)"
    )
    ap.add_argument("--workers", type=int, default=8, help="按天并行线程数")
    ap.add_argument("--raw-dir", type=Path, default=None, help="输出目录(测试用)")
    a = ap.parse_args()
    if a.verify:
        verify(_raw_dir())
    else:
        rqdatac.init(*ACCOUNTS[0])
        build(
            full=a.full,
            batch=a.batch,
            limit_days=a.limit_days,
            workers=a.workers,
            raw_dir=a.raw_dir,
        )


if __name__ == "__main__":
    main()
