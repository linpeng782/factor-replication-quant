"""
A 股原始 OHLCV（不复权）逐股 parquet 构建 / 日更
============================================================
输出（每只股票一个 parquet）：
  index   : DatetimeIndex, name='date'
  columns : open/high/low/close/volume/total_turnover/limit_up/limit_down
  口径    : adjust_type='none'（不复权；复权在读时由 cum_factor 实时算）

「零手动日期」设计
------------------
  起始日：增量=磁盘已有最大日期 +1 交易日（断点续传）；全量=FULL_START
  结尾日：恒 = rqdatac.get_latest_trading_date()（自动落到最近交易日，周末/节假日免管）

三种模式
--------
  python raw_ohlcv.py                 # 增量日更（默认）：补 disk_max+1 ~ 最新交易日
  python raw_ohlcv.py --full          # 全量重下（FULL_START ~ 最新，分年批量）
  python raw_ohlcv.py --verify-only   # 只做抽样对齐校验（fetch 重叠窗口 vs 磁盘逐值比对），不写盘

更新前自动做一次抽样对齐校验（fetch 与磁盘重叠的近期窗口比对），口径不一致则中止。

输出路径优先级：--output > $RAW_OHLCV_DIR > $FACTOR_REPL_DATA_ROOT/market-data/daily/stock-ohlcv/
依赖：rqdatac（bare init；账号走 $RQSDK_LICENSE / $RQDATAC_CONF）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rqdatac
from loguru import logger

# ==================== 常量 ====================
OHLCV_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "total_turnover",
    "limit_up",
    "limit_down",
]
FULL_START = "2005-01-01"
PROBE_STOCKS = ["000001.XSHE", "600000.XSHG", "300001.XSHE"]  # 推断增量起点用
VERIFY_N = 12  # 抽样对齐校验股票数
VERIFY_WINDOW = 6  # 抽样对齐校验：取磁盘最后 N 个交易日做重叠比对
VERIFY_TOL = 1e-6  # 不复权原始价应逐值相等


def _default_dir() -> Path:
    explicit = os.environ.get("RAW_OHLCV_DIR")
    if explicit:
        return Path(explicit)
    root = Path(
        os.environ.get("FACTOR_REPL_DATA_ROOT", "/nfs/ofs-prediction/peterzhenglinpeng")
    )
    return root / "market-data" / "daily" / "stock-ohlcv"


def _latest_trading_date() -> pd.Timestamp:
    """终点 = 最近一个已收盘且数据就绪的交易日（未过19:05则退到前一交易日）。"""
    latest = pd.Timestamp(rqdatac.get_latest_trading_date())
    now = pd.Timestamp.now()
    cutoff = latest.replace(hour=19, minute=5, second=0)
    if now < cutoff:
        return pd.Timestamp(rqdatac.get_previous_trading_date(latest))
    return latest


def _infer_start(out_dir: Path) -> pd.Timestamp:
    """增量起点 = 探针股磁盘最大日期 +1 交易日；本地空则 FULL_START。"""
    latest = []
    for s in PROBE_STOCKS:
        p = out_dir / f"{s}.parquet"
        if p.exists():
            d = pd.read_parquet(p, columns=[])  # 只读 index
            if len(d):
                latest.append(pd.to_datetime(d.index).max())
    if not latest:
        return pd.Timestamp(FULL_START)
    return pd.Timestamp(rqdatac.get_next_trading_date(max(latest)))


def _fetch(stocks, start, end) -> pd.DataFrame:
    df = rqdatac.get_price(
        stocks,
        start_date=str(start),
        end_date=str(end),
        frequency="1d",
        fields=OHLCV_FIELDS,
        adjust_type="none",
        skip_suspended=False,
        expect_df=True,
    )
    return df if df is not None else pd.DataFrame()


# ==================== 抽样对齐校验 ====================
def verify_sample(out_dir: Path, n: int = VERIFY_N) -> bool:
    """fetch 与磁盘重叠的近期窗口，逐值比对，证明 fetch 口径(none)与盘上一致。"""
    files = sorted(out_dir.glob("*.parquet"))
    if not files:
        logger.warning("磁盘无 ohlcv，跳过抽样对齐（首次全量无需）")
        return True
    rng = np.random.default_rng(0)
    sample = [files[0].stem] + [
        f.stem for f in rng.choice(files, min(n, len(files)), replace=False)
    ]
    sample = list(dict.fromkeys(sample))[:n]

    bad = 0
    logger.info(
        f"抽样对齐校验：{len(sample)} 只股票，重叠窗口=磁盘末 {VERIFY_WINDOW} 个交易日"
    )
    for s in sample:
        disk = pd.read_parquet(out_dir / f"{s}.parquet")
        disk.index = pd.to_datetime(disk.index)
        if len(disk) < 2:
            continue
        win = disk.tail(VERIFY_WINDOW)
        new = _fetch([s], win.index.min().date(), win.index.max().date())
        if new.empty:
            logger.warning(f"  {s}: fetch 空，跳过")
            continue
        new = (
            new.reset_index()
            .set_index("date")
            .drop(columns=["order_book_id"])
            .sort_index()
        )
        common_idx = win.index.intersection(new.index)
        common_col = [c for c in OHLCV_FIELDS if c in win.columns and c in new.columns]
        a = win.loc[common_idx, common_col].to_numpy(np.float64)
        b = new.loc[common_idx, common_col].to_numpy(np.float64)
        both = ~np.isnan(a) & ~np.isnan(b)
        maxd = float(np.abs(a[both] - b[both]).max()) if both.any() else np.nan
        ok = (maxd <= VERIFY_TOL) if not np.isnan(maxd) else False
        logger.info(
            f"  {'✅' if ok else '❌'} {s}: 重叠 {len(common_idx)} 日 maxΔ={maxd:.2e}"
        )
        bad += not ok
    if bad:
        logger.error(
            f"抽样对齐失败 {bad}/{len(sample)} 只——fetch 口径与磁盘不一致，中止更新"
        )
        return False
    logger.success(f"抽样对齐通过：{len(sample)} 只股票重叠窗口逐值一致（口径=不复权）")
    return True


# ==================== 主流程 ====================
def build(out_dir: Path, full: bool = False, verify_only: bool = False) -> None:
    rqdatac.init()
    try:
        logger.info(f"rqdatac 配额: {rqdatac.user.get_quota()}")
    except Exception as e:
        logger.warning(f"配额查询失败: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    end = _latest_trading_date()

    if verify_only:
        verify_sample(out_dir)
        return
    if not full and not verify_sample(out_dir):
        return  # 口径不一致已中止

    stocks = rqdatac.all_instruments(type="CS")["order_book_id"].tolist()
    start = pd.Timestamp(FULL_START) if full else _infer_start(out_dir)
    if start > end:
        logger.success(
            f"已最新（磁盘到 {pd.Timestamp(rqdatac.get_previous_trading_date(start)).date()}），无需更新"
        )
        return
    logger.info(
        f"{'全量重下' if full else '增量'} {start.date()} ~ {end.date()}，{len(stocks)} 只股票"
    )

    # 分年批量（全量）或单批（增量，区间短）
    spans = (
        [(f"{y}-01-01", f"{y}-12-31") for y in range(start.year, end.year + 1)]
        if full
        else [(start.date(), end.date())]
    )
    parts = []
    for s, e in spans:
        s = max(pd.Timestamp(s), start).date()
        e = min(pd.Timestamp(e), end).date()
        if pd.Timestamp(s) > pd.Timestamp(e):
            continue
        df = _fetch(stocks, s, e)
        if not df.empty:
            parts.append(df)
            logger.info(f"  fetch {s}~{e}: +{len(df)} 行")
    if not parts:
        logger.warning("fetch 无数据（非交易日或数据未就绪）")
        return
    allnew = pd.concat(parts).reset_index()

    stats = {"updated": 0, "new": 0}
    for stock, g in allnew.groupby("order_book_id"):
        g = g.drop(columns=["order_book_id"]).set_index("date").sort_index()
        path = out_dir / f"{stock}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            old.index = pd.to_datetime(old.index)
            comb = pd.concat([old, g])
            comb = comb[~comb.index.duplicated(keep="last")].sort_index()
            comb.to_parquet(path)
            stats["updated"] += 1
        else:
            g.to_parquet(path)
            stats["new"] += 1
    logger.success(
        f"写入完成：更新 {stats['updated']} 只 + 新建 {stats['new']} 只 → 末日 {end.date()}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="A股原始OHLCV(不复权)逐股 parquet 构建/日更"
    )
    ap.add_argument("--full", action="store_true", help="全量重下（默认增量日更）")
    ap.add_argument(
        "--verify-only", action="store_true", help="只做抽样对齐校验，不写盘"
    )
    ap.add_argument("--output", type=Path, default=_default_dir(), help="输出目录")
    args = ap.parse_args()
    logger.info(
        f"输出目录: {args.output} | 模式: "
        f"{'校验' if args.verify_only else ('全量' if args.full else '增量日更')}"
    )
    build(args.output, full=args.full, verify_only=args.verify_only)


if __name__ == "__main__":
    main()
