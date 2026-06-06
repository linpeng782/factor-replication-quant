"""
A 股复权因子（稀疏 cum_factor）逐股 parquet 构建 / 日更
============================================================
输出（每只股票一个 parquet，仅在除权日有行）：
  index   : DatetimeIndex, name='ex_date'
  columns : ex_cum_factor / ex_factor / ex_end_date
  用途    : 后复权价 = 原始价 × ffill(ex_cum_factor)（见 alpha158 adjusted_panels）

设计：大多数股票多数日子无除权事件 → 查一个近期窗口，只对有返回的股票 append。
  ex_cum_factor 是「从上市起累乘」的绝对值，查窗口只返回窗口内事件但 cum_factor 仍是绝对值，
  故 append + dedup(keep last) 后整条序列仍正确单调。

「零手动日期」：终点 = get_latest_trading_date()；起点 = 终点 - LOOKBACK_DAYS（默认 30，足够兜回补漏）。

模式：python ex_factors.py            # 增量日更（默认，查近 30 天）
      python ex_factors.py --full     # 全量重建（2005 至今，分年）
      python ex_factors.py --lookback 60

输出路径优先级：--output > $EX_FACTORS_DIR > $FACTOR_REPL_DATA_ROOT/market-data/daily/stock-ex-factors/
依赖：rqdatac（bare init；账号走 $RQSDK_LICENSE / $RQDATAC_CONF）。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import rqdatac
from loguru import logger

KEEP_COLS = ["ex_cum_factor", "ex_factor", "ex_end_date"]
FULL_START = "2005-01-01"
LOOKBACK_DAYS = 30


def _default_dir() -> Path:
    explicit = os.environ.get("EX_FACTORS_DIR")
    if explicit:
        return Path(explicit)
    root = Path(os.environ.get("FACTOR_REPL_DATA_ROOT", "/nfs/ofs-prediction/peterzhenglinpeng"))
    return root / "market-data" / "daily" / "stock-ex-factors"


def build(out_dir: Path, full: bool = False, lookback: int = LOOKBACK_DAYS) -> None:
    rqdatac.init()
    try:
        logger.info(f"rqdatac 配额: {rqdatac.user.get_quota()}")
    except Exception as e:
        logger.warning(f"配额查询失败: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(rqdatac.get_latest_trading_date())
    start = pd.Timestamp(FULL_START) if full else (end - pd.Timedelta(days=lookback))
    stocks = rqdatac.all_instruments(type="CS")["order_book_id"].tolist()
    logger.info(f"{'全量' if full else '增量'} 查除权 {start.date()} ~ {end.date()}，{len(stocks)} 只股票")

    df = rqdatac.get_ex_factor(stocks, start_date=str(start.date()), end_date=str(end.date()))
    if df is None or len(df) == 0:
        logger.success(f"区间 {start.date()}~{end.date()} 无除权事件，复权因子无需更新")
        return

    df = df.reset_index()
    logger.info(f"拉到 {len(df)} 条除权记录，涉及 {df['order_book_id'].nunique()} 只股票")
    stats = {"new": 0, "updated": 0}
    for stock, g in df.groupby("order_book_id"):
        g = g.drop(columns=["order_book_id"]).set_index("ex_date").sort_index()
        g = g[[c for c in KEEP_COLS if c in g.columns]]
        path = out_dir / f"{stock}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            comb = pd.concat([old, g])
            comb = comb[~comb.index.duplicated(keep="last")].sort_index()
            if len(comb) > len(old):
                comb.to_parquet(path); stats["updated"] += 1
                logger.info(f"  {stock} 新增除权 → 末 cum_factor={g['ex_cum_factor'].iloc[-1]:.4f}")
        else:
            g.to_parquet(path); stats["new"] += 1
            logger.info(f"  {stock} 首次除权，建文件")
    logger.success(f"复权因子更新完成：新建 {stats['new']} 只 + 更新 {stats['updated']} 只")


def main():
    ap = argparse.ArgumentParser(description="A股复权因子(稀疏)逐股 parquet 构建/日更")
    ap.add_argument("--full", action="store_true", help="全量重建（默认增量近 N 天）")
    ap.add_argument("--lookback", type=int, default=LOOKBACK_DAYS, help="增量回溯天数（默认30）")
    ap.add_argument("--output", type=Path, default=_default_dir(), help="输出目录")
    args = ap.parse_args()
    logger.info(f"输出目录: {args.output} | 模式: {'全量' if args.full else f'增量(近{args.lookback}天)'}")
    build(args.output, full=args.full, lookback=args.lookback)


if __name__ == "__main__":
    main()
