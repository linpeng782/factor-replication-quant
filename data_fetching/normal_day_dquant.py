"""
正常交易日 mask 面板 构建 / 日更（dquant 后端）
============================================================
输出 (T 交易日 × N 股票) 宽表面板：
  index   : DatetimeIndex, name='date'
  columns : 股票代码, name='stock'
  dtype   : float32，1.0 = 正常交易日，0.0 = 停牌或触及涨跌停，NaN = 当日无该股数据

口径（开源证券长端动量 2.0 研报表2 步骤2「剔除涨跌停及停牌交易日数据」）：
  · 停牌   ← combo_mask_long 的 is_suspended（上游 cron 维护）
  · 涨跌停 ← dquant get_limit 的 limit_up/limit_down **收盘价触及**判定：
             close >= limit_up − eps  或  close <= limit_down + eps
             （用**未复权** close 与涨跌停价同口径比较）

两种模式
--------
  python data_fetching/normal_day_dquant.py             # 增量日更（默认）
  python data_fetching/normal_day_dquant.py --full      # 全量重建（2006-至今）

依赖：dquant + 本地 RAW_OHLCV_DIR（未复权 close）+ COMBO_MASK_PATH。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger

import dquant  # noqa: F401
from dquant import data as ddata

from config import COMBO_MASK_PATH, NORMAL_DAY_PANEL_PATH, RAW_OHLCV_DIR

DTYPE = "float32"
FULL_START = "2005-01-01"
EPS = 1e-4          # 涨跌停价比较容差（元）


def _close_series(stock: str) -> tuple[str, pd.Series | None]:
    """读单股**未复权** close（与涨跌停价同口径）。"""
    p = RAW_OHLCV_DIR / f"{stock}.parquet"
    if not p.exists():
        return stock, None
    raw = pd.read_parquet(p, columns=["close"])
    raw.index = pd.to_datetime(raw.index)
    return stock, raw["close"]


def _load_close_wide(workers: int) -> pd.DataFrame:
    stocks = sorted(p.stem for p in RAW_OHLCV_DIR.glob("*.parquet"))
    logger.info(f"读本地未复权 close: {len(stocks)} 只股票, workers={workers}")
    series: dict[str, pd.Series] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_close_series, s) for s in stocks]
        for fut in as_completed(futs):
            stock, s = fut.result()
            if s is not None:
                series[stock] = s
    return pd.DataFrame(series).sort_index()


def _fetch_limit_wide(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """全市场涨跌停价 → 两张宽表 (date × stock)。分年批量，避免单次过大。"""
    ups, downs = [], []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        s = max(f"{y}-01-01", start)
        e = min(f"{y}-12-31", end)
        if s > e:
            continue
        df = ddata.get_limit(None, s, e)
        if df is None or len(df) == 0:
            continue
        up = df.pivot_table(index="trade_date", columns="order_book_id",
                            values="limit_up", aggfunc="last")
        dn = df.pivot_table(index="trade_date", columns="order_book_id",
                            values="limit_down", aggfunc="last")
        ups.append(up)
        downs.append(dn)
        logger.info(f"  {y}: +{up.shape[0]} 日 × {up.shape[1]} 股")
    if not ups:
        return pd.DataFrame(), pd.DataFrame()
    up = pd.concat(ups).sort_index()
    dn = pd.concat(downs).sort_index()
    up.index = pd.to_datetime(up.index)
    dn.index = pd.to_datetime(dn.index)
    return up, dn


def _suspended_wide(index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """combo_mask 长表 → is_suspended 宽表（缺格视为未停牌 False）。"""
    m = pd.read_parquet(COMBO_MASK_PATH, columns=["order_book_id", "datetime", "is_suspended"])
    m["datetime"] = pd.to_datetime(m["datetime"])
    wide = m.pivot_table(index="datetime", columns="order_book_id",
                         values="is_suspended", aggfunc="last")
    return wide.reindex(index=index, columns=columns).fillna(False).astype(bool)


def _build_panel(close: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """close 宽表 × 涨跌停价 × 停牌 → 正常交易日 mask（1/0，无数据处 NaN）。"""
    close = close.loc[(close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))]
    up, dn = _fetch_limit_wide(start, end)
    if up.empty:
        return pd.DataFrame()

    # 去重后取交集，再统一 reindex（分年拼接/多源可能带重复索引，直接 .loc 会错位）
    close = close[~close.index.duplicated(keep="last")]
    up = up[~up.index.duplicated(keep="last")]
    dn = dn[~dn.index.duplicated(keep="last")]
    idx = close.index.intersection(up.index).sort_values()
    cols = sorted(set(close.columns) & set(up.columns))
    c = close.reindex(index=idx, columns=cols)
    u = up.reindex(index=idx, columns=cols)
    d = dn.reindex(index=idx, columns=cols)

    hit_limit = (c >= u - EPS) | (c <= d + EPS)
    suspended = _suspended_wide(idx, cols)
    normal = (~hit_limit & ~suspended).astype(DTYPE)
    normal = normal.where(c.notna() & u.notna())      # 无行情/无涨跌停价 → NaN
    normal.index.name = "date"
    normal.columns.name = "stock"
    logger.info(
        f"正常交易日比例={normal.stack().mean():.3%} 反例(涨跌停/停牌)"
        f"={1 - normal.stack().mean():.3%}"
    )
    return normal


def build(output: Path, full: bool = False, workers: int = 32) -> pd.DataFrame:
    from data_fetching.dquant_source import latest_trading_date
    today = str(latest_trading_date())
    output = Path(output)

    close = _load_close_wide(workers)

    if full or not output.exists():
        logger.info(f"[全量] 重建 {FULL_START} ~ {today}")
        panel = _build_panel(close, FULL_START, today)
        if panel.empty:
            raise RuntimeError("全量构建无任何数据")
    else:
        existing = pd.read_parquet(output)
        existing.index = pd.to_datetime(existing.index)
        last = existing.index.max()
        start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if start > today:
            logger.success(f"[日更] 已最新（到 {last.date()}），无需更新")
            return existing
        logger.info(f"[日更] 现有到 {last.date()}，补齐 {start} ~ {today}")
        new = _build_panel(close, start, today)
        if new.empty:
            logger.success(f"[日更] {start}~{today} 无新数据，面板不变")
            return existing
        panel = pd.concat([existing, new])
        panel = panel[~panel.index.duplicated(keep="last")].sort_index()
        panel = panel.reindex(columns=sorted(set(panel.columns))).astype(DTYPE)
        panel.index.name = "date"
        panel.columns.name = "stock"

    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output)
    logger.success(
        f"已保存 {output} | shape={panel.shape} | "
        f"{panel.index.min().date()}~{panel.index.max().date()} | "
        f"NaN={panel.isna().values.mean():.1%} | {output.stat().st_size / 1024**2:.1f}MB"
    )
    return panel


def main():
    ap = argparse.ArgumentParser(
        description="正常交易日 mask 面板（1=非停牌且未触涨跌停）构建/日更"
    )
    ap.add_argument("--full", action="store_true", help="全量重建（默认增量日更）")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--output", type=Path, default=NORMAL_DAY_PANEL_PATH)
    args = ap.parse_args()
    logger.info(f"输出: {args.output} | 模式: {'全量重建' if args.full else '增量日更'}")
    build(args.output, full=args.full, workers=args.workers)


if __name__ == "__main__":
    main()
