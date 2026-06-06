"""
A 股中信(zx)行业 PIT 面板构建 / 日更
============================================================
输出 (T 交易日 × N 股票) 宽表面板：
  index   : DatetimeIndex, name='date'
  columns : 股票代码, name='stock'
  values  : 一级行业名（str，如 '银行'/'食品饮料'），未覆盖处 NaN

数据源：rqdatac 内部表 __internal__zx2019_industry（中信2019）
  每行 = 一只股票一段 [start_date, cancel_date) 的行业归属（含历史重分类）。
  PIT 口径：对每个交易日取「start_date ≤ 该日」的最新一段（按 start_date ffill），
            与 factor_analysis_fund.py 的 get_industry_exposure 方法一致。

注意：内部表跨 2019-12-02（中信2019发布日）混用新旧命名（如 电子元器件↔电子）。
      无需统一——中性化是逐日横截面，每天用当天自洽的 ~30 个行业 one-hot 即可。

两种模式
--------
  python industry.py            # 增量日更（默认）：仅追加缺失的近期交易日
  python industry.py --full     # 全量重建（2005-至今）

输出路径（优先级）
------------------
  1. --output 显式指定
  2. 环境变量 $INDUSTRY_PANEL_ZX_PATH（完整路径）
  3. $FACTOR_REPL_DATA_ROOT/market-data/industry/industry_panel_zx.parquet
     （FACTOR_REPL_DATA_ROOT 缺省为公司 NFS；本机 export 后自动重定向）

依赖：rqdatac（运行前需能 rqdatac.init() 认证）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import rqdatac
from loguru import logger

# ==================== 常量 ====================
INTERNAL_TABLE = "__internal__zx2019_industry"
INDUSTRY_COL = "first_industry_name"   # 一级行业名（中性化哑变量用一级）
FULL_START = "2005-01-01"


def _default_output() -> Path:
    explicit = os.environ.get("INDUSTRY_PANEL_ZX_PATH")
    if explicit:
        return Path(explicit)
    root = Path(os.environ.get("FACTOR_REPL_DATA_ROOT", "/nfs/ofs-prediction/peterzhenglinpeng"))
    return root / "market-data" / "industry" / "industry_panel_zx.parquet"


# ==================== 行业区间表 → 事件宽表 ====================
def _load_industry_events() -> pd.DataFrame:
    """拉中信区间表 → (start_date × stock) 事件宽表，值=一级行业名。

    每个 (stock, start_date) 一条「该日起归属某行业」事件；后续按交易日 ffill 即得 PIT。
    """
    raw = rqdatac.client.get_client().execute(INTERNAL_TABLE)
    df = pd.DataFrame(raw)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df = df[["order_book_id", "start_date", INDUSTRY_COL]].dropna()
    # 同股同日多条取最后（理论上不会有；保险）
    events = df.pivot_table(
        index="start_date", columns="order_book_id",
        values=INDUSTRY_COL, aggfunc="last",
    ).sort_index()
    events.columns.name = "stock"
    return events


def _panel_for_dates(events: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """对给定交易日，按 start_date ffill 取每股 as-of 行业 → (date × stock) 宽表。"""
    allidx = events.index.union(pd.DatetimeIndex(dates))
    panel = events.reindex(allidx).ffill().reindex(pd.DatetimeIndex(dates))
    panel.index.name = "date"
    panel.columns.name = "stock"
    return panel


def _normalize(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    panel = panel.reindex(columns=sorted(panel.columns))
    panel = panel.dropna(how="all", axis=1)  # 删全空股票列
    panel.index.name = "date"
    panel.columns.name = "stock"
    return panel


# ==================== 主流程 ====================
def build(output: Path, full: bool = False) -> pd.DataFrame:
    rqdatac.init()
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    output = Path(output)
    events = _load_industry_events()
    logger.info(f"中信区间表：{events.shape[1]} 股，{events.notna().sum().sum()} 段事件")

    if full or not output.exists():
        dates = pd.to_datetime(rqdatac.get_trading_dates(FULL_START, today))
        logger.info(f"[全量] 重建 {FULL_START} ~ {today}，{len(dates)} 交易日")
        panel = _normalize(_panel_for_dates(events, dates))
    else:
        existing = pd.read_parquet(output)
        existing.index = pd.to_datetime(existing.index)
        last = existing.index.max()
        new_dates = pd.to_datetime(rqdatac.get_trading_dates(
            (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), today))
        if len(new_dates) == 0:
            logger.success(f"[日更] 已最新（到 {last.date()}），无需更新")
            return _normalize(existing)
        logger.info(f"[日更] 现有到 {last.date()}，补齐 {len(new_dates)} 个新交易日")
        new = _panel_for_dates(events, new_dates)
        panel = _normalize(pd.concat([existing, new]))

    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output)
    size_mb = output.stat().st_size / 1024**2
    # 抽一天看活跃行业数
    last_row = panel.iloc[-1].dropna()
    logger.success(
        f"已保存 {output} | shape={panel.shape} | "
        f"{panel.index.min().date()}~{panel.index.max().date()} | "
        f"末日活跃行业={last_row.nunique()}个/{last_row.size}股 | {size_mb:.1f}MB"
    )
    return panel


def main():
    ap = argparse.ArgumentParser(description="A股中信行业 PIT 面板 构建/日更（zx 一级行业名）")
    ap.add_argument("--full", action="store_true", help="全量重建（默认增量日更）")
    ap.add_argument("--output", type=Path, default=_default_output(), help="输出 parquet 路径")
    args = ap.parse_args()
    logger.info(f"输出: {args.output} | 模式: {'全量重建' if args.full else '增量日更'}")
    build(args.output, full=args.full)


if __name__ == "__main__":
    main()
