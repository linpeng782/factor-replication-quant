"""
A 股基本面 PIT 基础数据（get_factor 点位字段）按字段 WIDE 面板 构建 / 日更
============================================================
设计见 docs/cxl_fundamental_incremental_design.md。

形态（每个字段一个 parquet，date×stock 宽表）：
  market-data/fundamentals/<field>.parquet
  index   : DatetimeIndex（交易日）
  columns : order_book_id

「必须存」的命门——冻结 PIT，防漂移/前视
------------------------------------------------
  某天 D 拉到的 get_factor 值，当天落盘即冻结、**历史永不改写**（只 append > last）。
  = as-first-reported point-in-time，免疫财报重述、零前视、可复现。
  ⚠️ 全量 bootstrap 拉的是 rqdatac 今天对历史的说法（老季度可能含重述）——这是 PIT 库起步现实
     （用厂商历史播种）；真正的 as-first-reported 从上线日起每天冻结快照前向积累。

「零手动日期」
--------------
  起点：增量=各字段面板磁盘最大日期 +1 交易日；全量=FULL_START
  终点：恒 = 最新就绪交易日（19:05 前退到前一交易日，与 raw_ohlcv 对齐）

三种模式
--------
  python data_fetching/fundamentals.py            # 增量日更（默认）：append disk_max+1 ~ 最新
  python data_fetching/fundamentals.py --full      # 全量 bootstrap（FULL_START ~ 最新，分年）
  python data_fetching/fundamentals.py --audit      # 重述审计：比对本地 vs API 当前，只报告不覆盖

复用 core.yolo_engine.incremental_append（与因子线同一内核，防脱节）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rqdatac
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import FUNDAMENTALS_DIR
from core.yolo_engine import DataFetcher, incremental_append

# ==================== 字段集（扫 spec get_factor 并集；新字段加这里）====================
# 财务点位（季频，mrq=most-recent-quarter，_N=往前 N 季；ttm=trailing-twelve-months）
FIELDS = [
    "net_profit_mrq_0", "net_profit_mrq_1", "net_profit_mrq_2", "net_profit_mrq_3",
    "net_profit_mrq_4", "net_profit_mrq_5", "net_profit_mrq_6", "net_profit_mrq_7",
    "net_profit_mrq_8",
    "net_profit_ttm_0",
    "total_equity_mrq_0", "total_equity_mrq_1", "total_equity_mrq_4",
    "cash_flow_from_operating_activities_ttm_0",
    "return_on_invested_capital_ttm",
    # 估值/市值（日频，由价格派生；一并快照存，口径统一）
    "market_cap_3", "pb_ratio_lf", "pe_ratio_ttm",
]
FULL_START = "2010-01-01"          # cxl 基线起点
AUDIT_N = 12                       # 审计抽样股票数
AUDIT_WINDOW = 20                  # 审计：比对磁盘末 N 个交易日


# ==================== 交易日 / 起点 ====================
def _latest_trading_date() -> pd.Timestamp:
    """终点 = 最近已收盘且就绪的交易日（未过 19:05 退到前一交易日，与 raw_ohlcv 对齐）。"""
    latest = pd.Timestamp(rqdatac.get_latest_trading_date())
    if pd.Timestamp.now() < latest.replace(hour=19, minute=5, second=0):
        return pd.Timestamp(rqdatac.get_previous_trading_date(latest))
    return latest


def _panel_path(field: str) -> Path:
    return FUNDAMENTALS_DIR / f"{field}.parquet"


def _infer_start() -> pd.Timestamp:
    """增量起点 = 各字段面板磁盘最大日期 +1 交易日；本地空则 FULL_START。

    取各字段 max 的最小值（保守对齐，缺字段不漏）。只读 index，便宜。
    """
    maxes = []
    for f in FIELDS:
        p = _panel_path(f)
        if p.exists():
            idx = pd.to_datetime(pd.read_parquet(p, columns=[]).index)
            if len(idx):
                maxes.append(idx.max())
    if len(maxes) < len(FIELDS):       # 有字段缺面板 → 当作首建
        return pd.Timestamp(FULL_START)
    return pd.Timestamp(rqdatac.get_next_trading_date(min(maxes)))


# ==================== fetch → 按字段 WIDE ====================
def _fetch_long(fetcher: DataFetcher, universe, start, end) -> pd.DataFrame:
    """get_factor(全市场, FIELDS, [start,end]) → long [order_book_id, date, *FIELDS]。"""
    df = fetcher.get_factor(universe, FIELDS, start_date=str(start), end_date=str(end))
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def _write_fields(long: pd.DataFrame, full: bool) -> dict:
    """逐字段 unstack → date×stock 宽表 → incremental_append（append > last / 全量覆盖）。"""
    stats = {"ok": 0, "skip": 0}
    for field in FIELDS:
        if field not in long.columns:
            logger.warning(f"  字段 {field} 不在 get_factor 返回中，跳过")
            stats["skip"] += 1
            continue
        wide = long.pivot(index="date", columns="order_book_id", values=field)
        path = _panel_path(field)
        last = None
        if not full and path.exists():
            idx = pd.to_datetime(pd.read_parquet(path, columns=[]).index)
            last = idx.max() if len(idx) else None
        incremental_append(path, wide, last, rebuild=full)   # ★ 生产内核：列并集纳新股 + dedup + 原子写
        stats["ok"] += 1
    return stats


# ==================== 重述审计（只报告，绝不覆盖）====================
def audit() -> None:
    """比对"本地存的"vs"API 当前"末 AUDIT_WINDOW 天，报告漂移幅度（财报重述监控）。

    与 raw_ohlcv 的 verify 语义相反：这里**不中止、不覆盖**——历史以本地冻结快照为准。
    """
    fetcher = DataFetcher()
    sample_field = "net_profit_mrq_0"
    p = _panel_path(sample_field)
    if not p.exists():
        logger.warning("无本地基础数据，跳过审计")
        return
    disk = pd.read_parquet(p)
    disk.index = pd.to_datetime(disk.index)
    win = disk.tail(AUDIT_WINDOW)
    rng = np.random.default_rng(0)
    stocks = list(rng.choice(disk.columns, min(AUDIT_N, len(disk.columns)), replace=False))
    fresh = _fetch_long(fetcher, stocks, win.index.min().date(), win.index.max().date())
    if fresh.empty:
        logger.warning("审计 fetch 空")
        return
    fw = fresh.pivot(index="date", columns="order_book_id", values=sample_field)
    d = win.index.intersection(fw.index)
    c = [s for s in stocks if s in win.columns and s in fw.columns]
    a, b = win.loc[d, c], fw.loc[d, c]
    both = ~(a.isna() | b.isna())
    n_diff = int((both & (a != b)).values.sum())
    maxd = float((a[both] - b[both]).abs().max().max()) if both.values.any() else 0.0
    logger.info(
        f"[审计] {sample_field} 末 {len(d)} 日 × {len(c)} 股：本地 vs API 当前 "
        f"不一致 {n_diff} 格 | maxΔ={maxd:.4g}"
    )
    if n_diff:
        logger.warning("  ⚠️ 检测到本地 PIT 快照与 API 当前口径不一致（财报重述/口径变更）——"
                       "本地以冻结快照为准，**不覆盖**；如需对齐请人工决策 --full reconcile。")
    else:
        logger.success("  ✅ 无漂移：本地快照与 API 当前一致")


# ==================== 主流程 ====================
def build(full: bool = False) -> None:
    rqdatac.init()
    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)
    fetcher = DataFetcher()

    end = _latest_trading_date()
    start = pd.Timestamp(FULL_START) if full else _infer_start()
    if start > end:
        logger.success(f"已最新（磁盘到 {pd.Timestamp(rqdatac.get_previous_trading_date(start)).date()}），无需更新")
        return

    universe = sorted(rqdatac.all_instruments(type="CS")["order_book_id"].tolist())
    logger.info(
        f"{'全量 bootstrap' if full else '增量'} {start.date()} ~ {end.date()} | "
        f"{len(universe)} 只股 × {len(FIELDS)} 字段"
    )

    spans = (
        [(f"{y}-01-01", f"{y}-12-31") for y in range(start.year, end.year + 1)]
        if full else [(start.date(), end.date())]
    )
    parts = []
    for s, e in spans:
        s = max(pd.Timestamp(s), start).date()
        e = min(pd.Timestamp(e), end).date()
        if pd.Timestamp(s) > pd.Timestamp(e):
            continue
        df = _fetch_long(fetcher, universe, s, e)
        if not df.empty:
            parts.append(df)
            logger.info(f"  fetch {s}~{e}: +{len(df):,} 行")
    if not parts:
        logger.warning("fetch 无数据（非交易日或数据未就绪）")
        return

    long = pd.concat(parts, ignore_index=True)
    stats = _write_fields(long, full=full)
    logger.success(
        f"✅ 基本面 PIT {'bootstrap' if full else '增量'}完成：{stats['ok']} 字段写入"
        f"（skip={stats['skip']}）→ 末日 {end.date()}"
    )


def main():
    ap = argparse.ArgumentParser(description="基本面 PIT 基础数据（get_factor）按字段 WIDE 构建/日更")
    ap.add_argument("--full", action="store_true", help="全量 bootstrap（默认增量日更）")
    ap.add_argument("--audit", action="store_true", help="重述审计：比对本地 vs API 当前，只报告不覆盖")
    args = ap.parse_args()
    if args.audit:
        rqdatac.init()
        audit()
        return
    logger.info(f"输出目录: {FUNDAMENTALS_DIR} | 模式: {'全量 bootstrap' if args.full else '增量日更'}")
    build(full=args.full)


if __name__ == "__main__":
    main()
