"""
A 股总市值面板构建 / 日更（dquant 后端）
============================================================
输出 (T 交易日 × N 股票) 宽表面板：
  index   : DatetimeIndex, name='date'
  columns : 股票代码, name='stock'
  dtype   : float32, 单位 = 亿元

字段：market_cap_3 —— A 股总市值（含限售，= a_share_market_val）/ 1e8。
      与 market_cap.py（rq 后端）口径完全一致，已 bit-exact 验证（max_rel=0）。

dquant 接口：ddata.get_market_cap(None, start, end) 全市场批量取数，~0.8s/月。

两种模式
--------
  python market_cap_dquant.py             # 增量日更（默认）
  python market_cap_dquant.py --full      # 全量重建（2005-至今）

输出路径（优先级）
------------------
  1. --output 显式指定
  2. 环境变量 $MARKET_CAP_PANEL_PATH（完整路径）
  3. $FACTOR_REPL_DATA_ROOT/market-data/market_cap/market_cap_panel.parquet

依赖：dquant（运行前需能 import dquant）。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import dquant  # noqa: F401  (确保 fork 子进程能继承)
from dquant import data as ddata

# ==================== 常量 ====================
FIELD = "market_cap_3"   # A 股总市值（含限售）
SCALE = 1e8              # 元 → 亿元
DTYPE = "float32"
FULL_START = "2005-01-01"


def _default_output() -> Path:
    """解析默认输出路径（见模块 docstring 优先级 2/3）。"""
    explicit = os.environ.get("MARKET_CAP_PANEL_PATH")
    if explicit:
        return Path(explicit)
    root = Path(os.environ.get("FACTOR_REPL_DATA_ROOT", "/nfs/ofs-prediction/peterzhenglinpeng"))
    return root / "market-data" / "market_cap" / "market_cap_panel.parquet"


# ==================== fetch ====================
def _fetch_wide(start: str, end: str) -> pd.DataFrame:
    """全市场批量 get_market_cap → pivot 宽表 (date × stock)，单位亿元。"""
    df = ddata.get_market_cap(None, start, end)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    wide = df.pivot(index="trade_date", columns="order_book_id", values=FIELD) / SCALE
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "date"
    wide.columns.name = "stock"
    return wide.sort_index().sort_index(axis=1)


def _normalize(panel: pd.DataFrame) -> pd.DataFrame:
    """统一 schema：去重日期（留最新）+ 排序行列 + float32。"""
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    panel = panel.reindex(columns=sorted(panel.columns))
    panel.index.name = "date"
    panel.columns.name = "stock"
    return panel.astype(DTYPE)


# ==================== 主流程 ====================
def build(output: Path, full: bool = False) -> pd.DataFrame:
    """构建或日更市值面板，返回最终面板。"""
    from data_fetching.dquant_source import latest_trading_date
    today = str(latest_trading_date())
    output = Path(output)

    if full or not output.exists():
        logger.info(f"[全量] 重建 {FULL_START} ~ {today}，分年批量 fetch")
        cur_year = int(today[:4])
        parts = []
        for y in range(int(FULL_START[:4]), cur_year + 1):
            s = f"{y}-01-01"
            e = today if y == cur_year else f"{y}-12-31"
            w = _fetch_wide(s, e)
            if not w.empty:
                parts.append(w)
                logger.info(f"  {y}: +{w.shape[0]} 日 × {w.shape[1]} 股")
        if not parts:
            raise RuntimeError("全量 fetch 无任何数据")
        panel = _normalize(pd.concat(parts))
    else:
        existing = pd.read_parquet(output)
        existing.index = pd.to_datetime(existing.index)
        last = existing.index.max()
        start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if start > today:
            logger.success(f"[日更] 已最新（到 {last.date()}），无需更新")
            return _normalize(existing)
        logger.info(f"[日更] 现有到 {last.date()}，补齐 {start} ~ {today}")
        new = _fetch_wide(start, today)
        if new.empty:
            logger.success(f"[日更] {start}~{today} 无新交易日数据，面板不变")
            return _normalize(existing)
        logger.info(f"  +{new.shape[0]} 新交易日（列取并集后合并）")
        panel = _normalize(pd.concat([existing, new]))

    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output)
    size_mb = output.stat().st_size / 1024**2
    logger.success(
        f"已保存 {output} | shape={panel.shape} | "
        f"{panel.index.min().date()}~{panel.index.max().date()} | "
        f"NaN={panel.isna().values.mean():.1%} | {size_mb:.1f}MB"
    )
    return panel


def main():
    ap = argparse.ArgumentParser(
        description="A股总市值面板 构建/日更（dquant 后端，market_cap_3，亿元，float32）"
    )
    ap.add_argument("--full", action="store_true", help="全量重建（默认增量日更）")
    ap.add_argument("--output", type=Path, default=_default_output(), help="输出 parquet 路径")
    args = ap.parse_args()
    logger.info(f"输出: {args.output} | 模式: {'全量重建' if args.full else '增量日更'}")
    build(args.output, full=args.full)


if __name__ == "__main__":
    main()
