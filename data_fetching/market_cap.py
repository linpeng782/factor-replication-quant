"""
A 股总市值面板构建 / 日更
============================================================
输出 (T 交易日 × N 股票) 宽表面板：
  index   : DatetimeIndex, name='date'
  columns : 股票代码, name='stock'
  dtype   : float32, 单位 = 亿元

字段：market_cap_3 —— A 股总市值（含限售，= a_share_market_val）/ 1e8。
      与 alpha-158 PRODUCTION_PIPELINE 口径一致；
      实测 300750.XSHE @2024-01-02 = 6899 亿（对万得/同花顺）。

两种模式
--------
  python -m alpha_shared.neu.market_cap            # 增量日更（默认）
      读现有面板 → 仅补齐缺失的近期交易日（全市场一次批量 fetch）→ 合并保存
  python -m alpha_shared.neu.market_cap --full     # 全量重建（分年批量，2005-至今）

输出路径（优先级）
------------------
  1. --output 显式指定
  2. 环境变量 $MARKET_CAP_PANEL_PATH（完整路径）
  3. $FACTOR_REPL_DATA_ROOT/market-data/market_cap/market_cap_panel.parquet
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
FACTOR_NAME = "market_cap_3"   # A 股总市值（含限售）
SCALE = 1e8                    # 元 → 亿元
DTYPE = "float32"
FULL_START = "2005-01-01"      # 全量重建起点


def _default_output() -> Path:
    """解析默认输出路径（见模块 docstring 优先级 2/3）。"""
    explicit = os.environ.get("MARKET_CAP_PANEL_PATH")
    if explicit:
        return Path(explicit)
    root = Path(os.environ.get("FACTOR_REPL_DATA_ROOT", "/nfs/ofs-prediction/peterzhenglinpeng"))
    return root / "market-data" / "market_cap" / "market_cap_panel.parquet"


# ==================== fetch ====================
def _fetch_wide(stocks: list[str], start: str, end: str) -> pd.DataFrame:
    """全市场一次批量 get_factor → (date × stock) 宽表，单位亿元。"""
    df = rqdatac.get_factor(stocks, FACTOR_NAME, start_date=start, end_date=end)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    s = df[FACTOR_NAME] if FACTOR_NAME in df.columns else df.iloc[:, 0]
    wide = s.dropna().unstack(level="order_book_id").sort_index() / SCALE
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "date"
    wide.columns.name = "stock"
    return wide


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
    rqdatac.init()
    stocks = rqdatac.all_instruments(type="CS")["order_book_id"].tolist()
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    output = Path(output)

    if full or not output.exists():
        logger.info(f"[全量] 重建 {FULL_START} ~ {today}，{len(stocks)} 股，分年批量 fetch")
        cur_year = int(today[:4])
        parts = []
        for y in range(int(FULL_START[:4]), cur_year + 1):
            s = f"{y}-01-01"
            e = today if y == cur_year else f"{y}-12-31"
            w = _fetch_wide(stocks, s, e)
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
        new = _fetch_wide(stocks, start, today)
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
        description="A股总市值面板 构建/日更（market_cap_3，亿元，float32）"
    )
    ap.add_argument("--full", action="store_true", help="全量重建（默认增量日更）")
    ap.add_argument("--output", type=Path, default=_default_output(), help="输出 parquet 路径")
    args = ap.parse_args()
    logger.info(f"输出: {args.output} | 模式: {'全量重建' if args.full else '增量日更'}")
    build(args.output, full=args.full)


if __name__ == "__main__":
    main()
