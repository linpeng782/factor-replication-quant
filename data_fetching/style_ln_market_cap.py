"""
ln(市值) 风格因子 构建/日更（dquant 后端）
============================================================
输出 (T 交易日 × N 股票) 宽表面板：
  index   : DatetimeIndex, name='date'
  columns : 股票代码, name='stock'
  dtype   : float32
  values  : ln(market_cap_3_亿元)，市值 <=0 置 NaN

定义：ln_market_cap = log(market_cap_3)，单位亿元（与 market_cap_panel 一致）。
      log(亿元) 与 log(元) 仅差常数 log(1e8)，对 rank/zscore/线性控制等价。

用途：大小盘画像 / regime 切分 / 其他因子的市值中性化控制变量 / ml_core style exposure。
      属风格暴露，不走 cleaned/neu 三阶段（对 size 做市值中性化是自我抵消）。

原料：market-data/market_cap/market_cap_panel.parquet（已日更到最新交易日）。
      纯函数变换，零 API，<5 秒。

两种模式
--------
  python style_ln_market_cap.py             # 增量日更（默认，跟在 market_cap_dquant.py 后）
  python style_ln_market_cap.py --full      # 全量重建（读全量市值面板重算）

输出路径（优先级）
------------------
  1. --output 显式指定
  2. config.RAW_FACTOR_BASE / "style-dquant" / "size" / "ln_market_cap.parquet"
     （dquant 后端 → factors/raw-dquant/style-dquant/size/ln_market_cap.parquet）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import config as cfg

# ==================== 常量 ====================
DTYPE = "float32"


def _default_output() -> Path:
    """默认输出路径：factors/raw-dquant/style-dquant/size/ln_market_cap.parquet。"""
    return cfg.RAW_FACTOR_BASE / "style-dquant" / "size" / "ln_market_cap.parquet"


def _to_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """统一 schema：去重日期（留最新）+ 排序行列 + float32。"""
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    panel = panel.reindex(columns=sorted(panel.columns))
    panel.index.name = "date"
    panel.columns.name = "stock"
    return panel.astype(DTYPE)


def build(output: Path, full: bool = False) -> pd.DataFrame:
    """构建或日更 ln(市值) 面板，返回最终面板。

    full=True  : 读全量市值面板重算
    full=False : 增量——只补齐市值面板末日 > 现有 ln 面板末日 的新交易日
    """
    mcap_path = cfg.MARKET_CAP_PANEL_PATH
    if not mcap_path.exists():
        raise FileNotFoundError(
            f"市值面板不存在: {mcap_path}\n"
            "请先运行: PYTHONPATH=. python data_fetching/market_cap_dquant.py"
        )
    mcap = pd.read_parquet(mcap_path)
    mcap.index = pd.to_datetime(mcap.index)
    mcap_last = mcap.index.max()
    logger.info(f"市值面板: {mcap_path.name} | shape={mcap.shape} | 末日={mcap_last.date()}")

    output = Path(output)
    if full or not output.exists():
        logger.info(f"[全量] 重算 ln(market_cap) {mcap.index.min().date()}~{mcap_last.date()}")
        panel = _to_wide(np.log(mcap.where(mcap > 0)))
    else:
        existing = pd.read_parquet(output)
        existing.index = pd.to_datetime(existing.index)
        last = existing.index.max()
        if mcap_last <= last:
            logger.success(f"[日更] 已最新（到 {last.date()}），无需更新")
            return _to_wide(existing)
        start = last + pd.Timedelta(days=1)
        logger.info(f"[日更] 现有到 {last.date()}，补齐 {start.date()} ~ {mcap_last.date()}")
        new = np.log(mcap.loc[mcap.index >= start].where(mcap.loc[mcap.index >= start] > 0))
        panel = _to_wide(pd.concat([existing, new]))

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
        description="ln(市值) 风格因子 构建/日更（读 market_cap_panel，log 变换，零 API）"
    )
    ap.add_argument("--full", action="store_true", help="全量重建（默认增量日更）")
    ap.add_argument("--output", type=Path, default=_default_output(), help="输出 parquet 路径")
    args = ap.parse_args()
    logger.info(f"输出: {args.output} | 模式: {'全量重建' if args.full else '增量日更'}")
    build(args.output, full=args.full)


if __name__ == "__main__":
    main()
