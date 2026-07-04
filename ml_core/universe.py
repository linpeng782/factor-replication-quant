"""
ml_core.universe —— 管线底座层（模型无关、因子无关）
============================================================
只回答一件事："哪些 (日期, 股票) 格子是合法样本候选"。由三件相互独立的事拼出：

  universe   公共网格列空间 = dquant instruments 全集（每日全市场快照的并集）
  can_buy    T+1 非 ST/停牌/新股 —— masks 派生（shift(-1) 语义），即旧 can_buy_mask
  has_label  forward_return_Nd 非 NaN —— labels 派生（依赖 horizon）

**不含 has_factor**：那是随因子集 + 模型完整性策略变化的动态量，归 features 层现算，
不进底座（避免像 ml_ht 长表那样把动态量焊死进静态产物 → 加因子就过期）。

口径与 ml/ml_ht 完全一致：
  - can_buy 走 alpha_shared.cleaning.mask_loader（向量化、已 bit 级验证），等价旧 can_buy_mask；
    不重写 ml_ht 那段 _next_day_map 循环。
  - 缺列/缺日期 reindex 一律补 False（不可买 / 无标签）→ 退市股、未上市段被自动滤掉。
  - 公共网格行=交易日历、列=全集股票；其余宽表（因子/标签）都 reindex 到此网格。

底座的"固定不变"体现在：它只依赖 masks + labels + instruments，而这三者本就在日更 cron 上；
所以无需单独物化一个会过期的底座文件——以一个权威函数 + 已日更的输入为真相源即可。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from alpha_shared.cleaning.mask_loader import load_filter_masks
from core import config


@dataclass
class Universe:
    """管线底座：公共网格 + 两个模型/因子无关的样本谓词。"""

    dates: pd.DatetimeIndex   # 公共网格行空间（交易日历）
    stocks: pd.Index          # 公共网格列空间（dquant 全集）
    can_buy: np.ndarray       # (T, N) bool —— T+1 可买入（旧 can_buy_mask）
    has_label: np.ndarray     # (T, N) bool —— 远期收益可计算
    horizon: int = 20         # has_label 对应的远期收益期数

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.dates), len(self.stocks))


def dquant_grid() -> tuple[pd.DatetimeIndex, pd.Index]:
    """dquant instruments 全集 → 公共网格 (dates, stocks)。

    stocks = 逐股 OHLCV 目录的文件名全集（已实测：逐股文件 == 每日 per-day 快照并集，差异 0）。
    dates  = per-day 快照目录的交易日（文件名即日期）。
    两者都只 listdir、不读 parquet 内容 → instant，且是 dquant instruments 的权威口径。
    """
    stock_dir = config.RAW_OHLCV_DIR                      # 后端感知：dquant→stock-ohlcv-dquant
    stocks = pd.Index(sorted(p.stem for p in stock_dir.glob("*.parquet")))

    per_day_dir = stock_dir.parent / "per-day"            # daily_dquant/per-day
    if per_day_dir.exists():
        dates = pd.DatetimeIndex(sorted(pd.Timestamp(p.stem) for p in per_day_dir.glob("*.parquet")))
    else:
        # 回退：无 per-day 快照（如 rq 后端）时，用标签的日期作交易日历
        ref = pd.read_parquet(config.LABELS_DIR / "forward_return_20d.parquet", columns=[])
        dates = pd.DatetimeIndex(pd.to_datetime(ref.index)).sort_values()
    return dates, stocks


def _load_label_wide(horizon: int) -> pd.DataFrame:
    path = config.LABELS_DIR / f"forward_return_{horizon}d.parquet"
    if not path.exists():
        raise FileNotFoundError(f"标签不存在：{path}")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def build_universe(
    dates: pd.DatetimeIndex | None = None,
    stocks: pd.Index | None = None,
    horizon: int = 20,
    start: str | None = None,
    end: str | None = None,
) -> Universe:
    """构建底座：公共网格 + can_buy + has_label（模型/因子均无关）。

    dates / stocks 留空则取 dquant_grid() 全集网格；显式传入则用调用方指定的网格
    （对齐验证时可传 ml 旧网格做逐 bit 比对）。
    start / end 对网格日期裁剪（YYYY-MM-DD，增量/预测窗口用，避免每次算全史）。
    """
    if dates is None or stocks is None:
        g_dates, g_stocks = dquant_grid()
        dates = g_dates if dates is None else dates
        stocks = g_stocks if stocks is None else stocks
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]

    # can_buy = 旧 can_buy_mask = NOT(st|suspended|new)@T+1；缺格补 False（不可买）
    can_buy_mask, _ = load_filter_masks(
        combo_mask_path=config.COMBO_MASK_PATH,
        new_stock_mask_path=config.NEW_STOCK_MASK_PATH,
    )
    can_buy = (
        can_buy_mask.reindex(index=dates, columns=stocks).fillna(False).to_numpy(dtype=bool)
    )

    # has_label = forward_return_{horizon}d 非 NaN；缺格 → False（无标签）
    lab = _load_label_wide(horizon).reindex(index=dates, columns=stocks)
    has_label = np.isfinite(lab.to_numpy(dtype=np.float32))

    logger.info(
        f"[universe] 网格 {len(dates)}×{len(stocks)} "
        f"({dates.min().date()}~{dates.max().date()}) | "
        f"can_buy={can_buy.sum():,} | has_label(h={horizon})={int(has_label.sum()):,}"
    )
    return Universe(
        dates=dates, stocks=stocks, can_buy=can_buy, has_label=has_label, horizon=horizon
    )
