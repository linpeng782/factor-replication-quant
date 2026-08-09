"""
开源证券 长端动量 1.0/2.0 论文对齐验证（月频 RankIC）
============================================================================
读 neu 三阶段面板（已行业市值中性化），月末截面 vs 次月 vwap 收益算月频 RankIC，
与研报（full.md line 178 / 206）对照。

研报口径：区间 2013-01-01~2022-10-31，因子做市值行业中性化，月末调仓。
  长端动量 1.0：RankIC 4.01%，RankICIR 1.76，胜率 74.36%
  长端动量 2.0：RankIC 6.92%，RankICIR 2.75，胜率 79.49%
RankICIR 为**年化**（月频 mean/std × √12）——由 2.0 的胜率 79.49% ≈ Φ(2.75/√12) 反推确认。

用法:
  PYTHONPATH=. python scripts/kysec_long_momentum_verify.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from alpha_shared.cleaning.mask_loader import load_filter_masks
from alpha_shared.evaluation.ic import compute_ic_series

NAMESPACE = "kysec-dquant/paper_67_long_momentum"
PAPER = {"long_mom_1": (4.01, 1.76, 74.36), "long_mom_2": (6.92, 2.75, 79.49)}
SEGMENTS = [("论文区间 2013-01~2022-10", "2013-01-01", "2022-10-31"),
            ("现代区间 2016-01~2026-07", "2016-01-01", "2026-07-31")]


def stats(factor: pd.DataFrame, ret_m: pd.DataFrame) -> tuple[float, float, float]:
    """月频 RankIC 均值(%) / 年化 ICIR / 胜率(%)。"""
    ic = compute_ic_series(factor, ret_m, method="spearman").dropna()
    if len(ic) < 12:
        return np.nan, np.nan, np.nan
    return ic.mean() * 100, ic.mean() / ic.std() * np.sqrt(12), (ic > 0).mean() * 100


def main() -> None:
    vwap = pd.read_parquet(config.VWAP_PANEL_PATH)
    vwap.index = pd.to_datetime(vwap.index)
    dates = vwap.index
    me = pd.DatetimeIndex(pd.Series(dates).groupby([dates.year, dates.month]).last())
    ret_m = vwap.loc[me].shift(-1) / vwap.loc[me] - 1

    can_buy, _ = load_filter_masks(
        combo_mask_path=config.COMBO_MASK_PATH,
        new_stock_mask_path=config.NEW_STOCK_MASK_PATH,
        reindex_columns=list(vwap.columns),
    )
    can_buy_me = can_buy.reindex(index=me, columns=vwap.columns)

    neu_dir = config.NEU_FACTOR_BASE / NAMESPACE
    print("\n" + "=" * 88)
    print("开源证券 长端动量 论文对齐验证（月频 RankIC，neu=行业市值中性化，vwap 次月收益）")
    print("=" * 88)
    for title, start, end in SEGMENTS:
        seg = me[(me >= start) & (me <= end)]
        print(f"\n【{title}】共 {len(seg)} 个月截面")
        print(f"  {'因子':<14}{'我 RankIC':>12}{'ICIR(年化)':>12}{'胜率':>9}"
              f"{'|':>4}{'论文 RankIC':>13}{'ICIR':>8}{'胜率':>9}")
        for name, (p_ic, p_ir, p_wr) in PAPER.items():
            path = neu_dir / f"{name}.parquet"
            if not path.exists():
                print(f"  {name:<14}{'面板缺失':>12}")
                continue
            fac = pd.read_parquet(path)
            fac.index = pd.to_datetime(fac.index)
            fac = fac.reindex(index=seg, columns=vwap.columns).where(can_buy_me.reindex(index=seg))
            ic, ir, wr = stats(fac, ret_m.reindex(seg))
            print(f"  {name:<14}{ic:>+11.2f}%{ir:>12.2f}{wr:>8.1f}%{'|':>4}"
                  f"{p_ic:>+12.2f}%{p_ir:>8.2f}{p_wr:>8.1f}%")
    print("\n  注：长端动量为正向因子，RankIC 应为正；2.0 应显著强于 1.0。")


if __name__ == "__main__":
    main()
