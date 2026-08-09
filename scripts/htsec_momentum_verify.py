"""
华泰改进动量因子族 论文对齐验证（月频 IC）
============================================================================
读 neu 三阶段面板（已行业市值中性化），在月末截面上算月频 IC 与 IR 比率，
与研报图表115（full.md line 1024）对照。

研报口径（line 313/1014）：
  · 截面期 = 每个自然月最后一个交易日，对下一整月收益
  · 因子暴露度经【市值 + 行业】调整 → 对应本仓 neu 阶段产物
  · IR 比率 = |IC 均值 / IC 标准差|（月频，非年化）
  · 研报区间 2005-05 ~ 2016-11；本仓换手率数据自 2006-01 起 → 对照区间取 2006-01 起

用法:
  PYTHONPATH=. python scripts/htsec_momentum_verify.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

import config
from alpha_shared.cleaning.mask_loader import load_filter_masks
from alpha_shared.evaluation.ic import compute_ic_series

NAMESPACE = "htsec/paper_04_momentum"
# 研报图表115：IC 序列均值(%) / IR 比率
PAPER = {
    "wgt_return_1m": (-7.24, 0.92), "wgt_return_3m": (-6.12, 0.74),
    "wgt_return_6m": (-5.16, 0.60), "wgt_return_12m": (-4.18, 0.51),
    "exp_wgt_return_1m": (-6.04, 0.75), "exp_wgt_return_3m": (-7.74, 0.96),
    "exp_wgt_return_6m": (-7.70, 0.92), "exp_wgt_return_12m": (-6.86, 0.78),
}
SEGMENTS = [("论文区间 2006-01~2016-11", "2006-01-01", "2016-11-30"),
            ("现代区间 2016-01~2026-07", "2016-01-01", "2026-07-31")]


def month_end_ic(factor: pd.DataFrame, ret_m: pd.DataFrame, method: str) -> tuple[float, float]:
    """月末截面 IC 均值(%) 与 IR 比率（|mean/std|，月频不年化）。"""
    ic = compute_ic_series(factor, ret_m, method=method).dropna()
    if len(ic) < 12:
        return np.nan, np.nan
    return ic.mean() * 100, abs(ic.mean() / ic.std())


def main() -> None:
    vwap = pd.read_parquet(config.VWAP_PANEL_PATH)
    vwap.index = pd.to_datetime(vwap.index)
    dates = vwap.index
    # 月末交易日截面 + 次月 vwap 收益（研报：月末核算，下月调仓）
    me = pd.DatetimeIndex(pd.Series(dates).groupby([dates.year, dates.month]).last())
    ret_m = vwap.loc[me].shift(-1) / vwap.loc[me] - 1

    can_buy, _ = load_filter_masks(
        combo_mask_path=config.COMBO_MASK_PATH,
        new_stock_mask_path=config.NEW_STOCK_MASK_PATH,
        reindex_columns=list(vwap.columns),
    )
    can_buy_me = can_buy.reindex(index=me, columns=vwap.columns)

    neu_dir = config.NEU_FACTOR_BASE / NAMESPACE
    print("\n" + "=" * 94)
    print("华泰改进动量因子族 论文对齐验证（月频 IC，neu=行业市值中性化，vwap 次月收益）")
    print("=" * 94)
    for title, start, end in SEGMENTS:
        seg_me = me[(me >= start) & (me <= end)]
        print(f"\n【{title}】")
        print(f"  {'因子':<20}{'我 RankIC/IR':>20}{'我 PearsonIC/IR':>22}{'论文 IC/IR':>18}")
        for name, (p_ic, p_ir) in PAPER.items():
            path = neu_dir / f"{name}.parquet"
            if not path.exists():
                print(f"  {name:<20}{'面板缺失':>20}")
                continue
            fac = pd.read_parquet(path)
            fac.index = pd.to_datetime(fac.index)
            fac = fac.reindex(index=seg_me, columns=vwap.columns).where(
                can_buy_me.reindex(index=seg_me)
            )
            r_ic, r_ir = month_end_ic(fac, ret_m.reindex(seg_me), "spearman")
            p2_ic, p2_ir = month_end_ic(fac, ret_m.reindex(seg_me), "pearson")
            print(f"  {name:<20}{r_ic:>+11.2f}%/{r_ir:>5.2f}{p2_ic:>+15.2f}%/{p2_ir:>5.2f}"
                  f"{p_ic:>+11.2f}%/{p_ir:>5.2f}")
    print("\n  注：因子方向为负（反转），IC 应为负；IR 比率取绝对值，越大越强。")


if __name__ == "__main__":
    main()
