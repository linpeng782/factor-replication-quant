"""
联合动量因子族 论文对齐验证（跨因子一致性检验）
============================================================================
复现论文全部因子 ICM/VICM/VICR/CMC（行业 signal）+ MCMC（中证全指 signal），
每个出 raw vs 行业市值中性化两版，并排对比论文。

【检验逻辑】若 neu 版【全部】≈论文、raw 版【全部】系统性偏低 → 跨因子一致地坐实
"论文 RankIC/ICIR 是行业市值中性化后口径"（单因子可能巧合，五因子全中则是规律）。

论文(月频RankIC% / 年化ICIR):
  ICM 5.59/3.85   VICM 5.90/3.90   VICR -5.93/-3.55   CMC 6.02/3.86   MCMC 6.40/4.00
口径：月频、vwap收益、股票池剔ST/停牌/新股、区间2010-2023（行业指数源2010-06，因子2010-07起有效）。
"""
from __future__ import annotations
import numpy as np, pandas as pd, rqdatac
from loguru import logger
import config
from alpha_shared.cleaning.mask_loader import load_filter_masks
from alpha_shared.evaluation.ic import compute_ic_series
from scripts.comomentum_cmc import build_ind_ret_panel, WINDOW, N_MOM, N_REV
from core.operators.industry_co_momentum import compute_factor

START, END = "2010-01-01", "2023-12-31"
PAPER = {"ICM": (5.59, 3.85), "VICM": (5.90, 3.90), "VICR": (-5.93, -3.55),
         "CMC": (6.02, 3.86), "MCMC": (6.40, 4.00)}


def factors_at(t, ret_np, vol_np, sig_np):
    """基于给定 signal 收益(行业 或 市场)，返回 ICM/VICM/VICR/CMC（算法委托算子 compute_factor）。"""
    return {n: compute_factor(ret_np, vol_np, sig_np, t, WINDOW, N_MOM, N_REV, n)
            for n in ("ICM", "VICM", "VICR", "CMC")}


def neut_both(fac_me, ind_me, lmc_me):
    """月度横截面 行业+市值 中性化（FWL：行业去均值 + 市值过原点回归残差）。"""
    out = pd.DataFrame(np.nan, index=fac_me.index, columns=fac_me.columns)
    for d in fac_me.index:
        df = pd.DataFrame({"y": fac_me.loc[d], "ind": ind_me.loc[d], "s": lmc_me.loc[d]}).dropna()
        if len(df) < 100:
            continue
        ys = df["y"] - df.groupby("ind")["y"].transform("mean")
        ss = df["s"] - df.groupby("ind")["s"].transform("mean")
        den = (ss * ss).sum()
        beta = (ys * ss).sum() / den if den > 0 else 0.0
        out.loc[d, df.index] = ys - beta * ss
    return out


def rankic(fac_me, ret_me):
    ic = compute_ic_series(fac_me, ret_me, method="spearman").dropna()
    return ic.mean() * 100, ic.mean() / ic.std() * np.sqrt(12)


def main():
    from core.producers.alpha158.adjusted_panels import load_adjusted_panels
    logger.info(f"加载后复权面板 {START}~{END} …")
    p = load_adjusted_panels(start=START, end=END, fields=("close", "volume"))
    close, volume = p["close"], p["volume"]
    ret = close.pct_change(fill_method=None)
    stocks, dates = list(close.columns), ret.index
    ind_ret = build_ind_ret_panel(ret.index, stocks)

    # 中证全指收益 broadcast → 全市场共享的市场 signal 面板 (T×N)
    rqdatac.init()
    mkt = rqdatac.get_price("000985.XSHG", start_date=START, end_date=END, frequency="1d", fields=["close"])
    mc_close = mkt["close"]
    if mc_close.index.nlevels > 1:
        mc_close = mc_close.reset_index(level=0, drop=True)
    mc_close.index = pd.to_datetime(mc_close.index)
    mkt_ret = mc_close.pct_change().reindex(dates)
    mkt_panel = pd.DataFrame(np.broadcast_to(mkt_ret.values[:, None], (len(dates), len(stocks))),
                             index=dates, columns=stocks)

    ret_np, vol_np, ind_np, mkt_np = ret.values, volume.values, ind_ret.values, mkt_panel.values
    month_ends = pd.Series(dates).groupby([dates.year, dates.month]).last().tolist()
    me = pd.DatetimeIndex([d for d in month_ends if dates.get_loc(d) >= WINDOW - 1])

    vwap = pd.read_parquet(config.VWAP_PANEL_PATH); vwap.index = pd.to_datetime(vwap.index)
    vwap = vwap.reindex(index=dates, columns=stocks)
    ret_m = vwap.loc[me].shift(-1) / vwap.loc[me] - 1

    can_buy_mask, _ = load_filter_masks(combo_mask_path=config.COMBO_MASK_PATH,
        new_stock_mask_path=config.NEW_STOCK_MASK_PATH, reindex_columns=stocks)
    pre_me = can_buy_mask.reindex(index=me, columns=stocks)
    ind_me = pd.read_parquet(config.INDUSTRY_PANEL_ZX_PATH); ind_me.index = pd.to_datetime(ind_me.index)
    ind_me = ind_me.reindex(index=me, columns=stocks)
    mc = pd.read_parquet(config.MARKET_CAP_PANEL_PATH); mc.index = pd.to_datetime(mc.index)
    lmc_me = np.log(mc.reindex(index=me, columns=stocks).where(lambda x: x > 0))

    names = ["ICM", "VICM", "VICR", "CMC", "MCMC"]
    panels = {n: pd.DataFrame(np.nan, index=me, columns=stocks) for n in names}
    logger.info(f"逐月组装五因子面板（{len(me)} 月）…")
    for d in me:
        t = dates.get_loc(d)
        fi = factors_at(t, ret_np, vol_np, ind_np)
        fm = factors_at(t, ret_np, vol_np, mkt_np)
        for n in ["ICM", "VICM", "VICR", "CMC"]:
            panels[n].loc[d] = fi[n]
        panels["MCMC"].loc[d] = fm["CMC"]

    print("\n" + "=" * 76)
    print("联合动量因子族 论文对齐验证（月频 RankIC，vwap收益，2010-2023）")
    print("=" * 76)
    print(f"  {'因子':<6}{'raw (RankIC/ICIR)':>20}{'neu (RankIC/ICIR)':>22}{'论文':>16}")
    for n in names:
        raw = panels[n].where(pre_me)
        neu = neut_both(raw, ind_me, lmc_me)
        rm, ri = rankic(raw, ret_m)
        nm, ni = rankic(neu, ret_m)
        pr, pi = PAPER[n]
        print(f"  {n:<6}{rm:>+9.2f}%/{ri:>+5.2f}{nm:>+14.2f}%/{ni:>+5.2f}{pr:>+10.2f}%/{pi:>+5.2f}")
    print("\n  判断：neu 列全部≈论文 且 raw 列全部偏低 → 跨因子坐实论文用了行业市值中性化。")


if __name__ == "__main__":
    main()
