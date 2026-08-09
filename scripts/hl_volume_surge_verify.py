"""
高/低位放量因子族 论文对齐验证（国盛"量价淘金"系列④）
============================================================================
从 minute_hlvol superset 缓存直接组装 4 个日频因子 + 综合因子的【月频】面板，
按论文口径评估：全体A股（剔ST/停牌/新股）、每月月底、横截面市值中性化、
月度IC/RankIC/年化ICIR + 5分组多空（分组1-分组5）绩效，与论文图表17/24/28/30对照。

论文（全A，2014/01/01-2023/10/31，月度IC均值/年化ICIR/RankIC/年化RankICIR/
      多空年化收益/年化波动/IR/月度胜率/最大回撤）：
  日频_高位波动占比  -0.057/-2.98/-0.080/-3.99 | 20.32%/7.49%/2.71/79.49%/6.61%
  日频_低位波动占比  +0.047/+2.63/+0.073/+3.93 | 17.74%/7.03%/2.52/76.92%/5.75%
  日频_高波价格占比  -0.061/-2.48/-0.088/-3.77 | 23.26%/9.18%/2.53/74.36%/9.89%
  日频_低波价格占比  +0.045/+1.91/+0.074/+3.17 | 17.11%/9.04%/1.89/74.36%/8.69%
  综合因子           -0.066/-3.00/-0.091/-4.13 | 24.68%/8.52%/2.90/78.38%/7.70%
综合因子 = 月底 zscore(高位波动占比_neu) + zscore(高波价格占比_neu)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

import config
from alpha_shared.cleaning.mask_loader import load_filter_masks
from alpha_shared.evaluation.ic import compute_ic_series
from core.operators.minute_hlvol import HLVolReducer

START, END = "2014-01-01", "2023-10-31"
W, G = 20, 5
PAPER = {  # ic, icir, rankic, rankicir, ann_ret, ann_vol, ir, win, mdd
    "d_highpos_vol_ratio":   (-0.057, -2.98, -0.080, -3.99, 20.32, 7.49, 2.71, 79.49, 6.61),
    "d_lowpos_vol_ratio":    (+0.047, +2.63, +0.073, +3.93, 17.74, 7.03, 2.52, 76.92, 5.75),
    "d_highvol_price_ratio": (-0.061, -2.48, -0.088, -3.77, 23.26, 9.18, 2.53, 74.36, 9.89),
    "d_lowvol_price_ratio":  (+0.045, +1.91, +0.074, +3.17, 17.11, 9.04, 1.89, 74.36, 8.69),
    "combo":                 (-0.066, -3.00, -0.091, -4.13, 24.68, 8.52, 2.90, 78.38, 7.70),
}


def load_superset_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """minute_hlvol superset 缓存 → 宽表 (date × stock)：intraday_vol / close_d。"""
    cache = HLVolReducer(cache_key="hlv_v1").cache_dir()
    files = sorted(cache.glob("[0-9]*.parquet"))
    logger.info(f"加载 superset 缓存 {cache.name}: {len(files)} 只股 …")
    parts = [pd.read_parquet(p) for p in files]
    long = pd.concat(parts, ignore_index=True)
    vol = long.pivot(index="date", columns="order_book_id", values="intraday_vol").sort_index()
    close = long.pivot(index="date", columns="order_book_id", values="close_d").sort_index()
    return vol, close


def month_end_factors(vol: pd.DataFrame, close: pd.DataFrame, me: pd.DatetimeIndex):
    """逐月底取过去 W 个交易日（逐股自身交易日，NaN 剔除后不足 W → NaN）算 4 因子。"""
    k = W // G
    names = ["d_highpos_vol_ratio", "d_lowpos_vol_ratio",
             "d_highvol_price_ratio", "d_lowvol_price_ratio"]
    out = {n: pd.DataFrame(np.nan, index=me, columns=vol.columns) for n in names}
    dates = vol.index
    V, C = vol.to_numpy(), close.to_numpy()
    for d in me:
        t = dates.get_loc(d)
        lo = max(0, t - 3 * W)                       # 缓冲：停牌股取自身最近 W 个交易日
        v_blk, c_blk = V[lo:t + 1], C[lo:t + 1]
        both = np.isfinite(v_blk) & np.isfinite(c_blk)
        cnt = both.sum(axis=0)
        cols = np.where(cnt >= W)[0]
        if cols.size == 0:
            continue
        n_blk = v_blk.shape[0]
        vw = np.empty((cols.size, W)); cw = np.empty((cols.size, W))
        for j, ci in enumerate(cols):               # 取每股最近 W 个双有效交易日
            idx = np.where(both[:, ci])[0][-W:]
            vw[j], cw[j] = v_blk[idx, ci], c_blk[idx, ci]
        tot_v, tot_c = vw.mean(axis=1), cw.mean(axis=1)
        oc = np.argsort(cw, axis=1, kind="stable")   # 按收盘价排序
        ov = np.argsort(vw, axis=1, kind="stable")   # 按波动率排序
        take = np.take_along_axis
        out["d_highpos_vol_ratio"].loc[d, vol.columns[cols]] = take(vw, oc[:, -k:], 1).mean(1) / tot_v
        out["d_lowpos_vol_ratio"].loc[d, vol.columns[cols]] = take(vw, oc[:, :k], 1).mean(1) / tot_v
        out["d_highvol_price_ratio"].loc[d, vol.columns[cols]] = take(cw, ov[:, -k:], 1).mean(1) / tot_c
        out["d_lowvol_price_ratio"].loc[d, vol.columns[cols]] = take(cw, ov[:, :k], 1).mean(1) / tot_c
    return out


def neut_size(fac_me: pd.DataFrame, lmc_me: pd.DataFrame) -> pd.DataFrame:
    """月度横截面 市值中性化（对 log 总市值回归取残差，含截距）。"""
    out = pd.DataFrame(np.nan, index=fac_me.index, columns=fac_me.columns)
    for d in fac_me.index:
        df = pd.DataFrame({"y": fac_me.loc[d], "s": lmc_me.loc[d]}).dropna()
        if len(df) < 100:
            continue
        ys = df["y"] - df["y"].mean()
        ss = df["s"] - df["s"].mean()
        den = (ss * ss).sum()
        beta = (ys * ss).sum() / den if den > 0 else 0.0
        out.loc[d, df.index] = ys - beta * ss
    return out


def zscore_cs(fac_me: pd.DataFrame) -> pd.DataFrame:
    mu = fac_me.mean(axis=1)
    sd = fac_me.std(axis=1)
    return fac_me.sub(mu, axis=0).div(sd, axis=0)


def evaluate(fac_me: pd.DataFrame, ret_m: pd.DataFrame, n_bins: int = 5):
    """月频 IC/RankIC/年化ICIR + 5分组（分组1=因子最小）多空 分组1-分组5 绩效。"""
    ic = compute_ic_series(fac_me, ret_m, method="pearson").dropna()
    ric = compute_ic_series(fac_me, ret_m, method="spearman").dropna()
    rk = fac_me.rank(axis=1, pct=True)
    ls = []
    for d in fac_me.index:
        r, q = ret_m.loc[d], rk.loc[d]
        ok = r.notna() & q.notna()
        if ok.sum() < 100:
            continue
        g = np.minimum(np.floor(q[ok] * n_bins) + 1, n_bins)
        ls.append((d, r[ok][g == 1].mean() - r[ok][g == n_bins].mean()))
    ls = pd.Series(dict(ls)).sort_index()
    n = len(ls)
    ann_ret = (1 + ls).prod() ** (12 / n) - 1
    ann_vol = ls.std() * np.sqrt(12)
    nav = (1 + ls).cumprod()
    mdd = (1 - nav / nav.cummax()).max()
    return (ic.mean(), ic.mean() / ic.std() * np.sqrt(12),
            ric.mean(), ric.mean() / ric.std() * np.sqrt(12),
            ann_ret * 100, ann_vol * 100, ann_ret / ann_vol,
            (ls > 0).mean() * 100, mdd * 100)


def main():
    vol, close = load_superset_panels()
    stocks = list(vol.columns)
    dates = vol.index

    # 月底序列：因子月 = [2013-12, 2023-09]，收益月 = [2014-01, 2023-10]
    all_me = pd.Series(dates).groupby([dates.year, dates.month]).last().tolist()
    me_ext = pd.DatetimeIndex([d for d in all_me if "2013-12-01" <= str(d.date()) <= END])
    me = me_ext[:-1]

    logger.info(f"组装 4 因子月频面板（{len(me)} 个月底）…")
    fac = month_end_factors(vol, close, me)

    vwap = pd.read_parquet(config.VWAP_PANEL_PATH)
    vwap.index = pd.to_datetime(vwap.index)
    vwap = vwap.reindex(columns=stocks)
    vw_me = vwap.reindex(index=me_ext)
    ret_m = (vw_me.shift(-1) / vw_me - 1).loc[me]

    can_buy, _ = load_filter_masks(combo_mask_path=config.COMBO_MASK_PATH,
        new_stock_mask_path=config.NEW_STOCK_MASK_PATH, reindex_columns=stocks)
    pool = can_buy.reindex(index=me, columns=stocks)
    mc = pd.read_parquet(config.MARKET_CAP_PANEL_PATH)
    mc.index = pd.to_datetime(mc.index)
    lmc = np.log(mc.reindex(index=me, columns=stocks).where(lambda x: x > 0))

    neu = {}
    rows = []
    for n in ["d_highpos_vol_ratio", "d_lowpos_vol_ratio",
              "d_highvol_price_ratio", "d_lowvol_price_ratio"]:
        neu[n] = neut_size(fac[n].where(pool), lmc)
        rows.append((n, evaluate(neu[n], ret_m)))
    combo = zscore_cs(neu["d_highpos_vol_ratio"]) + zscore_cs(neu["d_highvol_price_ratio"])
    rows.append(("combo", evaluate(combo, ret_m)))

    print("\n" + "=" * 118)
    print("高/低位放量因子族 论文对齐验证（全A剔ST/停牌/新股，市值中性化，月频，vwap收益，2014/01-2023/10）")
    print("=" * 118)
    hdr = f"{'因子':<24}{'IC':>8}{'ICIR':>7}{'RankIC':>8}{'RkICIR':>7}{'多空年化':>9}{'年化波动':>9}{'IR':>6}{'胜率':>8}{'回撤':>7}"
    print(hdr)
    for name, got in rows:
        p = PAPER[name]
        print(f"{name:<24}{got[0]:>+8.3f}{got[1]:>+7.2f}{got[2]:>+8.3f}{got[3]:>+7.2f}"
              f"{got[4]:>8.2f}%{got[5]:>8.2f}%{got[6]:>6.2f}{got[7]:>7.2f}%{got[8]:>6.2f}%")
        print(f"{'  └ 论文':<23}{p[0]:>+8.3f}{p[1]:>+7.2f}{p[2]:>+8.3f}{p[3]:>+7.2f}"
              f"{p[4]:>8.2f}%{p[5]:>8.2f}%{p[6]:>6.2f}{p[7]:>7.2f}%{p[8]:>6.2f}%")


if __name__ == "__main__":
    main()
