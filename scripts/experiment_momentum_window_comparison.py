"""
动量窗口对比实验 —— 20/40/60 日涨幅域,哪个窗口更好?
============================================================
对每个窗口长度,输出:
  1. 域的绝对收益(域内等权, 不排序) vs 基准 → 看域本身在反转年塌不塌
  2. 域内成长因子高/低档区分度(和原 double-sort 一样) → 看排序还有没有信号
  3. 2026H1 赢家圈住率 → 看域能不能抓住 2026 的赢家

用法: PYTHONPATH=. python scripts/experiment_momentum_window_comparison.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

import config

# ════════════════════ 参数区 ════════════════════
MOM_WINDOWS = [20, 40, 60]
MOM_TOP_PCT = 0.10
SAMPLE_STEP = 20
FWD_SHORT = 20
FWD_LONG = 60
CRASH_THRESHOLD = -0.40
EXP_START = "2014-01-01"
EXP_END = "2026-06-30"

_CXL = config.RAW_FACTOR_BASE / "cxl-dquant"
FACTORS = {
    "sue8":       _CXL / "npf_series" / "npf_mrq_sue8.parquet",
    "npf_pyoy":   _CXL / "npf_series" / "npf_pyoy_mrq.parquet",
    "roe_pyoy":   _CXL / "roe_series" / "roe_pyoy_mrq.parquet",
}

PERIODS = [
    ("2014-2017", "2014-01-01", "2017-12-31"),
    ("2018-2019", "2018-01-01", "2019-12-31"),
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2022-2023", "2022-01-01", "2023-12-31"),
    ("2024-2025", "2024-01-01", "2025-12-31"),
    ("2026H1",    "2026-01-01", "2026-06-30"),
]

WINNERS_2026H1 = [
    "688146.XSHG", "603256.XSHG", "002636.XSHE", "301396.XSHE", "603629.XSHG",
    "301526.XSHE", "603115.XSHG", "301362.XSHE", "603618.XSHG", "300489.XSHE",
    "688530.XSHG", "003036.XSHE", "688308.XSHG", "002980.XSHE", "002491.XSHE",
    "600396.XSHG", "300209.XSHE", "688766.XSHG", "300903.XSHE", "603773.XSHG",
]
# ════════════════════════════════════════════════


def main():
    vwap = pd.read_parquet(config.VWAP_PANEL_PATH)
    vwap.index = pd.to_datetime(vwap.index)
    vwap = vwap.loc["2013-06-01":EXP_END]
    P = vwap.values
    dates = vwap.index
    cols = vwap.columns
    n_dates = len(dates)
    logger.info(f"vwap 面板 {vwap.shape}")

    factor_panels = {}
    for name, fp in FACTORS.items():
        df = pd.read_parquet(fp)
        df.index = pd.to_datetime(df.index)
        factor_panels[name] = df

    # 基准收益(中证全指)
    bm = pd.read_parquet("/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir/benchmark.parquet")
    bm = bm.set_index("datetime")["000985.XSHG"]
    bm.index = pd.to_datetime(bm.index)

    max_mom = max(MOM_WINDOWS)
    start_i = int(np.searchsorted(dates, pd.Timestamp(EXP_START)))
    sample_is = list(range(max(start_i, max_mom), n_dates - FWD_LONG, SAMPLE_STEP))

    all_rows = []
    winner_hits = {}

    for i in sample_is:
        d = dates[i]
        p_now = P[i]
        fwd_s = P[i + FWD_SHORT] / p_now - 1 if i + FWD_SHORT < n_dates else np.full(len(cols), np.nan)
        if i + FWD_LONG < n_dates:
            fwd_l = P[i + FWD_LONG] / p_now - 1
            fwd_min = np.nanmin(P[i + 1:i + FWD_LONG + 1], axis=0) / p_now - 1
        else:
            fwd_l = np.full(len(cols), np.nan)
            fwd_min = np.full(len(cols), np.nan)

        bm_ret = bm.reindex(dates[i:i + FWD_LONG + 1]).dropna()
        bm_fwd60 = (bm_ret.iloc[-1] / bm_ret.iloc[0] - 1) if len(bm_ret) > 1 else np.nan

        for w in MOM_WINDOWS:
            ret_w = p_now / P[i - w] - 1
            valid = np.isfinite(ret_w)
            if valid.sum() < 500:
                continue
            cutoff = np.nanquantile(ret_w[valid], 1 - MOM_TOP_PCT)
            domain = valid & (ret_w >= cutoff)

            # 域整体收益(不排序,等权)
            domain_idx = np.where(domain)[0]
            domain_fwd60 = np.nanmean(fwd_l[domain_idx]) if len(domain_idx) > 0 else np.nan
            domain_fwd20 = np.nanmean(fwd_s[domain_idx]) if len(domain_idx) > 0 else np.nan
            domain_crash = (np.nanmean(fwd_min[domain_idx] < CRASH_THRESHOLD)
                            if np.isfinite(fwd_min[domain_idx]).any() else np.nan)

            all_rows.append(dict(
                date=d, mom_window=w, n_domain=len(domain_idx),
                domain_fwd20=domain_fwd20, domain_fwd60=domain_fwd60,
                domain_crash=domain_crash, bm_fwd60=bm_fwd60,
                domain_alpha60=domain_fwd60 - bm_fwd60,
            ))

            # 域内因子排序
            for fname, fpanel in factor_panels.items():
                pos = fpanel.index.searchsorted(d, side="right") - 1
                if pos < 0:
                    continue
                fvals = fpanel.iloc[pos].reindex(cols).values
                m = domain & np.isfinite(fvals)
                if m.sum() < 60:
                    continue
                fv = fvals[m]
                q_lo, q_hi = np.quantile(fv, [1 / 3, 2 / 3])
                hi = fv >= q_hi
                lo = fv <= q_lo
                m_idx = np.where(m)[0]

                for tag, grp in [("high", hi), ("low", lo)]:
                    idx = m_idx[grp]
                    all_rows.append(dict(
                        date=d, mom_window=w, factor=fname, group=tag, n=len(idx),
                        fwd20=np.nanmean(fwd_s[idx]),
                        fwd60=np.nanmean(fwd_l[idx]),
                        crash=np.nanmean(fwd_min[idx] < CRASH_THRESHOLD)
                              if np.isfinite(fwd_min[idx]).any() else np.nan,
                    ))

                # 赢家圈住(仅 2026)
                if d >= pd.Timestamp("2026-01-01"):
                    hi_codes = set(cols[m_idx[hi]])
                    for stock in WINNERS_2026H1:
                        if stock in hi_codes:
                            winner_hits.setdefault((w, stock), []).append((d.strftime("%m-%d"), fname))

    df = pd.DataFrame(all_rows)

    # ════════ 输出 1: 域的绝对收益 + 超额 ════════
    print("\n" + "=" * 110)
    print(f"动量窗口对比: 域(前{MOM_TOP_PCT:.0%})整体 60日远期收益 vs 基准")
    print("=" * 110)
    print(f"{'窗口':<6} {'期间':<12} {'样本':>5} {'域fwd60':>9} {'基准fwd60':>10} {'域超额':>9} {'域崩盘率':>9}")
    print("-" * 110)

    domain_df = df[df["factor"].isna()]
    for w in MOM_WINDOWS:
        wd = domain_df[domain_df["mom_window"] == w]
        for pname, ps, pe in PERIODS + [("全期", "2014-01-01", "2026-06-30")]:
            sub = wd[(wd["date"] >= ps) & (wd["date"] <= pe)]
            if sub.empty:
                continue
            print(f"{w:>3}d   {pname:<12} {len(sub):>5} {sub['domain_fwd60'].mean():>+8.1%}"
                  f" {sub['bm_fwd60'].mean():>+9.1%} {sub['domain_alpha60'].mean():>+8.1%}"
                  f" {sub['domain_crash'].mean():>8.1%}")
        print()

    # ════════ 输出 2: 域内因子区分度 ════════
    print("=" * 110)
    print("域内成长因子高/低档区分度(60日远期收益, 高-低差 + t值)")
    print("=" * 110)
    print(f"{'窗口':<6} {'因子':<12} {'期间':<12} {'高档':>8} {'低档':>8} {'高-低差':>9} {'t值':>6} {'崩盘差':>8}")
    print("-" * 110)

    for w in MOM_WINDOWS:
        for fname in FACTORS:
            for pname, ps, pe in PERIODS + [("全期", "2014-01-01", "2026-06-30")]:
                wd = df[(df["mom_window"] == w) & (df["factor"] == fname) & (df["date"] >= ps) & (df["date"] <= pe)]
                hi = wd[wd["group"] == "high"].set_index("date")
                lo = wd[wd["group"] == "low"].set_index("date")
                if hi.empty or lo.empty:
                    continue
                common = hi.index.intersection(lo.index)
                spread = (hi.loc[common, "fwd60"] - lo.loc[common, "fwd60"]).dropna()
                tstat = (spread.mean() / spread.std() * np.sqrt(len(spread))) if len(spread) > 3 and spread.std() > 0 else np.nan
                crash_diff = (hi.loc[common, "crash"] - lo.loc[common, "crash"]).dropna().mean()
                print(f"{w:>3}d   {fname:<12} {pname:<12} {hi['fwd60'].mean():>+7.1%}"
                      f" {lo['fwd60'].mean():>+7.1%} {spread.mean():>+8.1%} {tstat:>6.2f} {crash_diff:>+7.1%}")
        print()

    # ════════ 输出 3: 赢家圈住率 ════════
    print("=" * 110)
    print("2026H1 top20 赢家圈住率(域内×因子高档)")
    print("=" * 110)
    for w in MOM_WINDOWS:
        n_caught = sum(1 for s in WINNERS_2026H1 if (w, s) in winner_hits)
        print(f"  {w:>3}d 窗口: {n_caught}/{len(WINNERS_2026H1)} 圈住")

    out = config.ML_ROOT / "diagnostics" / "momentum_window_comparison.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    logger.info(f"明细已落盘 {out}")


if __name__ == "__main__":
    main()
