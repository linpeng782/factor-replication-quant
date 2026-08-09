"""
条件双重排序实验 —— 基本面因子能否区分"晚期动量"的续涨与崩盘？
============================================================
验证因子分域的地基假设：在"60日涨幅 top 10%"(晚期动量域)内，
基本面因子高低两组的远期收益/崩盘率是否有系统性差异。

实验设计：
  ① 每 20 个交易日采样一次(近似月频)
  ② 每个采样日：domain = 60日涨幅截面前 10% 的股票(晚期动量域)
  ③ 域内按基本面因子分三档(高/中/低,各1/3)
  ④ 对比高低两组：未来 20d 收益 / 未来 60d 收益 / 崩盘率(60d 内跌破入场价 40%)
  ⑤ 分年代汇总：2014-2017(训练段) / 2018-2019 / 2020-2023 / 2024-2026
  ⑥ 日期级 spread 序列做 t 检验(避免截面相关性夸大显著性)
  ⑦ 结尾:2026H1 top20 赢家有几只落在"域内×因子高档"(圈住率)

三种结果 → 三种行动：
  高组全期显著更好      → 先验成立,推进 quota/SHAP手术分域
  只在 2024-2026 好     → 先验 regime 依赖,分域需叠时间开关
  区分不开             → 方向毙掉,避免单案例过拟合

用法：改参数区后在仓库根
  PYTHONPATH=. python scripts/experiment_conditional_double_sort.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

import config

# ════════════════════ 参数区（手动改这里） ════════════════════
MOM_WINDOW = 60          # 动量窗口(交易日)
MOM_TOP_PCT = 0.10       # 晚期动量域 = 60日涨幅截面前 10%
SAMPLE_STEP = 20         # 采样间隔(交易日,20≈月频)
FWD_SHORT = 20           # 短远期窗口
FWD_LONG = 60            # 长远期窗口
CRASH_THRESHOLD = -0.40  # 崩盘定义:60d 内最低价较入场价跌幅超 40%
EXP_START = "2014-01-01" # 实验起点(cxl 8期因子 warmup 后)
EXP_END = "2026-06-30"

# 条件变量:基本面因子(名称 → parquet 路径)
_CXL = config.RAW_FACTOR_BASE / "cxl-dquant"
FACTORS = {
    "sue8(盈余惊喜)":       _CXL / "npf_series" / "npf_mrq_sue8.parquet",
    "npf_pyoy(净利同比)":    _CXL / "npf_series" / "npf_pyoy_mrq.parquet",
    "roe_new(ROE水平)":     _CXL / "roe_series" / "roe_mrq_new.parquet",
    "roe_pyoy(ROE改善)":    _CXL / "roe_series" / "roe_pyoy_mrq.parquet",
    "roic_rnk8(质量排名)":   _CXL / "roic_series" / "roic_ttm_all_rnk8.parquet",
    "roic_std8(质量稳定)":   _CXL / "roic_series" / "roic_ttm_dev_std8.parquet",
}

# 年代分桶
PERIODS = [
    ("2014-2017(训练段)", "2014-01-01", "2017-12-31"),
    ("2018-2019",         "2018-01-01", "2019-12-31"),
    ("2020-2023",         "2020-01-01", "2023-12-31"),
    ("2024-2026(当前)",   "2024-01-01", "2026-06-30"),
]

# 2026H1 top20 赢家(圈住率检查用,来自 diagnose_missed_winners_shap 2026H1 运行)
WINNERS_2026H1 = [
    "688146.XSHG", "603256.XSHG", "002636.XSHE", "301396.XSHE", "603629.XSHG",
    "301526.XSHE", "603115.XSHG", "301362.XSHE", "603618.XSHG", "300489.XSHE",
    "688530.XSHG", "003036.XSHE", "688308.XSHG", "002980.XSHE", "002491.XSHE",
    "600396.XSHG", "300209.XSHE", "688766.XSHG", "300903.XSHE", "603773.XSHG",
]
# ══════════════════════════════════════════════════════════════

pd.set_option("display.width", 240, "display.max_rows", 300)


def main():
    # ── 数据加载 ──────────────────────────────────────────
    vwap = pd.read_parquet(config.VWAP_PANEL_PATH)
    vwap.index = pd.to_datetime(vwap.index)
    vwap = vwap.loc["2013-06-01":EXP_END]  # 留 60d 动量 warmup
    logger.info(f"vwap 面板 {vwap.shape} | {vwap.index[0].date()}~{vwap.index[-1].date()}")

    factor_panels = {}
    for name, fp in FACTORS.items():
        df = pd.read_parquet(fp)
        df.index = pd.to_datetime(df.index)
        factor_panels[name] = df
        logger.info(f"因子 {name}: {df.shape} | {df.index[0].date()}~{df.index[-1].date()}")

    # ── 采样日 + 逐日双重排序 ─────────────────────────────
    P = vwap.values
    dates = vwap.index
    n_dates = len(dates)
    start_i = int(np.searchsorted(dates, pd.Timestamp(EXP_START)))
    sample_is = list(range(max(start_i, MOM_WINDOW), n_dates - 1, SAMPLE_STEP))

    rows = []          # 日期级分组统计
    winner_hits = {}   # 赢家圈住记录: code → [(date, factor, tercile)]

    for i in sample_is:
        d = dates[i]
        p_now = P[i]
        p_past = P[i - MOM_WINDOW]
        ret60 = p_now / p_past - 1
        valid = np.isfinite(ret60)
        if valid.sum() < 500:
            continue

        # 晚期动量域 = ret60 截面前 10%
        cutoff = np.nanquantile(ret60[valid], 1 - MOM_TOP_PCT)
        domain = valid & (ret60 >= cutoff)

        # 远期收益(短/长) + 崩盘(60d 内最低点较入场价)
        fwd_s = P[i + FWD_SHORT] / p_now - 1 if i + FWD_SHORT < n_dates else np.full(P.shape[1], np.nan)
        if i + FWD_LONG < n_dates:
            fwd_l = P[i + FWD_LONG] / p_now - 1
            fwd_min = np.nanmin(P[i + 1:i + FWD_LONG + 1], axis=0) / p_now - 1
        else:
            fwd_l = np.full(P.shape[1], np.nan)
            fwd_min = np.full(P.shape[1], np.nan)

        for fname, fpanel in factor_panels.items():
            # 因子值取采样日或之前最近一行(PIT)
            pos = fpanel.index.searchsorted(d, side="right") - 1
            if pos < 0:
                continue
            fvals = fpanel.iloc[pos].reindex(vwap.columns).values

            m = domain & np.isfinite(fvals)
            if m.sum() < 60:
                continue
            fv = fvals[m]
            codes_m = vwap.columns[m]
            q_lo, q_hi = np.quantile(fv, [1 / 3, 2 / 3])
            hi = fv >= q_hi
            lo = fv <= q_lo

            for tag, grp in [("high", hi), ("low", lo)]:
                idx = np.where(m)[0][grp]
                rows.append(dict(
                    date=d, factor=fname, group=tag, n=len(idx),
                    fwd20=np.nanmean(fwd_s[idx]),
                    fwd60=np.nanmean(fwd_l[idx]),
                    crash=np.nanmean(fwd_min[idx] < CRASH_THRESHOLD)
                          if np.isfinite(fwd_min[idx]).any() else np.nan,
                ))

            # 赢家圈住记录(仅 2026 年采样日)
            if d >= pd.Timestamp("2026-01-01"):
                hi_codes = set(codes_m[hi])
                for w in WINNERS_2026H1:
                    if w in hi_codes:
                        winner_hits.setdefault(w, []).append((d.strftime("%m-%d"), fname))

    df = pd.DataFrame(rows)
    logger.info(f"共 {df['date'].nunique()} 个采样日 × {len(FACTORS)} 因子")

    # ── 分年代汇总 + t 检验 ───────────────────────────────
    print("\n" + "=" * 130)
    print(f"条件双重排序:晚期动量域(60日涨幅前 {MOM_TOP_PCT:.0%})内,基本面因子高/低档(各1/3)的远期表现")
    print(f"崩盘定义: 未来 {FWD_LONG}d 内最低价较入场价跌幅超 {-CRASH_THRESHOLD:.0%}")
    print("=" * 130)

    for pname, ps, pe in PERIODS + [("全期 2014-2026", "2014-01-01", "2026-06-30")]:
        sub = df[(df["date"] >= ps) & (df["date"] <= pe)]
        if sub.empty:
            continue
        print(f"\n◆ {pname}")
        print(f"  {'因子':<20} {'组':>5} {'均只数':>6} {'fwd20':>8} {'fwd60':>8} {'崩盘率':>8} | {'60d高-低差':>10} {'t值':>6} {'崩盘差':>8}")
        print("  " + "-" * 110)
        for fname in FACTORS:
            fs = sub[sub["factor"] == fname]
            hi = fs[fs["group"] == "high"].set_index("date")
            lo = fs[fs["group"] == "low"].set_index("date")
            if hi.empty or lo.empty:
                continue
            common = hi.index.intersection(lo.index)
            spread = (hi.loc[common, "fwd60"] - lo.loc[common, "fwd60"]).dropna()
            tstat = (spread.mean() / spread.std() * np.sqrt(len(spread))) if len(spread) > 3 and spread.std() > 0 else np.nan
            crash_diff = (hi.loc[common, "crash"] - lo.loc[common, "crash"]).dropna().mean()
            for tag, g in [("high", hi), ("low", lo)]:
                extra = ""
                if tag == "high":
                    extra = f" | {spread.mean():>+10.2%} {tstat:>6.2f} {crash_diff:>+8.1%}"
                print(f"  {fname:<20} {tag:>5} {g['n'].mean():>6.0f} {g['fwd20'].mean():>+8.2%}"
                      f" {g['fwd60'].mean():>+8.2%} {g['crash'].mean():>8.1%}{extra}")

    # ── 2026H1 赢家圈住率 ─────────────────────────────────
    print("\n" + "=" * 130)
    print(f"2026H1 top20 赢家圈住检查(2026 年采样日落在\"域内×因子高档\"的记录)")
    print("=" * 130)
    for w in WINNERS_2026H1:
        hits = winner_hits.get(w, [])
        if hits:
            fac_count = pd.Series([h[1] for h in hits]).value_counts()
            summary = "  ".join(f"{f.split('(')[0]}×{c}" for f, c in fac_count.items())
            print(f"  {w}  圈住 {len(hits)} 次: {summary}")
        else:
            print(f"  {w}  从未圈住")
    n_caught = sum(1 for w in WINNERS_2026H1 if w in winner_hits)
    print(f"\n  圈住率: {n_caught}/{len(WINNERS_2026H1)}")

    # ── 落盘 ──────────────────────────────────────────────
    out = config.ML_ROOT / "diagnostics" / "conditional_double_sort.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    logger.info(f"日期级明细已落盘 {out}")


if __name__ == "__main__":
    main()
