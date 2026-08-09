"""
侧袋子组合归因 —— 回答 2021-2023 拖累是"缺陷"还是"保费"
============================================================
方法:
  1. 用与 build_quota_sidepocket_signals.py 完全一致的逻辑逐日重推侧袋名单
     (与 quota_sp10 信号文件前缀做一致性校验)
  2. 同一执行口径(每5个交易日调仓, T-1信号 T日vwap 等权链式)模拟4个子组合:
       sp_1_10    侧袋 1-10 席(quota_sp10 实际占用的席位)
       sp_11_20   侧袋 11-20 席(sp20 相对 sp10 的增量席位)
       base100    主模型 top100(基线组合)
       displaced  被侧袋顶掉的主模型尾部股(机会成本的直接对照)
  3. 分年对比绝对收益 + 相对基准(中证全指)超额
     → 侧袋自己亏钱/跑输指数 = 缺陷(可修) ; 赚钱但跑输主模型 = 保费(结构性)

产出:
  /nfs/ofs-prediction/peterzhenglinpeng/ml/analysis/sidepocket_attribution/
    daily_returns.parquet   4个子组合+基准 的日收益序列
    sp_membership.parquet   逐信号日侧袋完整名单(排序+复合分, B0 实验复用)
    summary.txt             分年归因表

用法: 仓库根 PYTHONPATH=. python scripts/attribute_sidepocket_subportfolio.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import config

# ════════════════════ 参数区(与生成脚本/回测保持一致) ════════════════════
BASE_RUN = "lgbm_a158_p27_shap_dq_pool2"
QUOTA_RUN = "quota_sp10"                      # 用于一致性校验
MOM_WINDOW = 60
MOM_TOP_PCT = 0.10
VETO_PCT = 1 / 3
TRADE_START = "2020-01-03"
TRADE_END = "2026-06-30"
REBALANCE_INTERVAL = 5
TOP_K = 100
OUT_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/ml/analysis/sidepocket_attribution")
# ══════════════════════════════════════════════════════════════════════

_CXL = config.RAW_FACTOR_BASE / "cxl-dquant"
SCORE_FACTORS = {
    "sue8": _CXL / "npf_series" / "npf_mrq_sue8.parquet",
    "npf_pyoy": _CXL / "npf_series" / "npf_pyoy_mrq.parquet",
    "roe_pyoy": _CXL / "roe_series" / "roe_pyoy_mrq.parquet",
}
VETO_FACTOR = _CXL / "roe_series" / "roe_mrq_new.parquet"


def derive_sidepocket(d: pd.Timestamp, vwap: pd.DataFrame, panels: dict, veto_panel: pd.DataFrame):
    """逐信号日重推侧袋有序名单(逻辑与生成脚本逐行一致), 返回 (codes, scores)"""
    P, dates, cols = vwap.values, vwap.index, vwap.columns
    pos = dates.searchsorted(d)
    if not (pos < len(dates) and dates[pos] == d and pos >= MOM_WINDOW):
        return [], []
    ret60 = P[pos] / P[pos - MOM_WINDOW] - 1
    valid = np.isfinite(ret60)
    if valid.sum() < 500:
        return [], []
    cutoff = np.nanquantile(ret60[valid], 1 - MOM_TOP_PCT)
    domain = valid & (ret60 >= cutoff)

    fvals = {}
    for name, fpanel in panels.items():
        r = fpanel.index.searchsorted(d, side="right") - 1
        fvals[name] = fpanel.iloc[r].reindex(cols).values if r >= 0 else np.full(len(cols), np.nan)
    r = veto_panel.index.searchsorted(d, side="right") - 1
    veto_vals = veto_panel.iloc[r].reindex(cols).values if r >= 0 else np.full(len(cols), np.nan)

    m = domain.copy()
    for v in fvals.values():
        m &= np.isfinite(v)
    m &= np.isfinite(veto_vals)
    if m.sum() < 30:
        return [], []
    idx = np.where(m)[0]
    pct = np.zeros(len(idx))
    for v in fvals.values():
        pct += pd.Series(v[idx]).rank(pct=True).values
    pct /= len(fvals)
    veto_pct = pd.Series(veto_vals[idx]).rank(pct=True).values
    keep = veto_pct >= VETO_PCT
    order = np.argsort(-pct[keep])
    return list(vwap.columns[idx[keep][order]]), list(pct[keep][order])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vwap = pd.read_parquet(config.VWAP_PANEL_PATH)
    vwap.index = pd.to_datetime(vwap.index)
    panels = {}
    for name, fp in SCORE_FACTORS.items():
        df = pd.read_parquet(fp)
        df.index = pd.to_datetime(df.index)
        panels[name] = df
    veto_panel = pd.read_parquet(VETO_FACTOR)
    veto_panel.index = pd.to_datetime(veto_panel.index)
    logger.info(f"vwap 面板 {vwap.shape} | 打分/否决因子加载完成")

    base_sig_dir = config.ML_PREDICTIONS_DIR / BASE_RUN / "signals"
    quota_sig_dir = config.ML_PREDICTIONS_DIR / QUOTA_RUN / "signals"

    # ── 调仓日程: 交易日历上每5天, T日执行, 用 T-1 信号 ──────────────
    tdays = vwap.index[(vwap.index >= TRADE_START) & (vwap.index <= TRADE_END)]
    reb_days = tdays[::REBALANCE_INTERVAL]
    logger.info(f"交易日 {len(tdays)} 个, 调仓日 {len(reb_days)} 个")

    # ── 逐调仓日构建4个子组合持仓 ───────────────────────────────────
    holdings, memb_rows, n_mismatch, overlap_cnt, fill_cnt = {}, [], 0, [], []
    for t in reb_days:
        sig_d = vwap.index[vwap.index.searchsorted(t) - 1]  # T-1 信号日
        sf = base_sig_dir / f"{sig_d.date()}.txt"
        if not sf.exists():
            logger.warning(f"缺基线信号 {sf.name}, 跳过调仓日 {t.date()}")
            continue
        base_codes = [l.split("_", 1)[1] for l in sf.read_text().splitlines() if l.strip()]
        sp_codes, sp_scores = derive_sidepocket(sig_d, vwap, panels, veto_panel)

        # 与 quota_sp10 信号文件做一致性校验(前 min(10,len) 行应完全一致)
        qf = quota_sig_dir / f"{sig_d.date()}.txt"
        if qf.exists():
            head = [l.split("_", 1)[1] for l in qf.read_text().splitlines()[: min(10, len(sp_codes))]]
            if head != sp_codes[:10][: len(head)]:
                n_mismatch += 1

        sp10, sp1120 = sp_codes[:10], sp_codes[10:20]
        base100 = base_codes[:TOP_K]
        sp_set = set(sp10)
        merged100 = (sp10 + [c for c in base_codes if c not in sp_set])[:TOP_K]
        displaced = [c for c in base100 if c not in set(merged100)]

        holdings[t] = dict(sp_1_10=sp10, sp_11_20=sp1120, base100=base100, displaced=displaced)
        overlap_cnt.append(len(sp_set & set(base100)))
        fill_cnt.append(len(sp10))
        for rk, (c, s) in enumerate(zip(sp_codes, sp_scores), 1):
            memb_rows.append((sig_d, t, rk, c, s))

    logger.info(f"一致性校验: {n_mismatch} 个信号日与 quota_sp10 文件不一致")
    logger.info(f"侧袋平均满席 {np.mean(fill_cnt):.1f}/10 | 与主模型top100平均重叠 {np.mean(overlap_cnt):.1f} 只")

    # ── 日收益链: 持仓自调仓日T的次日起计收益(T日vwap买入) ──────────
    R = vwap.pct_change()
    reb_list = sorted(holdings)
    port_names = ["sp_1_10", "sp_11_20", "base100", "displaced"]
    daily = {p: pd.Series(0.0, index=tdays) for p in port_names}
    for k, t in enumerate(reb_list):
        t_next = reb_list[k + 1] if k + 1 < len(reb_list) else tdays[-1]
        seg = tdays[(tdays > t) & (tdays <= t_next)]
        if len(seg) == 0:
            continue
        for p in port_names:
            cols = [c for c in holdings[t][p] if c in vwap.columns]
            if cols:
                daily[p].loc[seg] = R.loc[seg, cols].mean(axis=1, skipna=True).fillna(0.0)

    bm = pd.read_parquet("/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir/benchmark.parquet")
    bm = bm.set_index("datetime")["000985.XSHG"]
    bm.index = pd.to_datetime(bm.index)
    daily["benchmark"] = bm.pct_change().reindex(tdays).fillna(0.0)

    df = pd.DataFrame(daily)
    df.to_parquet(OUT_DIR / "daily_returns.parquet")
    memb = pd.DataFrame(memb_rows, columns=["signal_date", "trade_date", "rank", "code", "score"])
    memb.to_parquet(OUT_DIR / "sp_membership.parquet")

    # ── 分年归因表 ──────────────────────────────────────────────────
    yearly = (1 + df).groupby(df.index.year).prod() - 1
    lines = ["═" * 100,
             "分年子组合收益(等权链式, interval5/shift1/vwap, 未计费用)",
             "═" * 100,
             f"{'年份':<6}{'侧袋1-10':>10}{'侧袋11-20':>11}{'主模型100':>11}{'被顶掉股':>10}{'基准':>9}"
             f"{'│ 袋1-10超额':>13}{'袋超主模型':>11}{'袋超被顶':>10}"]
    for y, r in yearly.iterrows():
        lines.append(f"{y:<6}{r['sp_1_10']:>+9.1%}{r['sp_11_20']:>+10.1%}{r['base100']:>+10.1%}"
                     f"{r['displaced']:>+9.1%}{r['benchmark']:>+8.1%}"
                     f"│{r['sp_1_10'] - r['benchmark']:>+11.1%}{r['sp_1_10'] - r['base100']:>+10.1%}"
                     f"{r['sp_1_10'] - r['displaced']:>+9.1%}")
    total = (1 + df).prod() - 1
    lines.append("-" * 100)
    lines.append(f"{'全期':<6}{total['sp_1_10']:>+9.1%}{total['sp_11_20']:>+10.1%}{total['base100']:>+10.1%}"
                 f"{total['displaced']:>+9.1%}{total['benchmark']:>+8.1%}")

    # ── 侧袋股票级持有期统计(缺陷探测: 崩盘率) ──────────────────────
    lines += ["", "═" * 100, "侧袋1-10 股票级 5日持有期收益统计", "═" * 100,
              f"{'年份':<6}{'样本数':>8}{'均值':>9}{'中位数':>9}{'跌穿-15%占比':>13}{'涨超+15%占比':>13}"]
    rows = []
    Pv, dates_all = vwap.values, vwap.index
    for k, t in enumerate(reb_list):
        t_next = reb_list[k + 1] if k + 1 < len(reb_list) else None
        if t_next is None:
            continue
        i0, i1 = dates_all.searchsorted(t), dates_all.searchsorted(t_next)
        for c in holdings[t]["sp_1_10"]:
            j = vwap.columns.get_loc(c) if c in vwap.columns else None
            if j is not None and np.isfinite(Pv[i0, j]) and np.isfinite(Pv[i1, j]):
                rows.append((t.year, Pv[i1, j] / Pv[i0, j] - 1))
    hp = pd.DataFrame(rows, columns=["year", "ret"])
    for y, g in hp.groupby("year"):
        lines.append(f"{y:<6}{len(g):>8}{g['ret'].mean():>+8.1%}{g['ret'].median():>+8.1%}"
                     f"{(g['ret'] < -0.15).mean():>12.1%}{(g['ret'] > 0.15).mean():>12.1%}")

    report = "\n".join(lines)
    print(report)
    (OUT_DIR / "summary.txt").write_text(report)
    logger.info(f"产出已写入 {OUT_DIR}")


if __name__ == "__main__":
    main()
