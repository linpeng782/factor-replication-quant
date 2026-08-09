"""
错过赢家 SHAP 月度诊断 —— 模型风格洞常规监控工具
============================================================
把"模型为什么错过大牛股"从玄学变成因子级可审计叙事。

功能（一次运行全自动）：
  ① 在诊断期内找出全市场涨幅最大的 TOP_N_WINNERS 只股票
  ② 用模型预测面板判定每只是"抓住"还是"错过"（进过 top100 天数）
  ③ 对错过的前 N_DETAIL 只逐一做 SHAP 解剖：
     - 排名轨迹（全期逐周采样）
     - 最被嫌弃日的拖累因子 top10 + 因子族汇总
     - 拖累因子族的逐日演化轨迹
  ④ 落盘明细 parquet（audit trail，可回溯任意月份的诊断结论）

用法：改下方参数区，然后在仓库根：
  source /nfs/ofs-prediction/peterzhenglinpeng-code/peterdidi/bin/activate
  PYTHONPATH=. python scripts/diagnose_missed_winners_shap.py

建议每月跑一次（PERIOD 设为上个月），监控模型风格洞是否扩大：
  - "错过赢家占比"上升 → 风格洞在扩大（regime 持续偏离训练段先验）
  - 拖累因子族稳定为"价格反转+p27微结构" → 晚期动量风格洞（688146 型）
  - 若出现新的拖累族 → 新型风格洞，需要单独研究

案例沉淀：2026H1 诊断发现 688146(+763%) 被 ROC60/VWAP0/eruption 三族
接力绞杀（详见 docs/2026h1_miss_diagnosis_and_fundamental_momentum.md）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from loguru import logger

import config
from ml_core.features import HasFactorPolicy, build_feature_matrix
from ml_core.model import LGBMAdapter
from ml_core.scaling import WholeSetRobustZ
from ml_core.universe import build_universe

# ════════════════════ 参数区（手动改这里） ════════════════════
MODEL_RUN = "lgbm_a158_p27_shap_dq_pool2"   # 诊断的模型
PERIOD_START = "2026-01-01"                  # 诊断期起点（找赢家 + 算收益）
PERIOD_END = "2026-06-30"                    # 诊断期终点
TOP_N_WINNERS = 20                           # 赢家池大小（按期间涨幅取前 N）
N_DETAIL = 3                                 # 深挖前 N 只"错过的"赢家（SHAP 解剖）
CAUGHT_TOP100_DAYS = 3                       # 进 top100 天数 >= 该值 → 判定"抓住"
MIN_DATA_COVERAGE = 0.8                      # 期间行情覆盖率下限（排除次新/长停牌）
N_SHAP_SAMPLES = 10                          # 每只股票 SHAP 采样日数（均匀取样）
OUT_DIR = config.ML_ROOT / "diagnostics" / "missed_winners_shap"  # 明细落盘目录
# ══════════════════════════════════════════════════════════════

GLOSS = {
    "eruption_followup_ratio": "喷发后跟风拥挤度：高=散户追高→看空",
    "ridge_minute_return": "量岭分钟收益和：高=放量持续拉升→看空",
    "ridge_minute_count": "量岭分钟数：高=全天持续放量→看空",
    "valley_relative_vwap": "量谷VWAP/日VWAP：低=缩量时价格塌→弱",
    "valley_weighted_quantile": "量谷价格分位：低=缩量时价格在日内低位→弱",
    "peak_interval_std": "量峰间隔std：高=脉冲式放量不规律→看空",
    "peak_ridge_turnover_ratio": "量峰/量岭成交额：低=以持续放量为主→看空",
    "peak_minute_count": "量峰分钟数：低=缺少瞬时脉冲→看空",
    "eruption_turnover_sensitivity": "喷发对下一分钟成交额敏感度",
    "eruption_turnover_corr": "喷发与成交额相关性",
    "valley_ridge_price_ratio__mp10": "量谷/量岭价格比：高=缩量时价格不塌→强",
    "peak_interval_skew": "量峰间隔偏度",
    "ROC60": "60日涨幅反转：低=60日涨幅大→看空",
    "MA60": "MA60/今收：低=价格远超均线→反转看空",
    "QTLU60": "60日80分位价/今收：低=突破历史高位→反转看空",
    "QTLU20": "20日80分位价/今收",
    "QTLD60": "60日20分位价/今收",
    "MIN5": "5日最低/今收：低=短期急涨→反转看空",
    "MIN30": "30日最低/今收：低=一个月内涨幅大→反转看空",
    "MIN60": "60日最低/今收：低=三个月涨幅大",
    "MAX60": "60日最高/今收：大=从高点回落多→动量看空",
    "RSV5": "5日随机值", "RSV60": "60日随机值：高=接近区间顶部",
    "IMXD30": "30日(最高点位置-最低点位置)",
    "IMXD60": "60日(最高点位置-最低点位置)",
    "IMIN5": "5日最低点距今天数", "IMIN20": "20日最低点距今天数",
    "IMIN30": "30日最低点距今天数",
    "IMIN60": "60日最低点距今天数：低=刚创新低(超跌)→模型偏爱",
    "STD5": "5日波动率", "STD10": "10日波动率", "STD20": "20日波动率",
    "STD30": "30日波动率", "STD60": "60日波动率",
    "VSTD5": "5日成交量波动率", "VSTD10": "10日成交量波动率",
    "VSTD20": "20日成交量波动率", "VSTD60": "60日成交量波动率",
    "WVMA20": "20日加权成交量均线比", "WVMA30": "30日加权成交量均线比",
    "WVMA60": "60日加权成交量均线比",
    "BETA5": "5日Beta", "BETA30": "30日Beta", "BETA60": "60日Beta",
    "RESI5": "5日回归残差", "RESI10": "10日回归残差",
    "RESI30": "30日回归残差", "RESI60": "60日回归残差",
    "RSQR10": "10日R²",
    "CORR5": "5日量价相关", "CORR30": "30日量价相关", "CORR60": "60日量价相关",
    "CORD5": "5日量价相关(延迟)", "CORD10": "10日量价相关(延迟)",
    "CORD30": "30日量价相关(延迟)", "CORD60": "60日量价相关(延迟)",
    "CNTN30": "30日下跌天数占比", "CNTN60": "60日下跌天数占比",
    "SUMN20": "20日下跌日成交额占比", "SUMN60": "60日下跌日成交额占比",
    "VWAP0": "日VWAP/今收：低=收盘强于均价(尾盘抢筹)→看空",
    "VMA60": "60日成交量均线/今成交量",
    "KLOW": "K线下影线占比", "KUP": "K线上影线占比",
    "CNTP60": "60日上涨天数占比",
    "SUMD20": "20日涨跌日成交额差异比", "SUMD60": "60日涨跌日成交额差异比",
    "RANK60": "60日成交量排名分位", "VSTD30": "30日成交量波动率",
    "VMA20": "20日成交量均线比",
}

pd.set_option("display.width", 240, "display.max_rows", 300)


def family(f: str) -> str:
    """因子族归类（与 diagnose_shap_factor_anatomy.py 一致）。"""
    if f.split("_")[0].lower() in {"eruption", "peak", "valley", "ridge"}:
        return "p27微结构"
    p = f.rstrip("0123456789")
    if p in {"MAX", "MIN", "QTLU", "QTLD", "RSV", "IMAX", "IMIN", "IMXD", "RANK", "ROC", "MA"}:
        return "价格位置/反转"
    if p in {"STD", "VSTD", "WVMA", "BETA", "RESI", "RSQR"}:
        return "波动率"
    if p in {"CORR", "CORD"}:
        return "量价相关"
    if p in {"CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD"}:
        return "涨跌计数"
    if p in {"VMA", "VWAP", "KLOW", "KUP", "KMID", "KSFT", "KLEN"}:
        return "K线/成交量"
    return "其他"


def find_winners_and_classify() -> pd.DataFrame:
    """① 找期间赢家 + ② 用预测面板判定抓住/错过。"""
    vwap = pd.read_parquet(config.VWAP_PANEL_PATH)
    vwap.index = pd.to_datetime(vwap.index)
    period = vwap.loc[PERIOD_START:PERIOD_END]

    # 期间收益（首个有效价 → 末个有效价），要求覆盖率达标
    valid_days = period.notna().sum()
    coverage = valid_days / len(period)
    first_price = period.bfill().iloc[0]
    last_price = period.ffill().iloc[-1]
    ret = (last_price / first_price - 1).where(coverage >= MIN_DATA_COVERAGE)
    winners = ret.nlargest(TOP_N_WINNERS)

    # 模型预测面板 → 每只赢家的排名统计
    pred = pd.read_parquet(config.ML_PREDICTIONS_DIR / MODEL_RUN / "pred_panel_live.parquet")
    pred.index = pd.to_datetime(pred.index)
    p = pred.loc[PERIOD_START:PERIOD_END]
    ranks = p.rank(axis=1, ascending=False, method="first")
    n_valid = p.notna().sum(axis=1)

    rows = []
    for code, r in winners.items():
        if code not in ranks.columns:
            rows.append(dict(code=code, ret=r, best_rank=np.nan, median_pct=np.nan,
                             days_top100=0, caught=False))
            continue
        rk = ranks[code].dropna()
        pct = (rk / n_valid.reindex(rk.index)).dropna()
        days_top100 = int((rk <= 100).sum())
        rows.append(dict(
            code=code, ret=r,
            best_rank=int(rk.min()) if len(rk) else np.nan,
            median_pct=float(pct.median()) if len(pct) else np.nan,
            days_top100=days_top100,
            caught=days_top100 >= CAUGHT_TOP100_DAYS,
        ))
    return pd.DataFrame(rows)


def shap_anatomy(missed: pd.DataFrame) -> pd.DataFrame:
    """③ 对错过的前 N_DETAIL 只做逐日 SHAP 解剖；返回明细长表。"""
    model_dir = config.ML_MODELS_DIR / MODEL_RUN
    meta = json.loads((model_dir / "run_meta.json").read_text())
    feats = meta["feature_order"]
    adapter = LGBMAdapter().load(model_dir)
    sx = WholeSetRobustZ.load(model_dir / "scaler_x.parquet", feats)

    # 不要求 has_label —— 诊断不需要标签，特征能覆盖到最新日期
    u = build_universe(horizon=meta["horizon"], start=PERIOD_START, end=PERIOD_END)
    base = u.eligible_today & u.can_buy
    fm = build_feature_matrix(
        u, base, sources=meta["sources"], neu_sources=meta.get("neu_sources"),
        has_factor_policy=HasFactorPolicy(meta["has_factor_policy"]), feature_order=feats)
    fm_dates = pd.to_datetime(fm.dates)
    fm_stocks = fm.stocks
    logger.info(f"特征矩阵 {fm.X.shape} | {fm_dates.min().date()}~{fm_dates.max().date()}")

    Z_all = sx.transform(fm.X)
    scores_all = adapter.booster.predict(Z_all)

    # date → 样本索引（截面排名用）
    date_order = pd.Series(np.arange(len(fm_dates)), index=fm_dates)
    date_groups = {d: idx.values for d, idx in date_order.groupby(level=0)}

    detail_rows = []
    for _, w in missed.head(N_DETAIL).iterrows():
        code = w["code"]
        t_idx = np.where(fm_stocks == code)[0]
        if len(t_idx) == 0:
            logger.warning(f"{code} 不在特征矩阵（可能长期涨停/停牌被 mask），跳过")
            continue

        # 均匀采样 N_SHAP_SAMPLES 个交易日
        t_idx = t_idx[np.argsort(fm_dates[t_idx])]
        step = max(1, len(t_idx) // N_SHAP_SAMPLES)
        sampled = list(t_idx[::step])
        if t_idx[-1] not in sampled:
            sampled.append(t_idx[-1])

        # 逐采样日 SHAP + 截面排名
        day_stats = []
        for si in sampled:
            d = pd.Timestamp(fm_dates[si])
            contrib = adapter.booster.predict(Z_all[si:si + 1], pred_contrib=True)
            shaps = dict(zip(feats, contrib[0, :-1]))
            day_idx = date_groups[d]
            rank = int(pd.Series(scores_all[day_idx]).rank(ascending=False, method="first")
                       .iloc[np.searchsorted(day_idx, si)])
            all_pos = sum(v for v in shaps.values() if v > 0)
            all_neg = sum(v for v in shaps.values() if v < 0)
            day_stats.append(dict(date=d, rank=rank, n=len(day_idx),
                                  score=float(scores_all[si]),
                                  all_pos=all_pos, all_neg=all_neg, shaps=shaps))
            for f, v in shaps.items():
                detail_rows.append(dict(code=code, date=d, factor=f, family=family(f),
                                        shap=v, rank=rank, score=float(scores_all[si])))

        # ── 控制台报告 ────────────────────────────────────
        print("\n" + "=" * 120)
        print(f"◆ 错过的赢家 {code} | 期间收益 {w['ret']:+.0%} | 期间最好排名 {w['best_rank']}"
              f" | 中位分位 {w['median_pct']:.2f} | top100 天数 {w['days_top100']}")
        print("=" * 120)

        print("\n  排名轨迹（采样日）:")
        for ds in day_stats:
            bar = "█" * max(0, int(50 * (1 - ds["rank"] / ds["n"])))
            print(f"    {ds['date'].strftime('%m-%d')}  排名 {ds['rank']:>5}/{ds['n']}"
                  f"  净分 {ds['score']:+.3f}  正 {ds['all_pos']:+.3f}  负 {ds['all_neg']:+.3f}  {bar}")

        # 最被嫌弃日（净分最低）深挖
        worst = min(day_stats, key=lambda x: x["score"])
        shaps = worst["shaps"]
        neg = sorted(shaps.items(), key=lambda x: x[1])[:10]
        pos = sorted(shaps.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  最被嫌弃日 {worst['date'].strftime('%Y-%m-%d')}"
              f"（排名 {worst['rank']}，净分 {worst['score']:+.3f}）拖累因子 top10:")
        for f, v in neg:
            pct = v / worst["all_neg"] * 100 if worst["all_neg"] < 0 else 0
            print(f"    {f:<35} {family(f):<12} {v:>+9.4f} ({pct:4.1f}%)  {GLOSS.get(f, '')}")
        print("  还在帮忙的因子 top5:")
        for f, v in pos:
            print(f"    {f:<35} {family(f):<12} {v:>+9.4f}  {GLOSS.get(f, '')}")

        # 因子族净贡献演化
        fams = sorted({family(f) for f in feats})
        print(f"\n  因子族净贡献演化:")
        header = f"    {'日期':<8} {'排名':>6}" + "".join(f"  {fam:>12}" for fam in fams)
        print(header)
        for ds in day_stats:
            fam_net = {fam: 0.0 for fam in fams}
            for f, v in ds["shaps"].items():
                fam_net[family(f)] += v
            line = f"    {ds['date'].strftime('%m-%d'):<8} {ds['rank']:>6}"
            line += "".join(f"  {fam_net[fam]:>+12.3f}" for fam in fams)
            print(line)

    return pd.DataFrame(detail_rows)


def main():
    logger.info(f"诊断期 {PERIOD_START}~{PERIOD_END} | 模型 {MODEL_RUN}")

    # ①② 赢家池 + 抓住/错过判定
    summary = find_winners_and_classify()
    caught_n = int(summary["caught"].sum())
    print("\n" + "=" * 120)
    print(f"① 期间涨幅 top{TOP_N_WINNERS} 赢家 | 抓住 {caught_n} 只 / 错过 {len(summary) - caught_n} 只"
          f"（判定口径: 进 top100 ≥ {CAUGHT_TOP100_DAYS} 天）")
    print("=" * 120)
    print(f"  {'代码':<14} {'期间收益':>9} {'最好排名':>8} {'中位分位':>8} {'top100天数':>9}  判定")
    for _, r in summary.iterrows():
        tag = "✓ 抓住" if r["caught"] else "✗ 错过"
        br = f"{int(r['best_rank'])}" if pd.notna(r["best_rank"]) else "无预测"
        mp = f"{r['median_pct']:.2f}" if pd.notna(r["median_pct"]) else "-"
        print(f"  {r['code']:<14} {r['ret']:>+9.0%} {br:>8} {mp:>8} {int(r['days_top100']):>9}  {tag}")

    # ③ 对错过赢家做 SHAP 解剖（按涨幅降序取前 N_DETAIL）
    missed = summary[~summary["caught"]].sort_values("ret", ascending=False)
    if missed.empty:
        logger.info("本期无错过的赢家，诊断结束")
        return
    detail = shap_anatomy(missed)

    # ④ 落盘 audit trail
    out_dir = OUT_DIR / f"{MODEL_RUN}_{PERIOD_START}_{PERIOD_END}".replace("-", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out_dir / "winners_summary.parquet")
    if not detail.empty:
        detail.to_parquet(out_dir / "shap_detail.parquet")
    logger.info(f"明细已落盘 {out_dir}")

    # 风格洞监控指标（跨月对比用）
    missed_ratio = (len(summary) - caught_n) / len(summary)
    print("\n" + "=" * 120)
    print(f"② 风格洞监控指标（跨月追踪，比率上升 = 风格洞扩大）")
    print(f"   错过赢家占比: {missed_ratio:.0%} ({len(summary) - caught_n}/{len(summary)})")
    if not detail.empty:
        drag = (detail[detail['shap'] < 0].groupby('family')['shap'].sum()
                .sort_values())
        print(f"   错过赢家的拖累因子族排名（负 SHAP 总和）:")
        for fam, v in drag.items():
            print(f"     {fam:<14} {v:>+10.3f}")
    print("=" * 120)


if __name__ == "__main__":
    main()
