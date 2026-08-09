"""
SHAP 因子解剖 —— 模型到底从每个因子学到了什么？
============================================================
对 train+valid 段全部样本算 SHAP，逐因子做：
  ① mean|SHAP| 重要性排序（复现 Stage-1 因子选择的排序口径）
  ② 按因子值分5档，看每档 SHAP 均值 → 因子值大小与模型态度的单调关系
  ③ 因子值 vs SHAP 相关系数 → 方向判定（正=值大加分 / 负=值大压分）
  ④ 按因子族汇总 → 哪一族贡献最大、方向是否一致
  ⑤ 输出 parquet 明细 + 控制台报告

用途：理解 LGBM 从 670 万样本中自动学到的因子→预测分映射关系，
回答"模型到底在用什么逻辑选股"。

运行：source peterdidi 环境后在仓库根 PYTHONPATH=. python scripts/diagnose_shap_factor_anatomy.py
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
from ml_core.splits import assign_segments

# ── 参数（手动改这里）─────────────────────────────────────────
MODEL_RUN = "lgbm_a158_p27_shap_dqlabels"
N_QUANTILES = 5          # 分档数
TOP_K_DETAIL = 20        # 控制台打印 top-K 因子明细
SAMPLE_LIMIT = 200_000   # SHAP 计算采样上限（内存保护；None=全量）
OUT_PARQUET = "/nfs/ofs-prediction/peterzhenglinpeng/tmp/diagnose_shap_factor_anatomy.parquet"
pd.set_option("display.width", 220, "display.max_rows", 200)

# ── 因子白话直觉（研报口径；方向 = 模型实际学到的用法）──────────
GLOSS = {
    # p27 微结构
    "eruption_followup_ratio": "喷发后1分钟成交额/喷发时成交额：高=散户跟风拥挤→模型看空",
    "ridge_minute_return": "量岭(持续放量段)分钟收益和：高=放量持续拉升→模型看空",
    "ridge_minute_count": "量岭分钟数：高=全天持续放量→模型看空",
    "valley_relative_vwap": "量谷VWAP/日VWAP：低=缩量时段价格塌→弱",
    "valley_weighted_quantile": "量谷价格分位：低=缩量时价格在日内低位→弱",
    "peak_interval_std": "量峰间隔std：高=脉冲式放量不规律→看空",
    "peak_ridge_turnover_ratio": "量峰成交额/量岭成交额：低=放量以持续岭为主→看空",
    "peak_minute_count": "量峰分钟数：低=缺少瞬时脉冲峰→看空",
    "eruption_turnover_sensitivity": "喷发对下一分钟成交额敏感度",
    "eruption_turnover_corr": "喷发与成交额相关性",
    "valley_ridge_price_ratio__mp10": "量谷/量岭价格比：高=缩量时价格不塌→强",
    "peak_interval_skew": "量峰间隔偏度",
    # alpha158 价格位置/反转
    "ROC60": "60日前价/今收：低=60日涨幅大→反转看空",
    "MA60": "MA60/今收：低=价格远超均线→反转看空",
    "QTLU60": "60日80分位价/今收：低=突破历史高位→反转看空",
    "QTLU20": "20日80分位价/今收",
    "QTLD60": "60日20分位价/今收",
    "MIN5": "5日最低/今收：低=短期急涨→反转看空",
    "MIN30": "30日最低/今收：低=一个月内涨幅大→反转看空",
    "MIN60": "60日最低/今收：低=三个月涨幅大",
    "MAX60": "60日最高/今收：大=从高点回落多→动量看空",
    "RSV5": "5日随机值(现价在区间位置)",
    "RSV60": "60日随机值：高=接近区间顶部",
    "IMXD30": "30日(最高点位置-最低点位置)：高=先低后高上行结构",
    "IMXD60": "60日(最高点位置-最低点位置)",
    "IMIN5": "5日最低点距今天数",
    "IMIN20": "20日最低点距今天数",
    "IMIN30": "30日最低点距今天数",
    "IMIN60": "60日最低点距今天数：低=刚创新低(超跌)→模型偏爱",
    # 波动率
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
    # 量价相关
    "CORR5": "5日量价相关", "CORR30": "30日量价相关", "CORR60": "60日量价相关",
    "CORD5": "5日量价相关(延迟)", "CORD10": "10日量价相关(延迟)",
    "CORD30": "30日量价相关(延迟)", "CORD60": "60日量价相关(延迟)",
    # 涨跌计数
    "CNTN30": "30日下跌天数占比", "CNTN60": "60日下跌天数占比",
    "SUMN20": "20日下跌日成交额占比", "SUMN60": "60日下跌日成交额占比",
    # K线/成交量
    "VWAP0": "日VWAP/今收：低=收盘强于均价(尾盘抢筹)→看空",
    "VMA60": "60日成交量均线/今成交量",
    "KLOW": "K线下影线占比：高=盘中砸出长下影",
    "KUP": "K线上影线占比",
}


def family(f: str) -> str:
    """因子族归类（与 shap_attribution.py 一致）。"""
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


def direction_label(corr: float) -> str:
    """因子值 vs SHAP 方向标签。"""
    if corr > 0.3:
        return "值大→加分(正向)"
    elif corr < -0.3:
        return "值大→压分(负向)"
    else:
        return "非线性/弱相关"


def main():
    model_dir = config.ML_MODELS_DIR / MODEL_RUN
    meta = json.loads((model_dir / "run_meta.json").read_text())
    feats = meta["feature_order"]
    adapter = LGBMAdapter().load(model_dir)
    sx = WholeSetRobustZ.load(model_dir / "scaler_x.parquet", feats)
    logger.info(f"模型 {MODEL_RUN} | 因子 {len(feats)} | horizon={meta['horizon']}")

    # ── 构建特征矩阵（与训练同口径）──────────────────────────
    u = build_universe(horizon=meta["horizon"])
    base = u.eligible_today & u.can_buy & u.has_label
    fm = build_feature_matrix(
        u, base, sources=meta["sources"], neu_sources=meta.get("neu_sources"),
        has_factor_policy=HasFactorPolicy(meta["has_factor_policy"]), feature_order=feats)
    idx = fm.index
    df_raw = fm.to_frame()

    # ── 时间切分（与训练一致）────────────────────────────────
    seg = assign_segments(u.dates)
    didx = {d: i for i, d in enumerate(u.dates)}
    ri = np.array([didx[pd.Timestamp(d)] for d in fm.dates])
    tr, va = seg["train"][ri], seg["valid"][ri]
    tv = tr | va
    logger.info(f"train+valid 样本 {tv.sum():,} | 日期 {fm.dates[tv].min()}~{fm.dates[tv].max()}")

    # ── 采样（保护内存）──────────────────────────────────────
    X_tv = fm.X[tv]
    if SAMPLE_LIMIT and tv.sum() > SAMPLE_LIMIT:
        rng = np.random.RandomState(0)
        samp = rng.choice(tv.sum(), size=SAMPLE_LIMIT, replace=False)
        logger.info(f"采样 {SAMPLE_LIMIT:,} / {tv.sum():,} 行算 SHAP")
    else:
        samp = np.arange(tv.sum())
        logger.info(f"全量 {tv.sum():,} 行算 SHAP")

    Z = sx.transform(X_tv[samp])
    contrib = adapter.booster.predict(Z, pred_contrib=True)
    C = pd.DataFrame(contrib[:, :-1], columns=feats)
    bias = contrib[0, -1]
    raw_tv = df_raw[tv].iloc[samp]
    logger.info(f"SHAP 矩阵 {C.shape} | bias={bias:+.6f}")

    # ── 逐因子统计 ──────────────────────────────────────────
    rows = []
    for f in feats:
        vals = raw_tv[f].values
        shaps = C[f].values
        m = np.isfinite(vals) & np.isfinite(shaps)
        vals, shaps = vals[m], shaps[m]
        if len(vals) < 100:
            continue
        corr = float(np.corrcoef(vals, shaps)[0, 1]) if vals.std() > 0 else 0.0
        # 分档统计
        qs = np.quantile(vals, np.linspace(0, 1, N_QUANTILES + 1)[1:-1])
        bins = np.concatenate([[-np.inf], qs, [np.inf]])
        groups = pd.cut(vals, bins=bins, labels=False)
        q_stats = []
        for g in range(N_QUANTILES):
            gm = groups == g
            if gm.sum() == 0:
                q_stats.append((np.nan, 0, np.nan, np.nan))
            else:
                q_stats.append((float(shaps[gm].mean()), int(gm.sum()),
                                float(vals[gm].min()), float(vals[gm].max())))
        rows.append(dict(
            factor=f, family=family(f),
            mean_abs_shap=float(np.abs(shaps).mean()),
            shap_mean=float(shaps.mean()),
            shap_std=float(shaps.std()),
            corr_val_shap=corr,
            direction=direction_label(corr),
            val_min=float(vals.min()), val_max=float(vals.max()),
            val_mean=float(vals.mean()), val_median=float(np.median(vals)),
            gloss=GLOSS.get(f, ""),
            **{f"q{i+1}_shap": q_stats[i][0] for i in range(N_QUANTILES)},
            **{f"q{i+1}_n": q_stats[i][1] for i in range(N_QUANTILES)},
            **{f"q{i+1}_val_lo": q_stats[i][2] for i in range(N_QUANTILES)},
            **{f"q{i+1}_val_hi": q_stats[i][3] for i in range(N_QUANTILES)},
        ))
    df = pd.DataFrame(rows).set_index("factor")
    df = df.sort_values("mean_abs_shap", ascending=False)

    # ── 输出 ① 重要性排名 ───────────────────────────────────
    print("\n" + "=" * 100)
    print(f"① SHAP 因子重要性排名（train+valid {SAMPLE_LIMIT or '全量'} 样本，mean|SHAP| 降序）")
    print("=" * 100)
    print(f"  {'排名':>4} {'因子':<35} {'族':<14} {'mean|SHAP|':>12} {'方向':>18}")
    print("  " + "-" * 90)
    for i, (f, r) in enumerate(df.head(TOP_K_DETAIL).iterrows(), 1):
        print(f"  {i:>4} {f:<35} {r['family']:<14} {r['mean_abs_shap']:>12.6f} {r['direction']:>18}")

    # ── 输出 ② top-K 因子分档明细 ───────────────────────────
    print(f"\n② Top-{TOP_K_DETAIL} 因子分档 SHAP 明细（因子值 Q1=最小→Q5=最大）")
    print("=" * 100)
    for f in df.head(TOP_K_DETAIL).index:
        r = df.loc[f]
        print(f"\n  ◆ {f}  [{r['family']}]  corr={r['corr_val_shap']:+.3f}  {r['direction']}")
        print(f"    直觉: {r['gloss']}")
        print(f"    {'档位':<10} {'因子值范围':>22} {'样本数':>8} {'SHAP均值':>12} {'方向':>8}")
        for i in range(N_QUANTILES):
            lo = r[f"q{i+1}_val_lo"]; hi = r[f"q{i+1}_val_hi"]
            sm = r[f"q{i+1}_shap"]; n = r[f"q{i+1}_n"]
            d = "加分↑" if sm > 0 else "压分↓"
            print(f"    Q{i+1}{'(最小)' if i==0 else '(最大)' if i==4 else '':<5}"
                  f"  [{lo:.4f}, {hi:.4f}]  {n:>8}  {sm:>+12.6f}  {d:>8}")

    # ── 输出 ③ 因子族汇总 ───────────────────────────────────
    print(f"\n③ 因子族汇总")
    print("=" * 100)
    fam = df.groupby("family").agg(
        因子数=("mean_abs_shap", "count"),
        平均mean_SHAP=("mean_abs_shap", "mean"),
        总mean_SHAP=("mean_abs_shap", "sum"),
        正向因子数=("corr_val_shap", lambda x: (x > 0.3).sum()),
        负向因子数=("corr_val_shap", lambda x: (x < -0.3).sum()),
    ).sort_values("总mean_SHAP", ascending=False)
    fam["占比"] = fam["总mean_SHAP"] / fam["总mean_SHAP"].sum()
    print(fam.round(4).to_string())

    # ── 输出 ④ 方向分布 ─────────────────────────────────────
    print(f"\n④ 因子方向分布（值大→加分 vs 值大→压分 vs 非线性）")
    print("=" * 100)
    dirs = df["direction"].value_counts()
    for d, n in dirs.items():
        examples = df[df["direction"] == d].head(5).index.tolist()
        print(f"  {d:<20} {n:>3} 个 | 例: {', '.join(examples[:5])}")

    # ── 落盘 ────────────────────────────────────────────────
    df.to_parquet(OUT_PARQUET)
    logger.info(f"逐因子解剖明细已落盘 {OUT_PARQUET}")


if __name__ == "__main__":
    main()
