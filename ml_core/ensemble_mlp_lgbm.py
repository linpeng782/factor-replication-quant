"""集成 MLP + LGBM（固定/滚动）：名额分配法

MLP 是最强单模型（年化35%），与 LGBM 持仓重合度仅 20-24%。
测试组合：
  MLP + A固定  (50/50, 60/40, 70/30)
  MLP + B滚动  (50/50, 60/40, 70/30)
  MLP + A + B  (三者名额分配)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from loguru import logger

PRED_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/ml/predictions")
TOP_K = 100
TOP_N = 500

# 三个模型面板
M = pd.read_parquet(PRED_DIR / "mlp_a158_p27_top178" / "pred_panel_live.parquet")
A = pd.read_parquet(PRED_DIR / "lgbm_a158_p27_top64" / "pred_panel_live.parquet")
B = pd.read_parquet(PRED_DIR / "lgbm_rolling_concat_2019_2025" / "pred_panel_live.parquet")
for p in (M, A, B):
    p.index = pd.to_datetime(p.index)

cd = M.index.intersection(A.index).intersection(B.index)
cs = M.columns.intersection(A.columns).intersection(B.columns)
M, A, B = M.loc[cd, cs], A.loc[cd, cs], B.loc[cd, cs]
logger.info(f"对齐: {len(cd)} 天 × {len(cs)} 股 | {cd.min().date()}~{cd.max().date()}")


def quota_2model(pred1, pred2, w1, w2, name):
    """两模型名额分配：top-K = model1 top-n1 + model2 top-n2（去重补足）"""
    n1 = int(TOP_K * w1); n2 = TOP_K - n1
    out_dir = PRED_DIR / name
    sig_dir = out_dir / "signals"
    sig_dir.mkdir(parents=True, exist_ok=True)
    for d in cd:
        p1 = pred1.loc[d].dropna(); p2 = pred2.loc[d].dropna()
        common = p1.index.intersection(p2.index)
        p1, p2 = p1[common], p2[common]
        top1 = list(p1.nlargest(n1).index)
        top2 = list(p2.nlargest(n2).index)
        merged, seen = [], set()
        for s in top1 + top2:
            if s not in seen: merged.append(s); seen.add(s)
        # 集成 rank 补足
        r1 = pd.Series((rankdata(p1, method="average")-1)/max(len(p1)-1,1), index=p1.index)
        r2 = pd.Series((rankdata(p2, method="average")-1)/max(len(p2)-1,1), index=p2.index)
        ens = (w1*r1 + w2*r2).sort_values(ascending=False)
        for s in ens.index:
            if len(merged) >= TOP_K: break
            if s not in seen: merged.append(s); seen.add(s)
        rest = [s for s in ens.index if s not in seen][:TOP_N-len(merged)]
        final = merged + rest
        ds = d.strftime("%Y-%m-%d")
        (sig_dir / f"{ds}.txt").write_text("\n".join(f"{ds}_{c}" for c in final[:TOP_N]) + "\n", encoding="utf-8")
    logger.success(f"{name} ({w1:.0%}/{w2:.0%}) → {sig_dir} | {len(cd)} 份")


def quota_3model(preds, weights, name):
    """三模型名额分配"""
    ns = [int(TOP_K * w) for w in weights]
    ns[-1] = TOP_K - sum(ns[:-1])  # 尾部补齐
    out_dir = PRED_DIR / name
    sig_dir = out_dir / "signals"
    sig_dir.mkdir(parents=True, exist_ok=True)
    for d in cd:
        ps = [p.loc[d].dropna() for p in preds]
        common = ps[0].index
        for p in ps[1:]: common = common.intersection(p.index)
        ps = [p[common] for p in ps]
        merged, seen = [], set()
        for p, n in zip(ps, ns):
            for s in p.nlargest(n).index:
                if s not in seen: merged.append(s); seen.add(s)
        # 集成 rank 补足
        ranks = [pd.Series((rankdata(p, method="average")-1)/max(len(p)-1,1), index=p.index) for p in ps]
        ens = sum(w*r for w, r in zip(weights, ranks)).sort_values(ascending=False)
        for s in ens.index:
            if len(merged) >= TOP_K: break
            if s not in seen: merged.append(s); seen.add(s)
        rest = [s for s in ens.index if s not in seen][:TOP_N-len(merged)]
        final = merged + rest
        ds = d.strftime("%Y-%m-%d")
        (sig_dir / f"{ds}.txt").write_text("\n".join(f"{ds}_{c}" for c in final[:TOP_N]) + "\n", encoding="utf-8")
    logger.success(f"{name} ({'/'.join(f'{w:.0%}' for w in weights)}) → {sig_dir} | {len(cd)} 份")


# MLP + A 固定
quota_2model(M, A, 0.5, 0.5, "ensemble_mlp_a_50_50")
quota_2model(M, A, 0.6, 0.4, "ensemble_mlp_a_60_40")
quota_2model(M, A, 0.7, 0.3, "ensemble_mlp_a_70_30")

# MLP + B 滚动
quota_2model(M, B, 0.5, 0.5, "ensemble_mlp_b_50_50")
quota_2model(M, B, 0.6, 0.4, "ensemble_mlp_b_60_40")
quota_2model(M, B, 0.7, 0.3, "ensemble_mlp_b_70_30")

# 三模型
quota_3model([M, A, B], [0.5, 0.25, 0.25], "ensemble_mlp_a_b_50_25_25")
quota_3model([M, A, B], [0.4, 0.3, 0.3], "ensemble_mlp_a_b_40_30_30")

print("---CMD_DONE---")
