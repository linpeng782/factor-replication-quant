"""集成 D(dquant) + B(滚动) ：名额分配法

D vs B 持仓重合度仅 26.64%，rank-IC 0.735，差异化足够。
D 在 2022/2024 大幅跑赢 B，B 在 2021/2026 大幅跑赢 D——互补性强。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from loguru import logger

PRED_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/ml/predictions")
TOP_K = 100
TOP_N = 500

D = pd.read_parquet(PRED_DIR / "a158_p27_shap_dquant_wowy" / "pred_panel_live.parquet")
B = pd.read_parquet(PRED_DIR / "lgbm_rolling_concat_2019_2025" / "pred_panel_live.parquet")
D.index = pd.to_datetime(D.index)
B.index = pd.to_datetime(B.index)

cd = D.index.intersection(B.index)
cs = D.columns.intersection(B.columns)
D, B = D.loc[cd, cs], B.loc[cd, cs]
logger.info(f"对齐: {len(cd)} 天 × {len(cs)} 股 | {cd.min().date()}~{cd.max().date()}")

SCHEMES = [
    ("ensemble_d_b_50_50", 0.5, 0.5),
    ("ensemble_d_b_60_40", 0.6, 0.4),
    ("ensemble_d_b_70_30", 0.7, 0.3),
    ("ensemble_d_b_40_60", 0.4, 0.6),
]

for name, w_d, w_b in SCHEMES:
    n_d = int(TOP_K * w_d); n_b = TOP_K - n_d
    out_dir = PRED_DIR / name
    sig_dir = out_dir / "signals"
    sig_dir.mkdir(parents=True, exist_ok=True)
    for d in cd:
        p1 = D.loc[d].dropna(); p2 = B.loc[d].dropna()
        common = p1.index.intersection(p2.index)
        p1, p2 = p1[common], p2[common]
        top1 = list(p1.nlargest(n_d).index)
        top2 = list(p2.nlargest(n_b).index)
        merged, seen = [], set()
        for s in top1 + top2:
            if s not in seen: merged.append(s); seen.add(s)
        r1 = pd.Series((rankdata(p1, method="average")-1)/max(len(p1)-1,1), index=p1.index)
        r2 = pd.Series((rankdata(p2, method="average")-1)/max(len(p2)-1,1), index=p2.index)
        ens = (w_d*r1 + w_b*r2).sort_values(ascending=False)
        for s in ens.index:
            if len(merged) >= TOP_K: break
            if s not in seen: merged.append(s); seen.add(s)
        rest = [s for s in ens.index if s not in seen][:TOP_N-len(merged)]
        final = merged + rest
        ds = d.strftime("%Y-%m-%d")
        (sig_dir / f"{ds}.txt").write_text("\n".join(f"{ds}_{c}" for c in final[:TOP_N]) + "\n", encoding="utf-8")
    logger.success(f"{name} (D{n_d}/B{n_b}) → {sig_dir} | {len(cd)} 份")

print("---CMD_DONE---")
