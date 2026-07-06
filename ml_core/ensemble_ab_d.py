"""集成 (A+B 5050) + D(dquant)：名额分配 50/50

A+B 5050 是 LGBM 固定+滚动的集成（年化29.18%），D 是 dquant 模型（年化27.39%）。
两者在 2020/2026 vs 2022/2024 互补性强。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from loguru import logger

PRED_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/ml/predictions")
TOP_K = 100
TOP_N = 500

# A+B 5050 的预测面板 = A_rank * 0.5 + B_rank * 0.5（排序平均，用于补足和候选池）
# 但名额分配的核心是直接从两个模型的 top-N 取名额
# 这里用 A+B 的集成信号 top-50 + D 的 top-50
# A+B 5050 的信号已经生成在 ensemble_quota_50_50/signals/
# D 的信号在 a158_p27_shap_dquant_wowy/signals/
# 直接从信号文件取 top-100，各取前 50 名额

# 更精确的做法：从预测面板重新算
A = pd.read_parquet(PRED_DIR / "lgbm_a158_p27_top64" / "pred_panel_live.parquet")
B = pd.read_parquet(PRED_DIR / "lgbm_rolling_concat_2019_2025" / "pred_panel_live.parquet")
D = pd.read_parquet(PRED_DIR / "a158_p27_shap_dquant_wowy" / "pred_panel_live.parquet")
for p in (A, B, D):
    p.index = pd.to_datetime(p.index)

cd = A.index.intersection(B.index).intersection(D.index)
cs = A.columns.intersection(B.columns).intersection(D.columns)
A, B, D = A.loc[cd, cs], B.loc[cd, cs], D.loc[cd, cs]
logger.info(f"对齐: {len(cd)} 天 × {len(cs)} 股 | {cd.min().date()}~{cd.max().date()}")

# A+B 5050 的集成分（rank 平均）
A_rank = A.rank(axis=1, pct=True)
B_rank = B.rank(axis=1, pct=True)
AB = (0.5 * A_rank + 0.5 * B_rank)  # A+B 集成 rank

name = "ensemble_ab_d_50_50"
n_ab = 50  # A+B 名额
n_d = 50   # D 名额
out_dir = PRED_DIR / name
sig_dir = out_dir / "signals"
sig_dir.mkdir(parents=True, exist_ok=True)

for d in cd:
    ab = AB.loc[d].dropna()
    dd = D.loc[d].dropna()
    common = ab.index.intersection(dd.index)
    ab, dd = ab[common], dd[common]

    # A+B top-50 + D top-50
    ab_top = list(ab.nlargest(n_ab).index)
    d_top = list(dd.nlargest(n_d).index)

    merged, seen = [], set()
    for s in ab_top + d_top:
        if s not in seen:
            merged.append(s)
            seen.add(s)

    # 集成 rank 补足
    ab_r = pd.Series((rankdata(ab, method="average") - 1) / max(len(ab) - 1, 1), index=ab.index)
    d_r = pd.Series((rankdata(dd, method="average") - 1) / max(len(dd) - 1, 1), index=dd.index)
    ens = (0.5 * ab_r + 0.5 * d_r).sort_values(ascending=False)
    for s in ens.index:
        if len(merged) >= TOP_K:
            break
        if s not in seen:
            merged.append(s)
            seen.add(s)

    rest = [s for s in ens.index if s not in seen][:TOP_N - len(merged)]
    final = merged + rest
    ds = d.strftime("%Y-%m-%d")
    (sig_dir / f"{ds}.txt").write_text("\n".join(f"{ds}_{c}" for c in final[:TOP_N]) + "\n", encoding="utf-8")

logger.success(f"{name} (AB50/D50) → {sig_dir} | {len(cd)} 份")
print("---CMD_DONE---")
