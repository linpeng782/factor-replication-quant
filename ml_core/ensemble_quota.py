"""集成 A(固定) + B(滚动) 全周期：名额分配法（对齐 ensemble_methodology.md v3 方案）

top-100 = 强模型 top-nA + 弱模型 top-nB（去重后从集成 rank 补足）
top-101~500 按集成 rank 填充（回测引擎候选池用）

测试多种权重组合，导出信号供回测对比。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from loguru import logger

PRED_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/ml/predictions")
TOP_K = 100   # 回测实际持仓数
TOP_N = 500   # 信号文件总行数

# 权重方案: (name, w_rolling, w_fixed) — 滚动在前，固定在后
SCHEMES = [
    ("ensemble_quota_50_50", 0.5, 0.5),
    ("ensemble_quota_60_40", 0.6, 0.4),
    ("ensemble_quota_70_30", 0.7, 0.3),
    ("ensemble_quota_40_60", 0.4, 0.6),
]

rolling = pd.read_parquet(PRED_DIR / "lgbm_rolling_concat_2019_2025" / "pred_panel_live.parquet")
fixed = pd.read_parquet(PRED_DIR / "lgbm_a158_p27_top64" / "pred_panel_live.parquet")
rolling.index = pd.to_datetime(rolling.index)
fixed.index = pd.to_datetime(fixed.index)

common_dates = rolling.index.intersection(fixed.index)
common_stocks = rolling.columns.intersection(fixed.columns)
rolling = rolling.loc[common_dates, common_stocks]
fixed = fixed.loc[common_dates, common_stocks]
logger.info(f"对齐: {len(common_dates)} 天 × {len(common_stocks)} 股 | {common_dates.min().date()}~{common_dates.max().date()}")

for name, w_r, w_f in SCHEMES:
    n_r = int(TOP_K * w_r)
    n_f = TOP_K - n_r
    out_dir = PRED_DIR / name
    sig_dir = out_dir / "signals"
    sig_dir.mkdir(parents=True, exist_ok=True)

    for d in common_dates:
        r = rolling.loc[d].dropna()
        f = fixed.loc[d].dropna()
        common = r.index.intersection(f.index)
        r, f = r[common], f[common]

        # 名额分配：滚动 top-nR + 固定 top-nF
        r_top = list(r.nlargest(n_r).index)
        f_top = list(f.nlargest(n_f).index)

        # 合并去重（保持顺序）
        top_k_merged, seen = [], set()
        for s in r_top + f_top:
            if s not in seen:
                top_k_merged.append(s)
                seen.add(s)

        # 集成 rank（用于补足 + top-101~500）
        r_rank = pd.Series((rankdata(r, method="average") - 1) / max(len(r) - 1, 1), index=r.index)
        f_rank = pd.Series((rankdata(f, method="average") - 1) / max(len(f) - 1, 1), index=f.index)
        ens_rank = (w_r * r_rank + w_f * f_rank).sort_values(ascending=False)

        # 不足 100 从集成 rank 补
        for s in ens_rank.index:
            if len(top_k_merged) >= TOP_K:
                break
            if s not in seen:
                top_k_merged.append(s)
                seen.add(s)

        # top-101~500 按集成 rank 填充
        rest = [s for s in ens_rank.index if s not in seen][:TOP_N - len(top_k_merged)]
        final = top_k_merged + rest

        ds = d.strftime("%Y-%m-%d")
        (sig_dir / f"{ds}.txt").write_text(
            "\n".join(f"{ds}_{c}" for c in final[:TOP_N]) + "\n", encoding="utf-8")

    logger.success(f"{name} (滚动{n_r}/固定{n_f}) → {sig_dir} | {len(common_dates)} 份")

# 验证第一天
d0 = common_dates[0]
r0 = rolling.loc[d0].dropna()
f0 = fixed.loc[d0].dropna()
common0 = r0.index.intersection(f0.index)
r0, f0 = r0[common0], f0[common0]
sig0 = open(PRED_DIR / SCHEMES[1][0] / "signals" / f"{d0.strftime('%Y-%m-%d')}.txt").read().strip().split("\n")
sig0_codes = [s.split("_")[1] for s in sig0[:100]]
ens_top100 = set(sig0_codes)
print(f"\n=== 验证第一天 {d0.date()} top-100（60/40方案）===")
print(f"集成 top-100 中属于滚动 top-60: {len(ens_top100 & set(r0.nlargest(60).index))}")
print(f"集成 top-100 中属于固定 top-40: {len(ens_top100 & set(f0.nlargest(40).index))}")
print(f"滚动 top-60 ∪ 固定 top-40 去重后: {len(set(r0.nlargest(60).index) | set(f0.nlargest(40).index))} 只")
print("\n---CMD_DONE---")
