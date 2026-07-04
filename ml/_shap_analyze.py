"""临时分析脚本：算每个头部因子的 SHAP 方向性与单调性（一次性，用完可删）。"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from scipy.stats import spearmanr
import config
from ml.shap_plot import build_sample
from ml.predict_live import _load_scaler

RUN = "shap_top64"
model_dir = config.ML_MODELS_DIR / RUN
sel = json.loads((model_dir / "selected_features.json").read_text())["features"]
model = lgb.Booster(model_file=str(model_dir / "model.txt"))

X_raw = build_sample(sel, n_sample=30000, date_step=10)
sx = _load_scaler(model_dir, sel)
Xz = sx.transform(X_raw)
sv = shap.TreeExplainer(model).shap_values(Xz)
SV = pd.DataFrame(sv, columns=sel, index=Xz.index)

# 每因子：mean|SHAP|、dir=corr(因子值, SHAP值)、正向/负向 SHAP 质量占比、尾部
rows = []
for f in sel:
    x = Xz[f]; s = SV[f]
    m = x.notna() & s.notna()
    rho = spearmanr(x[m], s[m]).statistic if m.sum() > 50 else np.nan
    pos = (s > 0).mean(); neg = (s < 0).mean()
    rows.append(dict(
        factor=f,
        mean_abs=float(s.abs().mean()),
        dir_corr=float(rho),                      # >0: 因子值越高越看多；<0: 越高越看空
        shap_p95=float(s.quantile(0.95)),
        shap_p05=float(s.quantile(0.05)),
        skew=float(s.skew()),                     # SHAP 分布偏度（尾部方向）
        frac_pos=float(pos),
    ))
df = pd.DataFrame(rows).sort_values("mean_abs", ascending=False)
pd.set_option("display.width", 160, "display.max_rows", 70, "display.float_format", lambda v: f"{v:+.4f}")
print(df.head(25).to_string(index=False))
print("\n=== 按 |dir_corr| 看方向最清晰的因子 ===")
print(df.reindex(df.dir_corr.abs().sort_values(ascending=False).index).head(15)[
    ["factor","mean_abs","dir_corr","frac_pos","skew"]].to_string(index=False))
