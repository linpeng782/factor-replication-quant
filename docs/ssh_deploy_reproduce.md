# SSH 部署与复现 Runbook（给 SSH 上的 agent / 操作者）

> 目标：在 SSH 服务器（/nfs 数据根）上**零改代码**跑通 ML 因子合成流水线，并复现样本外 IC。
> 设计详见 `docs/ml_pipeline_plan.md`。

---

## 0. 环境（一次性）

```bash
# Python 3.11 venv
python3.11 -m venv ~/venv-factor && source ~/venv-factor/bin/activate
pip install -U pip
# ML 流水线依赖（Linux 上 lightgbm wheel 自带 OpenMP，无需额外装 libomp）
pip install lightgbm pandas pyarrow numpy loguru
# 仅 SHAP 筛选阶段需要（--select-method shap）
pip install shap
```

## 1. 代码

```bash
git clone https://github.com/linpeng782/factor-replication-quant.git
cd factor-replication-quant
git checkout feature/operators-upgrade        # ML 流水线在此分支
```

## 2. 数据根（关键：无需改代码）

代码默认数据根 = `/nfs/ofs-prediction/peterzhenglinpeng`（见 core/config.py），
**SSH 上不用设任何环境变量**即指向这里。若要改：`export FACTOR_REPL_DATA_ROOT=<其他路径>`。

### 必需数据清单（只需这 3 类，~14.3GB）
放到数据根下，保持以下相对结构：
```
<DATA_ROOT>/
  factors/raw/<source>/<group>/*.parquet              # 特征：203 因子，~13.9GB
  market-data/labels/forward_return_*.parquet         # 标签，~0.4GB
  market-data/masks/combo_mask_long.parquet           # mask，~23MB
```
> 不需要：minute/ cleaned/ neu/ intermediate-cache/ industry/ market_cap/。

### 从 Mac 同步数据（在 Mac 上执行，保持相对路径）
```bash
SSH=用户@SSH地址; DST=/nfs/ofs-prediction/peterzhenglinpeng
rsync -avP /Users/didi/DATA/factors/raw            $SSH:$DST/factors/
rsync -avP /Users/didi/DATA/market-data/labels     $SSH:$DST/market-data/
rsync -avP /Users/didi/DATA/market-data/masks      $SSH:$DST/market-data/
```

### 数据自检（SSH 上执行）
```bash
python - <<'PY'
from core import config
import pandas as pd
print("DATA_ROOT:", config._DATA_ROOT)
raw = list(config.RAW_FACTOR_BASE.glob("*/*/*.parquet"))
print("factors/raw 因子数:", len(raw), "(应=203)")
print("labels 20d 存在:", (config.LABELS_DIR/"forward_return_20d.parquet").exists())
print("combo_mask 存在:", config.COMBO_MASK_PATH.exists())
assert len(raw) == 203 and config.COMBO_MASK_PATH.exists()
print("✅ 数据齐全")
PY
```

## 3. 冒烟复现（必须先做，验证迁移无误）

```bash
python -m ml.run --date-sample 10 --run-id smoke
```
**期望结果（与 Mac 一致 → 迁移正确）：**
```
样本外模型 IC 均值=+0.1138  ICIR=+1.104  t=+11.21  天数=103
最强因子: eruption_followup_ratio
```
日志在 `ml/logs/smoke/run_<时间戳>.log`（含 train/valid 损失曲线 + 入选 top-64）。

## 4. 全样本正式跑（800G 内存无压力，无需任何内存优化）

```bash
python -m ml.run --run-id full_gbdt        # 全 203 因子、全样本、GBDT 选 64
```
- 预计：建数据集 + 训练，128 核下数分钟～十几分钟。
- 产物：
  - 模型/选择：`<DATA_ROOT>/ml/models/full_gbdt/`（model.txt + selected_features.json）
  - 预测/IC：`<DATA_ROOT>/ml/predictions/full_gbdt/`（pred_panel.parquet + ic_series.parquet）
  - 日志：`<repo>/ml/logs/full_gbdt/run_<时间戳>.log`

### 分年 IC 速查（跑完后）
```bash
python - <<'PY'
import pandas as pd; from core import config
ic = pd.read_parquet(config.ML_PREDICTIONS_DIR/"full_gbdt"/"ic_series.parquet")["ic"]
ic.index = pd.to_datetime(ic.index); g = ic.groupby(ic.index.year)
print(pd.DataFrame({"IC":g.mean(),"ICIR":g.mean()/g.std(),"天数":g.size()}).round(4))
PY
```

## 5. 后续实验（128 核可并行）
- SHAP 对照：`python -m ml.run --select-method shap --run-id full_shap`（比 gbdt 选的 64 因子重合度 + IC 差异）
- 多种子：循环 `--seed`，对 ŷ/IC 取均值（国金 5 种子）
- DART：在 `ml/train.py` 的 `DEFAULT_PARAMS["boosting_type"]` 改 "dart"

## 6. 故障排查
| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: core` | 在仓库根目录运行（`python -m ml.run`），勿进 ml/ 子目录 |
| lightgbm 加载报 OpenMP | Linux 一般无此问题；如有 `pip install lightgbm` 重装或装系统 `libgomp` |
| 数据自检 assert 失败 | 检查 rsync 是否保持了 `factors/raw`、`market-data/{labels,masks}` 相对结构 |
| IC 与 +0.114 不符 | 检查因子数=203、标签/ mask 版本一致；数据不全会改变结果 |
