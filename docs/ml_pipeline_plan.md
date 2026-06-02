# LightGBM 因子合成训练流水线 — 可执行计划

> 参考：国金证券《之十：机器学习全流程重构》《之十三：特征筛选、SHAP、中性化》。
> 定位：本流水线**消费** factor-rep 的因子库 + 标签 + mask，**产出**合成选股信号 ŷ。
> 与「单因子评估」（core/evaluation.py）解耦，独立于本仓顶层 `ml/` 目录。

## 0. 目标与总体结构（两阶段）

用 LightGBM（GBDT / MSE）把 ~200 raw 因子**先筛后合**为一个截面选股信号 ŷ，
预测未来 20 日截面超额收益；样本外用逐日截面 IC 评估（对齐国金图表1）。

```
Stage 1 — 样本内因子筛选 (只用 train+valid, test 全程不可见)
   全因子(RobustZScore后) → 训 LightGBM → feature_importance 排序 → 选 top-64
        │   (先用 GBDT 自带重要性；跑通后再用 SHAP 选一遍做对照)
        ▼
Stage 2 — 样本外最终模型
   仅用选出的 64 因子 → 重训最终 GBDT → test 评估 (模型 IC)
```

> 反直觉点（国金之十三§3 原文）：「GBDT 自带特征选择能力，故之十三的筛选实验仅用于喂 NN/GRU」。
> 即对纯 GBDT 全喂亦可。我们仍做筛选，理由：① 203 因子高度冗余（IC 序列实测仅 ~34 独立 alpha）；
> ② 降过拟合/成本、提升可解释；③ 这是本研究目标（SHAP 选因子）。

---

## 1. 特征（模型输入）+ 预处理

| 步骤 | 设定 | 依据 |
|---|---|---|
| 起点 | `factors/raw/`（203 个，**不用 cleaned/neu**） | cleaned/neu 已逐日 CSZScore（实测每日 mean0/std1），抹掉跨日水位，不能喂树 |
| ① inf→NaN | `replace([inf,-inf], nan)` | YOLO 引擎可能产 inf |
| ② pre_mask | 剔 ST/停牌/新股 | 不污染标准化尺子；非可投样本 |
| ③ **全集 RobustZScore** | 每因子 `z=(x−median)/(1.4826·MAD)`；全集 per-factor（非逐日）；median/MAD **只在 train 段 fit**，三段同尺子 transform。**无额外 clip**（靠 MAD 自身抗极值，对齐国金原文） | 国金之十§1.2：特征整体标准化保留跨日，RobustZScore 最稳健 |
| ④ post_mask | 剔涨停 | 仅作用于**预测/可买集合**；训练样本是否剔涨停后续再议 |
| **不做** 行业市值中性化 | — | neutralize 亦为逐日截面 + 末尾 CSZScore，会抹跨日；风格暴露由「超额标签 + universe」处理 |

> 纪律：凡"从数据学出的参数"（RobustZScore 尺子）只能用 train 段 fit；valid/test 仅 transform。

---

## 2. 标签 + 预处理

| 步骤 | 设定 |
|---|---|
| 基础收益 | `forward_return_20d = vwap[t+21]/vwap[t+1] − 1`（T+1 进场，持有 20 日，已避前视） |
| **超额（路 A）** | **截面等权 demean**：`excess = r − 当日 pre_mask universe 等权均值`（与中证全指超额相关 0.94、IC 等价） |
| **标准化** | 对 excess 再套全集 RobustZScore（train 段 fit，全数据 transform） |
| 预测目标 | **超额收益**（绝对收益含市场整体涨跌，模型会被行情带偏） |

> 国金之十§1.2：「GBDT 类模型，使用超额收益率作预测目标，**特征和标签均用 RobustZscore 处理**」。
> demean=定义超额（减市场）；RobustZScore=标准化尺度（全集、train-fit、保留跨日）。两步分工不同，不冲突。

---

## 3. 数据集划分（一次性训练 · 方案 A）

```
train : 2012-01 ~ 2019-11   (8 年, fit 尺子 + fit 模型梯度)
        └ embargo: 2019-12 整月丢弃 (≥20 交易日 horizon, 防标签泄露)
valid : 2020-01 ~ 2021-11   (2 年, 只早停 + 选超参)
        └ embargo: 2021-12
test  : 2022-01 ~ 2026-03   (样本外, 只最终评估一次)
```
- 严格**按时间切，不 shuffle**；embargo = 1 个交易月（标签 horizon=20 日）。
- 国金之十图表13：LightGBM **一次性训练**（IC 10.69%）优于滚动(8.14%)/扩展(8.42%)。

---

## 4. Stage 1 — 因子筛选（样本内，只用 train+valid）

| 项 | 当前设定 | 后续 |
|---|---|---|
| 方法 | **GBDT 自带 `feature_importance`**（gain）排序（先跑通） | 再用 **SHAP TreeExplainer + mean(\|SHAP\|)** 选一遍，**对比两者选出的因子差异** |
| 选多少 | **top-64**（国金从 158 选 64） | — |
| MMR 去冗余 | **不做** | — |
| 筛选窗口 | **一次性**（train+valid 一把，与 GBDT one-shot 一致） | — |
| 数据卫生 | 筛选只用 train+valid，**test 不可见** | — |
| SHAP 采样 | （SHAP 阶段）随机采 1e5 样本（国金验证够稳） | — |

产出：`selected_features.json`（64 因子名 + 重要性分数 + 方法标记）。

---

## 5. Stage 2 — 最终模型训练 LightGBM

| 项 | 当前设定 | 后续 |
|---|---|---|
| 输入 | **仅 Stage 1 选出的 64 因子** | — |
| objective | `regression`（MSE/L2） | 国金：改 IC loss 无显著提升 |
| boosting | **GBDT（先跑通）** | 再上 DART |
| 早停 | `early_stopping` on valid（patience≈80），回滚 best_iteration | — |
| 随机种子 | **1 个（先跑通）** | 再扩 5 个取均值 |
| universe | **全 A 训练** | 再做成分股对比 |
| 关键超参 | num_leaves / min_child_samples / learning_rate / feature_fraction / bagging_fraction / lambda | 在 valid 上调 |

---

## 6. 评估

- test：`predict → ŷ` → **逐日截面 Spearman IC(ŷ, 真实超额) → 序列**；均值=IC，均值/std=ICIR，t=ICIR·√天数（之十图表1 口径）。
- **不做分层回测**（国金图表1 亦只看 IC；分层/多空等以后需要再加）。
- **特征不做截面标准化 ≠ IC 不按截面算**：IC 永远逐日截面 rank 相关（输入口径 vs 评价口径是两件事）。
- **对照实验**：跑通 GBDT-importance 选 64 后，用 SHAP 选 64，比较 ①入选因子重合度 ②样本外模型 IC 差异（接 docs/shap_selection_plan.md）。

---

## 7. 目录与产物

**代码（本仓顶层 `ml/`，与 factor_production/ 平级）**
```
ml/
  __init__.py
  labels.py      # 超额标签构造（demean）
  preprocess.py  # RobustZScoreScaler（fit/transform，全集 per-factor，train-only）
  dataset.py     # 组装特征矩阵 + 标签 + 时间划分（含 embargo）+ mask 对齐
  select.py      # Stage 1 因子筛选：GBDT feature_importance（→ 后续 SHAP）→ top-64
  train.py       # Stage 2 LightGBM(GBDT/MSE) + 早停
  evaluate.py    # 模型 IC（逐日截面 Spearman）
  run.py         # CLI 入口，串起 dataset → select → train → evaluate
```

**数据产物（`FACTOR_REPL_DATA_ROOT/ml/`，与 factors/ 平级）**
```
ml/
  models/<run_id>/       # lgbm 模型 + 超参 + RobustZScore 尺子 + selected_features.json
  predictions/<run_id>/  # ŷ 面板 + ic_series（逐日截面 IC）
  datasets/<run_id>/     # (可选) train/valid/test 矩阵，便于复跑
```
