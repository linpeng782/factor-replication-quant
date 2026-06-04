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
test  : 2022-01 ~ 2026-03   (样本外, 只最终评估一次; 末日=标签可兑现末日 2026-03-31)
```
- 严格**按时间切，不 shuffle**；embargo = 1 个交易月（标签 horizon=20 日）。
- 国金之十图表13：LightGBM **一次性训练**（IC 10.69%）优于滚动(8.14%)/扩展(8.42%)。
- test 末日 **2026-03-31 = 标签可兑现末日**（20 日 forward 收益需未来 20 个交易日，受行情末日所限）；**评估口径固定于此，不随因子更新前移**。
- **实盘推理**（`predict_live`）用同一把 train 段尺子，把 ŷ 补到所有入选因子**共同覆盖的最新交易日**（当前 2026-05-27，只过 pre_mask、不过 label，是评估面板的超集）；评估口径不受影响，两面板在共有格子上 ŷ 逐元素相等。

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
| 早停 | `early_stopping` on valid，**patience=200 / num_boost_round=1000**，回滚 best_iteration | patience=80 曾把 final 截到 **8 棵树**（valid 早期局部极小被过早锚定，第二次下降还没超过它就触发）→ ŷ 粗、tie 多、top-N 边界靠 tie-break 抖动；放宽到 200 后收敛到 ~200–230 棵，test IC **+0.110→+0.122**（见 §8） |
| 随机种子 | 当前 **seed=42**（train.py 默认；曾用 1） | 种子是**噪声级**变量：seed 1↔42 入选因子 **56/64 重合**（仅尾部近义因子互换）、test IC 仅差 ±0.0014；多种子取均值仍待办 |
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
  run.py         # CLI 入口，串起 dataset → select → train → evaluate；并物化 scaler_x（供实盘复用）
  predict_live.py# 实盘推理支路：raw → train 段 scaler → 打分补到最新因子日（只过 pre_mask，不过 label）
  export_signal.py# 导出回测可读信号：ŷ 面板 → 每日 YYYY-MM-DD.txt（默认 live 口径/daily 布局；可切 eval/merged）
```

**数据产物（`FACTOR_REPL_DATA_ROOT/ml/`，与 factors/ 平级）**
```
ml/
  models/<run_id>/       # model.txt + selected_features.json + scaler_x.parquet(median/scale, train 段) + 超参
  predictions/<run_id>/  # pred_panel(评估口径,止于2026-03-31) + pred_panel_live(实盘口径,到最新因子日) + ic_series
  signals/<run_id>/      # 回测信号：每日 YYYY-MM-DD.txt（top-N，行=「日期_代码」，行序即优先级）；可选 merged signal.txt
  datasets/<run_id>/     # (可选) train/valid/test 矩阵，便于复跑
```
+ 训练日志落在 **repo 内 `ml/logs/<run_id>_<时间戳>.log`**（含入选因子重要性；带时间戳，复用 run_id 也不覆盖；.gitignore 排除）。

> 全链路一条命令串起：`python -m ml.run --run-id X` → `python -m ml.predict_live --run-id X` → `python -m ml.export_signal --run-id X`。
> run_id 驱动所有目录，便于并存多版模型对比（如 `full_gbdt_es200`、`full_gbdt_seed42`）。

---

## 8. 实测结论（2026-06）

### 8.1 top_k 扫描：64 是边际效率拐点（非精度最优）

固定同一份 gain 重要性降序，逐 k 取前 k 因子重训 → 样本外 test IC（口径同 §6）：

| top_k | 8 | 16 | 24 | 32 | 48 | **64** | 96 | 128 | 160 | 203 |
|---|---|---|---|---|---|---|---|---|---|---|
| test IC | .1024 | .1008 | .1024 | .1176 | .1202 | **.1217** | .1233 | .1239 | **.1244** | .1241 |
| best_it | 5 | — | — | 316 | — | 231 | — | — | 峰值 | 回落 |

三段形态：① **k≤24 塌陷**（因子太少，早停只长出 ~5 棵树，ŷ 粗）；② **24→32 相变**（树数从个位数跳到数百，IC 阶跃 +0.015）；③ **k≥32 平台**，边际增益 64→96→128 递减、**96~128 归零**，**160 见顶 .1244、203 回落 .1241**（过拟合起点）。
→ **64 ≈ 峰值 98%**，是「3× 更轻的因子依赖 vs 噪声级 IC 代价」的工程甜点。（曲线见 `ml/topk_sweep.png` / `topk_sweep.csv`）

> 易混点：表里**逐档边际增益 ΔIC**（台阶高度/斜率）≠ **IC 水平**（楼层）。IC 是累积量，故 160 楼层最高；但 ΔIC 在 64 之后趋平 → 64 是「效率」拐点而非「精度」最优。

### 8.2 203 vs 64：为何不用 l2 更低的全因子模型

203 因子 valid_l2=3.4328 < 64 因子 3.4375（低 0.14%），但**样本外 test IC 仅 +0.1243 vs +0.1217（Δ+0.0026，噪声级）**。
valid_l2 更低 ≠ 选股更强：l2 是逐样本回归误差，IC 是逐日截面**排序**相关——口径不同。冗余因子降 l2 靠拟合标签噪声，未转化为跨日选股力。故取 64：**因子依赖轻 3 倍，IC 代价噪声级**。

### 8.3 三个噪声级变量（不值得纠结）

| 变量 | 实测 | 结论 |
|---|---|---|
| 早停 patience | 80 → final 仅 8 棵树（valid 早期局部极小被过早锚定）；200 → ~231 棵，test IC **+0.110→+0.122** | **唯一非噪声**：patience 必须够大，否则 ŷ 粗、tie 多 |
| 随机种子 | seed 1↔42：入选因子 **56/64 重合**（8 个尾部近义因子互换 KLOW/MIN10/MIN20/STD5↔LOW0/MIN5/RSV60/SUMD5）；test IC ±0.0014、ICIR ±0.018 | 噪声级；多种子取均值待办 |
| num_threads | 并行直方图浮点求和顺序不定 → IC ±0.002 抖动 | 噪声级；要严格复现需固定单线程 |

---

## 9. 实盘信号 → 回测对接

### 9.1 链路

```
ml.run --run-id X                          # 训练 + 评估口径 pred_panel(止于 2026-03-31)
  → ml.predict_live --run-id X             # 实盘口径 pred_panel_live(补到最新因子日)
  → ml.export_signal --run-id X --source live --layout daily   # 每日 YYYY-MM-DD.txt
  → 回测项目 signal_dir 指向 signals/<run_id>/                  # signal_file 留空=daily 模式
```
回测（daily-realtime-backtest-pipeline）只读**排序**不读分数，故导出无损；daily 模式由 `signal_file` 为空触发（正则 `^\d{4}-\d{2}-\d{2}\.txt`），逐日读 `{date}.txt`。

### 9.2 「8 棵树假优势」回测实证（重要警示）

把 8 棵树版（full_gbdt）与 231 棵版（es200）都转 daily、同区间（2022-01-05~2026-04-01, top_k=100, T+1, netting）对比：

| | 8 棵树 | 231 棵 |
|---|---|---|
| 超额夏普 / IR | **1.19 / 1.19** | 1.09 / 1.09 |
| 年化超额 | **23.79%** | 22.15% |
| 超额最大回撤 | **31.72%** | 35.91% |
| 换手 | 53.59 | 49.26 |

**反直觉：树更少（IC 更低）的版本回测反而更稳、更均匀。** 机理：8 棵树 → ŷ 仅几个离散值 → 大量 tie → top-100 边界由 **tie-break（pandas 按股票代码字典序）** 决定 → 选出篮子被推向近随机/类指数 → 跟踪误差低、回撤小、年度收益均匀。
这是**伪优势**（代码序偏置的被动分散），非真 alpha。
→ 核心教训：**IC↑ ≠ top-100 等权回测↑**（评价口径 vs 组合口径的 gap）；ŷ 必须足够细（树够多）才能让排序真正反映模型观点，而非被 tie-break 接管。
