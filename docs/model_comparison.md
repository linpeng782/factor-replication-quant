# 量化选股模型综合对比

> 2026-07-02。4 个模型（MLP / A+B 5050 / A LGBM固定 / B LGBM滚动），全周期 2020-01 ~ 2026-06 回测。

---

## 1. 模型详情

### MLP 

| 项 | 值 |
|---|---|
| 模型 ID | `mlp_a158_p27_top178` |
| 模型类型 | PyTorch FCNN（全连接神经网络） |
| 训练方式 | 一次性固定训练，时间切分 train/valid/test 三段 |
| 训练区间 | 2005-01-01 ~ 2017-11-30（train） |
| 验证区间 | 2018-01-01 ~ 2019-11-30（valid，用于早停） |
| 样本外区间 | 2020-01-01 ~ 2026-06-30（test） |
| 特征 | alpha158-dquant（158）+ microstructure（27）= 185 个，排除 3 个稀疏因子 → **178 个全特征** |
| 特征选择 | 无（`select_method=null`，全特征直喂） |
| 标签 | BinaryMedian（逐日截面中位数二分类 0/1） |
| 特征标准化 | DailyCrossSectionMAD（逐日截面 MAD 去极值 + zscore，无状态） |
| has_factor 策略 | ALL（要求全部 178 特征非 NaN，任一 NaN 即排除该行） |
| 预测面板 | `/nfs/ofs-prediction/peterzhenglinpeng/ml/predictions/mlp_a158_p27_top178/pred_panel_live.parquet` |
| 模型目录 | `/nfs/ofs-prediction/peterzhenglinpeng/ml/models/mlp_a158_p27_top178/` |
| 回测目录 | `results/mlp_a158_p27_top178_20200103_20260630_topk100_netting_vwapam_shift1_interval5/` |


### A — LGBM 固定训练

| 项 | 值 |
|---|---|
| 模型 ID | `lgbm_a158_p27_top64` |
| 模型类型 | LightGBM GBDT 回归 |
| 训练方式 | 一次性固定训练，时间切分 train/valid/test 三段 |
| 训练区间 | 2005-01-01 ~ 2017-11-30（train） |
| 验证区间 | 2018-01-01 ~ 2019-11-30（valid，用于早停） |
| 样本外区间 | 2020-01-01 ~ 2026-06-30（test） |
| 特征 | alpha158-dquant（158）+ microstructure（27）= 185 个 |
| 特征选择 | SHAP GBDT 选 top-64 |
| 标签 | ExcessReturn（截面等权 demean 超额收益，回归目标） |
| 特征标准化 | WholeSetRobustZ（全样本 per-feature Robust Z-score，有状态持久化） |
| has_factor 策略 | NONE（LGBM 原生吃 NaN，不剔除缺失因子行） |
| 预测面板 | `/nfs/ofs-prediction/peterzhenglinpeng/ml/predictions/lgbm_a158_p27_top64/pred_panel_live.parquet` |
| 模型目录 | `/nfs/ofs-prediction/peterzhenglinpeng/ml/models/lgbm_a158_p27_top64/` |
| 回测目录 | `results/lgbm_a158_p27_top64_20200103_20260630_topk100_netting_vwapam_shift1_interval5/` |

---

## 2. 训练超参数

### MLP 超参

| 参数 | 值 | 来源 |
|---|---|---|
| 网络结构 | 178 → 80 → 20 → 1 | `model.py:_build_mlp` |
| 激活函数 | Tanh | `model.py:_build_mlp` |
| Dropout | 0.3（两个隐藏层后） | `model.py:_build_mlp` |
| 权重初始化 | Xavier Uniform + bias zeros | `model.py:_build_mlp` |
| 输出激活 | Sigmoid（概率 0~1） | `model.py:MLPAdapter.predict` |
| 优化器 | Adam | `model.py:MLPAdapter.__init__` |
| 学习率 | 0.001 | `train_config.yaml: mlp.lr` |
| 权重衰减 | 0.00001 | `train_config.yaml: mlp.weight_decay` |
| 损失函数 | BCEWithLogitsLoss（二分类交叉熵） | `model.py:MLPAdapter.fit` |
| 早停耐心 | 15 epochs（valid loss 不降则停） | `train_config.yaml: mlp.patience` |
| 最大 epoch | 100 | `train_config.yaml: mlp.max_epochs` |
| batch_size | 8192 | `train_config.yaml: mlp.batch_size` |
| 设备 | 自动 cuda/cpu | `train_config.yaml: mlp.device` |

> 超参定义位置：`ml_core/model.py` 第 140-170 行，`ml_core/train_config.yaml` 第 52-58 行。

### LGBM 超参

| 参数 | 值 | 来源 |
|---|---|---|
| objective | regression（MSE 回归） | `model.py:LGBM_DEFAULT_PARAMS` |
| metric | l2（valid 评估指标） | `model.py:LGBM_DEFAULT_PARAMS` |
| boosting_type | gbdt | `model.py:LGBM_DEFAULT_PARAMS` |
| learning_rate | 0.05 | `train_config.yaml: lgbm.learning_rate` |
| num_leaves | 31 | `train_config.yaml: lgbm.num_leaves` |
| min_child_samples | 200 | `model.py:LGBM_DEFAULT_PARAMS` |
| feature_fraction | 0.8 | `model.py:LGBM_DEFAULT_PARAMS` |
| bagging_fraction | 0.8 | `model.py:LGBM_DEFAULT_PARAMS` |
| bagging_freq | 1 | `model.py:LGBM_DEFAULT_PARAMS` |
| lambda_l1 | 0.0 | `model.py:LGBM_DEFAULT_PARAMS` |
| lambda_l2 | 0.0 | `model.py:LGBM_DEFAULT_PARAMS` |
| max_depth | -1（无限制） | LightGBM 默认 |
| max_bin | 255 | LightGBM 默认 |
| num_threads | 64 | `train_config.yaml: lgbm.num_threads` |
| seed | 42 | `train_config.yaml: lgbm.seed` |
| deterministic | True | `model.py:LGBM_DEFAULT_PARAMS` |
| force_row_wise | True | `model.py:LGBM_DEFAULT_PARAMS` |
| num_boost_round | 1000（最大迭代轮数） | `model.py:LGBM_NUM_BOOST_ROUND` |
| early_stopping_rounds | 200（valid l2 连续 200 轮不降则停） | `model.py:LGBM_EARLY_STOPPING` |
| **best_iteration（A 固定）** | **406**（有效早停） | `model.txt` 树数 |
| best_iteration（B 滚动） | 949~1000（几乎跑满，未有效早停） | 各年 `model.txt` |

> 超参定义位置：`ml_core/model.py` 第 42-50 行（`LGBM_DEFAULT_PARAMS`），`ml_core/train_config.yaml` 第 47-51 行（`lgbm:` 段）。
>
> **best_iteration 差异**：A 固定在 406 轮有效早停（时间切分 valid 是真外样本），B 滚动几乎全部跑满 1000 轮（`split_mode=stock` 股票随机切分，valid 与 train 分布过于一致，无法触发早停）。

---

## 3. 全局绩效（2020-01 ~ 2026-06，6.5 年）

### 绩效对比

| 指标 | MLP | LGBM |
|---|---|---|
| **累计收益** | **550.53%** | 393.33% |
| **年化收益** | **35.04%** | 29.18% |
| 最大回撤 | -38.26% | **-38.12%** |
| **夏普比率** | **1.20** | 1.03 |
| **卡玛比率** | **0.92** | 0.77 |
| **信息比率** | **1.40** | 1.24 |
| 换手率 | 33.64% | 33.33% |

---

## 4. 逐年收益 vs 私募量化选股基准

> 数据来源：私募量化【量化选股】策略样本统计（逐年收益分布）。样本数为当年有业绩记录的私募产品数量。
>
> 评价规则：收益与私募中位数差异 > +3% = 🔴 跑赢；差异 < -3% = 🟢 跑输；差异在 ±3% 以内 = 🟡 接近（A 股配色：红涨绿跌）

| 年份 | 私募样本数 | 私募中位数 | MLP | MLP评价 | LGBM | LGBM评价 |
|---|---|---|---|---|---|---|
| 2020 | 3 | 🟠 +48.83% | +35.58% | 🟢 跑输 | +29.29% | 🟢 跑输 |
| 2021 | 5 | 🟠 +40.78% | +49.72% | 🔴 跑赢 | +55.21% | 🔴 跑赢 |
| 2022 | 16 | 🟠 -2.36% | +16.30% | 🔴 跑赢 | +3.41% | 🔴 跑赢 |
| 2023 | 23 | 🟠 +11.66% | +21.27% | 🔴 跑赢 | +19.00% | 🔴 跑赢 |
| 2024 | 30 | 🟠 +15.62% | +32.29% | 🔴 跑赢 | +10.11% | 🟡 接近 |
| 2025 | 38 | 🟠 +58.29% | +72.05% | 🔴 跑赢 | +56.88% | 🟡 接近 |
| 2026 | 48 | 🟠 +17.14% | -0.17% | 🟢 跑输 | +15.64% | 🟡 接近 |

### 评价汇总

| 模型 | 🔴 跑赢 | 🟡 接近 | 🟢 跑输 |
|---|---|---|---|
| **MLP** | **5 年**（2021-2025） | 0 | 2 年（2020/2026） |
| LGBM | 3 年（2021-2023） | 3 年（2024-2026） | 1 年（2020） |

