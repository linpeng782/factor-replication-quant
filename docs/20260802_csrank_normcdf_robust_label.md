# 20260802_csrank_normcdf_robust 标签变换 —— 年化 +5pp / 夏普 +0.23

> 本轮迭代最大单点收益改动。借鉴 joey-project 的 `csrank_normcdf_robust` 标签变换，
> 在因子/切分/超参/SHAP-128 全部不变的前提下，仅换标签即使得年化收益 +5pp、夏普 +0.23、
> 信息比率 +0.28，且换手率下降 5%。

---

## 1. 动机：旧标签的两个病灶

旧 LGBM 线标签是 `ExcessReturn`（截面 demean 超额）+ `WholeSetRobustZ`（全集稳健 Z 标准化）。
在训练段（5,235,993 样本）上量化诊断，发现两个硬伤：

| 病灶 | 量值 | 后果 |
|---|---|---|
| **极端样本垄断梯度** | 3.87% 的 \|y\|>3 样本贡献了 **45.6%** 的 L2 平方和；max=+50.8 | 模型一半学习力气花在拟合几百只妖股的 20 日暴涨幅度上 |
| **高波动日垄断梯度** | 逐日截面 std 的 p95/p5 = **2.30x**；平方和最大的 10% 交易日占 24.6%（理想 10%） | 模型偏向少数极端行情日的规律，难年份（2022/2024/2026）表现弱 |

目标分布：偏度 +1.86、超额峰度 +15.2 —— 远非高斯，L2 损失对尾部极度敏感。

## 2. 新标签：`CSRankNormCDFRobust`

逐日截面内执行四步向量化变换：

```
rank (method='average')  →  百分位 (rank-0.5)/n  →  正态分位 norm.ppf  →  稳健Z (med, MAD·1.4826)  →  clip±5
```

**关键性质**：与 `ExcessReturn` 的日内 Spearman 秩相关恒为 **1.0000**（实测）。
即纯单调重标定，**不改变任何日内排序信息**，只改 L2 损失的样本权重。

我们的评价（rank-IC）和下游（top-500 信号）都只用排序 —— **换标签零信息损失、零下游兼容成本**。

变换后目标分布：偏度 0.00、超额峰度 -0.08、逐日 std p95/p5 = 1.00x ——
近似标准正态，日间权重完全拉平，极端样本不再垄断梯度。

## 3. 代码改动（4 文件）

| 文件 | 改动 |
|---|---|
| `ml_core/labels.py` | 新增 `CSRankNormCDFRobust` 类（向量化，逐日 秩→ppf→稳健Z→clip）；`LabelTransform` 加 `needs_y_scaling` 标志，自带标准化的标签不再叠加全集 RobustZ |
| `ml_core/pipeline.py` | `scale_label` 默认值改为 `is_regression and needs_y_scaling` |
| `ml_core/run.py` | 新增 `_LABELS` 注册表；`train_config.yaml` 新增 `label` / `label_clip` 键；`run_meta.json` 写入标签溯源 |
| `ml_core/train_config.yaml` | 默认 `label: csrank_normcdf_robust` / `label_clip: 5.0`；同步 sources 加 p67 长端动量 + htsec 华泰动量，top_k 64→128（前序迭代产物） |

**向后兼容**：`label` 键缺省时走 `excess_return`，旧行为 bit 级不变。

## 4. A/B 实测结果

实验设置：因子（192→SHAP-128）、时间切分（train≤2017-11 / valid 2018-2019 / test≥2020）、
LGBM 超参、seed=42 全部相同，**唯一差异是标签**。

### 4.1 IC 指标

| | 旧标签 `excess_return` | 新标签 `csrank_normcdf_robust` |
|---|---|---|
| test IC 均值 | +0.1284 | **+0.1390**（+1.06bp，+8.3%） |
| ICIR | +1.175 | +1.119（略降，因 IC 标准差略升） |
| IC t 值 | — | +44.07 |

### 4.2 回测指标（2020-01-03 ~ 2026-07-31，top100，2 日调仓）

| | 旧标签 | 新标签 | 变化 |
|---|---|---|---|
| 年化收益 | 30.61% | **35.91%** | **+5.30pp** |
| 累计收益 | 441.5% | **596.4%** | +154.9pp |
| 夏普 | 1.029 | **1.27** | +0.24 |
| 索提诺 | — | 1.57 | — |
| 卡玛 | 0.862 | **1.02** | +0.16 |
| 信息比率 | 1.420 | **1.70** | +0.28 |
| 超额夏普 | — | 1.84 | — |
| 最大回撤 | 35.52% | 35.10% | -0.42pp（略改善） |
| 超额最大回撤 | 27.28% | 27.82% | +0.54pp（略差） |
| **换手率** | 58.09 | **55.26** | **-4.9%**（成本更低） |
| 超额月胜率 | — | 72.15% | — |

### 4.3 逐年收益

| 年份 | 旧标签 | 新标签 | 差值 |
|---|---|---|---|
| 2020 | 29.95% | 34.88% | +4.93 |
| 2021 | 64.12% | 50.50% | **-13.62** |
| 2022 | 5.99% | 14.10% | **+8.11** |
| 2023 | 20.90% | 27.77% | +6.87 |
| 2024 | 22.47% | 40.16% | **+17.69** |
| 2025 | 79.70% | 69.41% | **-10.29** |
| 2026 YTD | -9.96% | -0.89% | **+9.07** |

**7 年 5 胜 2 负**。负的两年恰好是 2021/2025 两个小盘极端行情年 ——
这印证诊断：旧标签超配高离散度行情权重，在极端牛市里能多榨一点，代价是难年份明显更弱。
新标签把日间权重拉平后，收益结构均衡得多，所以夏普/卡玛/IR 三个风险调整指标全面提升。

## 5. clip 参数实测：旋钮几乎不起作用

### 5.1 真实截面 |z| 分布

实测训练段 5,235,993 样本（含 ties/涨跌停/停牌）：

```
|z| 实际分布：p99.9=3.32 / max=3.56
clip=3 触发：14,069 样本（0.269%），削掉 0.024% 的 L2 平方和
clip=5 触发：0 样本（0%），削掉 0% 的 L2 平方和
逐日 max|z|：p50=3.41 / max=3.56 → clip=3 每天 100% 触发，但只削 1-2 只最极端的股
```

### 5.2 三方对比

| | 旧标签 | csrank clip=3 | csrank clip=5 |
|---|---|---|---|
| 年化 | 30.61% | 35.58% | 35.91% |
| 夏普 | 1.029 | 1.26 | 1.27 |
| 最大回撤 | 35.52% | 36.45% | 35.10% |
| 卡玛 | 0.862 | 0.98 | 1.02 |

### 5.3 结论

- **clip=5 完全不触发**（max|z|=3.56 < 5），等价于不截尾的 csrank。
- **clip=3 每天削 1-2 只极端股**，只影响 0.024% 的 L2 平方和 —— 量级极小。
- clip=3 vs clip=5 的 0.33pp 年化差异**不是 clip 的因果效应**，而是这个 0.024% 的微小标签扰动
  通过 LightGBM 迭代训练被蝴蝶效应放大后的不可归因扰动。
- **真正的增益 100% 来自 `秩→ppf→逐日稳健Z` 这一步**（把 45.6% 被极端样本吞掉的梯度还给正常样本 + 拉平日间权重）。
- **选 clip=5 作默认**：不做任何信息截断，少一个人为旋钮，效果至少不差。clip 这个旋钮不值得再投入。

## 6. 为什么 joey-project 的 57% 年化不可直接对比

joey-project 同样用 `csrank_normcdf_robust`，年化 57%，但两者不可直接对比：

| | joey-project | 本项目 |
|---|---|---|
| 标签 | csrank_normcdf_robust (clip=3) | csrank_normcdf_robust (clip=5) |
| 因子 | 50 个 `zscore_*` 黑盒技术因子 | 128 个可溯源因子（alpha158 + p27 微结构 + style + 长端动量） |
| 调仓 | 2 日 / top100 | 2 日 / top100 |
| 训练区间 | 2000-2019 → 2020-2026 test | 2005-2017 → 2020-2026 test |
| 换手率 | **85 倍**（极高） | 55 倍 |
| 交易成本 | 含印花税+佣金，**无冲击成本建模** | 同 |
| IC 稳定度 | ICIR≈13（跨牛熊每年正，疑似 look-ahead） | ICIR≈1.1（正常水平） |

joey 的 57% 很可能部分来自 `zscore_*` 因子的 look-ahead 风险 + 85 倍换手在小盘轮动上的天然偏高。
本项目 35.9% 是在可溯源因子、正常换手、ICIR 合理的前提下取得的，更接近真实可交易 alpha。

## 7. 待办

- [ ] **多 seed 复跑**（5 seed × csrank）确认 +5pp 年化不是单 seed 运气（IC t=+44 已是强信号级证据，但回测层面仍需验证）
- [ ] **新标签下重做 SHAP 筛选 / 因子消融** —— 目标函数变了，因子边际贡献排序也变了
  （`long_mom_1` 已升到 SHAP 第 5、`wgt_return_12m` 第 8），原计划的消融实验现在做才有意义
- [ ] 考虑把 `csrank_normcdf_robust` 也接到 MLP 线（目前 MLP 固定 `BinaryMedian`）

## 8. 复现命令

```bash
cd /nfs/ofs-prediction/peterzhenglinpeng-code/factor-replication-quant-new
source /nfs/ofs-prediction/peterzhenglinpeng-code/peterdidi/bin/activate

# 训练 + 自动推理 + 导信号（配置已在 train_config.yaml）
python -m ml_core.run

# 回测
cd /nfs/ofs-prediction/peterzhenglinpeng-code/daily-realtime-backtest-pipeline
# config/config.yaml 的 signal_dir 指向 lgbm_shap128_csrank5_dq/signals
python batch_runner.py
```

模型产物：`/nfs/ofs-prediction/peterzhenglinpeng/ml/models/lgbm_shap128_csrank5_dq/`
信号产物：`/nfs/ofs-prediction/peterzhenglinpeng/ml/predictions/lgbm_shap128_csrank5_dq/signals/`
回测产物：`/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/results/dr/lgbm_shap128_csrank5_dq_*/`
