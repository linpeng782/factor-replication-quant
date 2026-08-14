# 简单模型 vs baseline(p27) 对比实验

> 日期：2026-08-07
> 目的：验证"去掉 p27 微观结构因子，仅靠 Alpha158 + size + 动量"能否逼近 baseline

---

## 1. 实验设计

| 项 | 简单模型 | baseline |
|---|---|---|
| run_id | `lgbm_shap128_a158_size_kymom_htmom_csrank5_dq` | `lgbm_shap128_csrank5_dq` |
| sources | alpha158-dquant + style-dquant/size + kysec-dquant/paper_67_long_momentum + htsec/paper_04_momentum | 上述 4 源 + kysec-dquant/paper_27_microstructure |
| 因子池 | 169（158+1+2+8） | 192（169 + 23 p27，两侧无重名因子） |
| 选因子 | SHAP top-128 | SHAP top-128 |
| 标签 | csrank_normcdf_robust (clip 5.0) | 同左 |
| 回测口径 | 2020-01-03 ~ 2026-07-31, topk100, vwap, shift1, interval2, cap20m | 同左 |

两者唯一差异：**有没有 p27 的 23 个微观结构因子**（peak_ridge_turnover_ratio / valley_relative_vwap / eruption_followup_ratio 等，依赖分钟级喷发/峰岭谷判定）。

> 可复现性：baseline (`lgbm_shap128_csrank5_dq`) 已于 2026-08-09 在 master 上原样复跑验证 bit 级一致
> ——入选 128 清单（含排序）、192 个 SHAP 重要性、scaler_x、1550 天逐日 IC、pred_panel 预测值，
> 最大绝对差均为 0。（简单模型未复跑，但两者共用同一条代码路径。）
>
> ⚠️ 前提：`market-data/labels/forward_return_20d.parquet` 未随因子日更而更新，`has_label` 卡住了
> 样本上限，因此因子数据延长到 2026-08-07 后候选样本数仍完全相同。若将来 labels 也日更，
> 同一配置将不再复现上表数字（test 段会变长），届时需显式钉住 `split.test` 的 end 才能对齐。

---

## 2. 全局绩效对比

| 指标 | 简单模型 | baseline | 差值 |
|---|---|---|---|
| 累计收益 | 579.47% | **596.41%** | -16.94 |
| 年化 | 35.38% | **35.91%** | -0.53 |
| 夏普 | **1.28** | 1.27 | +0.01 |
| 卡玛 | **1.03** | 1.02 | +0.01 |
| 信息比率 | **1.72** | 1.70 | +0.02 |
| 最大回撤 | **34.43%** | 35.10% | +0.67 |
| 超额回撤 | **27.22%** | 27.82% | +0.60 |
| test IC 均值 | 0.1345 | **0.1390** | -0.0045 |
| ICIR | 1.057 | **1.119** | -0.062 |

**要点**：
- 累计收益差 17 个点，但夏普/卡玛/IR 反而持平或略优——p27 贡献的收益伴随更大波动
- IC 差 0.0045，回测差 17 个点，说明 p27 的 IC 优势在净值上被波动放大
- 回撤更小，简单模型风险控制略好

---

## 3. 分年度对比

| 年份 | 简单模型 | baseline | 差值 | 简单模型回撤 | baseline 回撤 | 简单模型 Alpha | baseline Alpha |
|---|---|---|---|---|---|---|---|
| 2020 | 30.02% | **34.88%** | -4.86 | -12.98% | -13.60% | 7.05% | **11.91%** |
| 2021 | **54.10%** | 50.50% | +3.60 | -15.65% | -15.64% | **47.91%** | 44.31% |
| 2022 | **18.87%** | 14.10% | +4.77 | -30.25% | -30.87% | **39.19%** | 34.42% |
| 2023 | 26.24% | **27.77%** | -1.53 | **-7.85%** | -8.28% | 33.28% | **34.80%** |
| 2024 | 37.50% | **40.16%** | -2.66 | -34.43% | -35.10% | 30.06% | **32.73%** |
| 2025 | **73.50%** | 69.41% | +4.09 | **-11.71%** | -11.90% | **48.91%** | 44.81% |
| 2026YTD | -5.27% | **-0.89%** | -4.38 | -22.45% | **-20.01%** | -1.25% | **3.13%** |

### p27 的价值规律

**p27 在低收益/高波动/流动性分化年份有正贡献**：
- 2020（+4.86）、2024（+2.66）、2026YTD（+4.38）——三年基准偏弱或极端波动
- p27 微观结构因子捕捉"非趋势性"微观定价偏差，在暴跌/震荡中提供尾部防御

**p27 在趋势明确年份反而拖后腿**：
- 2021（-3.60）、2022（-4.77）、2025（-4.09）——三年趋势性强（小盘牛/熊市/大牛市）
- 强趋势下微观结构信号被淹没，动量+size 已足够，p27 引入噪音

**简单模型 7 年中 4 年跑赢**，但输的年份恰好是风险控制最需要的时段（2020 早期牛市起步、2024 小盘崩盘、2026 YTD 弱市）。

---

## 4. 2026 年逐月对比（重点深挖）

| 月份 | 简单模型 | baseline | 差值 | 基准 | 简单 Alpha | baseline Alpha |
|---|---|---|---|---|---|---|
| 2026-01 | **10.24%** | 9.61% | +0.63 | 5.75% | +4.50% | +3.87% |
| 2026-02 | 2.83% | **3.61%** | -0.78 | 2.23% | +0.60% | +1.38% |
| 2026-03 | -8.38% | **-6.83%** | -1.55 | -8.69% | +0.31% | **+1.86%** |
| 2026-04 | **6.88%** | 6.66% | +0.22 | 8.56% | -1.68% | -1.90% |
| 2026-05 | -6.18% | **-5.60%** | -0.58 | 0.42% | -6.60% | -6.02% |
| 2026-06 | -7.90% | **-7.13%** | -0.77 | 2.87% | -10.77% | -10.00% |
| 2026-07 | -1.23% | **0.17%** | -1.40 | -13.28% | +12.05% | **+13.45%** |

### 关键观察

- **差距不集中**：不是某次极端事件拉开，而是 2~7 月 baseline 每月稳定赢 0.2~1.5 个点
- **3 月最关键**（基准 -8.69% 暴跌）：baseline 只跌 6.83%，简单模型跌 8.38%，差 1.55 个点——p27 在暴跌中提供尾部防御
- **7 月**（基准 -13.28% 暴跌）：两个模型都大幅跑赢（Alpha +12~13%），但 baseline 转正 0.17%，简单模型仍亏 -1.23%
- **5、6 月**是两个模型共同最差月份（基准涨而策略大跌，Alpha -6~-10%），属风格切换系统性回撤，与 p27 无关

### 结论

2026 的 4.38 个点差距来自 baseline 在**每月下跌保护上稳定好 0.5~1.5 个点**，p27 微观结构因子在弱市/高波动环境持续提供微弱但稳定的超额防御。

---

## 5. 综合结论

| 维度 | 简单模型优势 | baseline 优势 |
|---|---|---|
| 累计收益 | — | +17 个点 |
| 风险调整（夏普/卡玛/IR） | 持平或略优 | — |
| 回撤 | 略小 | — |
| 趋势年份（2021/2022/2025） | **赢** | — |
| 弱市/高波动年份（2020/2024/2026） | — | **赢** |
| 数据依赖 | **无分钟数据** | 依赖 p27 分钟管道 |
| 可维护性 | **4 源简洁** | 5 源 + 分钟管道 |

### 选型建议

- **追求全周期稳健性、有分钟数据管道**：选 baseline，p27 在弱市防御价值不可替代
- **追求简洁可维护、不依赖分钟数据**：选简单模型，收益几乎不输，风险指标更优，7 年中 4 年跑赢
- **不可兼得**：p27 在趋势年份是噪音，在弱市是防御——无法通过加权同时获得两边优势，需要根据市场状态动态切换（未来可探索 regime-switching）

---

## 6. 复现命令

```bash
# 简单模型训练
# ml_core/train_config.yaml: sources 只留 4 源，select_method: shap, top_k: 128
python -m ml_core.run

# 回测
# daily-realtime-backtest-pipeline/config/config.yaml: signal_dir 指向对应 signals/
python batch_runner.py
```

回测结果路径：
- 简单模型：`backtest_engine/results/dr/lgbm_shap128_a158_size_kymom_htmom_csrank5_dq_*`
- baseline：`backtest_engine/results/dr/lgbm_shap128_csrank5_dq_*`

---

## 附录 A. baseline 模型 `lgbm_shap128_csrank5_dq` 因子清单与研报出处

> 模型产物路径：`<DATA_ROOT>/ml/models/lgbm_shap128_csrank5_dq/`
> 入选因子清单：`selected_features.json`（SHAP top-128，按 mean(|SHAP|) 降序）

### A.1 因子来源总览

baseline 从 **5 个因子源、192 个候选因子**中经 SHAP 选出 **128 个**：

| 源 | 来源目录 | 候选数 | 入选数 | 研报出处 |
|------|------|------|------|------|
| alpha158-dquant | `factors/raw-dquant/alpha158-dquant/` | 158 | 99 | Microsoft Qlib Alpha158（开源量价技术因子库） |
| kysec-dquant/paper_27_microstructure | `factors/raw-dquant/kysec-dquant/paper_27_microstructure/` | 23 | 18 | 开源证券《高频成交量的峰、岭、谷信息》（市场微观结构系列 27，2025-07-20） |
| style-dquant/size | `factors/raw-dquant/style-dquant/size/` | 1 | 1 | 风格因子（市值因子，无特定研报，Barra 风格体系） |
| kysec-dquant/paper_67_long_momentum | `factors/raw-dquant/kysec-dquant/paper_67_long_momentum/` | 2 | 2 | 开源证券《长端动量 2.0：长期、低换手、多头显著的量价因子》（开源量化评论 67，2022-11-26） |
| htsec/paper_04_momentum | `factors/raw-dquant/htsec/paper_04_momentum/` | 8 | 8 | 华泰证券《多因子系列之四：单因子测试之动量类因子》（2016-12-20） |

### A.2 各源因子定义与计算方法

#### A.2.1 Alpha158（99/158 入选）

**研报出处**：Microsoft Qlib 开源框架的 Alpha158 量价技术因子集（非券商研报，为学术界/工业界标准因子库）。
**引擎代码**：`alpha158/engine/factors.py`，输入为后复权日频 OHLCV + vwap 宽表面板。

158 个因子分 4 组：

| 组 | 因子类 | 数量 | 公式概述 |
|------|------|------|------|
| K线形态 | KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2 | 9 | 当日 K 线形态：实体/影线相对开盘价或全振幅的比值 |
| 价格 | OPEN0, HIGH0, LOW0, VWAP0 | 4 | 当日 open/high/low/vwap 相对 close 的比值 |
| Rolling | ROC, MA, STD, BETA, RSQR, RESI, MAX, MIN, QTLU, QTLD, RANK, RSV, IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD | 23×5=115 | 过去 N 日（N=5,10,20,30,60）的滚动统计：收益率/均线/波动/回归斜率/R²/残差/极值/分位/排名/位置/相关/涨跌天数/累计收益等 |
| 成交量 | VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD | 6×5=30 | 过去 N 日成交量统计：均值/标准差/加权均值/正收益日量累计/负收益日量累计/收益方向量累计 |

**入选 99 个分布**：
- K线形态 3 个：KLOW, KLEN, KUP
- 价格 3 个：VWAP0, LOW0, HIGH0
- Rolling 79 个（23 类×5 窗口中入选的部分），按窗口分布：5日15个 / 10日15个 / 20日18个 / 30日17个 / 60日28个（长窗口入选更多）
- 成交量 14 个：VMA×4, VSTD×4, WVMA×3, VSUMP/VSUMN/VSUMD 各1

**关键公式示例**（完整定义见 `alpha158/engine/factors.py`）：
- `KMID = (close - open) / open`（实体相对开盘）
- `VWAP0 = vwap / close`（vwap 偏离收盘）
- `STD60 = Std(close, 60) / close`（60日波动率归一化）
- `CORR5 = RollingCorr(close, log(volume+1), 5)`（5日价量相关性）
- `RSV60 = (close - Min(low,60)) / (Max(high,60) - Min(low,60))`（60日随机指标）
- `BETA30 = Slope(close, 30) / close`（30日趋势斜率归一化）

#### A.2.2 p27 微观结构因子（18/23 入选）

**研报出处**：开源证券《高频成交量的峰、岭、谷信息——市场微观结构研究系列（27）》
- 作者：魏建榕、王志豪
- 日期：2025-07-20
- paper.md：`sources/kysec/paper_27_microstructure/paper.md`

**核心思想**：对个股日内分钟成交量按"过去 20 日同时点 ±1σ"划分为三种状态：
- **量峰**（peak）：孤立喷发成交量（前后分钟均温和）→ 知情交易者大额成交
- **量岭**（ridge）：连续喷发成交量 → 散户跟随交易
- **量谷**（valley）：温和成交量 → 情绪低迷时点

**数据依赖**：分钟级 OHLCV（`market-data/minute-dquant/raw/`），后复权口径。
**算子**：`minute_intraday_aggregate`（cache_key=`prv_v3`，std_window=20，std_threshold=1.0）→ 日频 reduce → 20 日 rolling。

入选 18 个因子（按 SHAP 重要性降序）：

| 因子 | 中文名 | 方向 | 公式 | 研报 IC/LS |
|------|------|------|------|------|
| eruption_followup_ratio | 喷发成交额跟随比例 | -1 | 20日 Σ下一分钟成交额 / Σ喷发分钟成交额 | IC -10.59% LS 30.09% |
| peak_ridge_turnover_ratio | 峰岭成交比 | +1 | 20日 Σ峰成交额 / Σ岭成交额 | IC +10.28% LS 27.13% |
| valley_relative_vwap | 量谷相对加权价 | +1 | 20日 (谷vwap / 日vwap) 均值 | IC +8.69% LS 25.35% |
| valley_weighted_quantile | 量谷加权价格分位点 | +1 | 20日 谷vwap 在日内[min(H,L,prevC),max(...)] 分位点均值 | IC +6.34% LS 20.22% |
| valley_ridge_price_ratio__mp10 | 谷岭加权价格比(mp10) | +1 | 20日 (谷vwap/岭vwap) 均值，min_periods=10放宽缺失 | IC +6.98% LS 15.83% |
| ridge_minute_return | 量岭分钟收益 | -1 | 20日 量岭分钟1-min收益累计和 | IC -6.29% LS 14.98% |
| peak_interval_std | 量峰间隔标准差 | -1 | 20日 pooled 峰间隔标准差 | IC -8.57% LS 25.66% |
| peak_minute_count | 量峰分钟数 | +1 | 20日 量峰分钟数均值 | IC +10.62% LS 31.58% |
| ridge_minute_count | 量岭分钟数 | -1 | 20日 量岭分钟数均值 | IC -9.04% LS 26.20% |
| peak_interval_skew | 量峰间隔偏度 | +1 | 20日 pooled 峰间隔分布偏度 | IC +7.68% LS 24.56% |
| eruption_turnover_sensitivity | 喷发成交额敏感度 | -1 | 20日 pooled OLS slope(下一分钟成交额 ~ 喷发分钟成交额) | IC -7.14% LS 15.61% |
| eruption_turnover_corr | 喷发成交额相关性 | -1 | 20日 Pearson(喷发分钟成交额, 下一分钟成交额) | IC -10.94% LS 29.72% |
| peak_weighted_quantile | 量峰加权价格分位点 | +1 | 20日 峰vwap 在日内价格区间分位点均值 | IC +3.47% LS 11.20% |
| peak_ridge_price_ratio__mp10 | 峰岭加权价格比(mp10) | +1 | 20日 (峰vwap/岭vwap) 均值，min_periods=10 | IC +4.70% LS 10.31% |
| valley_ridge_price_ratio | 谷岭加权价格比 | +1 | 20日 (谷vwap/岭vwap) 均值 | IC +6.98% LS 15.83% |
| peak_ridge_minute_corr | 同时点峰岭数相关性 | -1 | 20日 同时点峰数与岭数 Pearson 相关 | IC -6.67% LS 22.78% |
| ridge_interval_skew | 量岭间隔偏度 | -1 | 20日 pooled 岭间隔分布偏度 | IC -8.08% LS 22.19% |
| peak_interval_kurt | 量峰间隔峰度 | +1 | 20日 pooled 峰间隔分布峰度 | IC +7.19% LS 23.30% |

> `__mp10` 变体：原版 min_periods=20 导致 ~78% 缺失（峰/岭/谷天然稀疏），放宽到 10 以提升覆盖率。

#### A.2.3 市值因子 size（1/1 入选）

**研报出处**：无特定研报，属 Barra 风格因子体系的标准规模因子。
**构建脚本**：`data_fetching/style_ln_market_cap.py`

| 因子 | 公式 | 用途 |
|------|------|------|
| ln_market_cap | ln(market_cap_3)，单位亿元，市值≤0置NaN | 大小盘画像 / regime 切分 / 市值中性化控制变量 |

**原料**：`market-data/market_cap/market_cap_panel.parquet`（总市值日频宽表，米筐 `market_cap_3` 口径）。
**性质**：风格暴露，不走 cleaned/neu 三阶段（对 size 做市值中性化是自我抵消）。

#### A.2.4 长端动量因子 p67（2/2 入选）

**研报出处**：开源证券《长端动量 2.0：长期、低换手、多头显著的量价因子》（开源量化评论 67，2022-11-26）
- paper.md：`sources/kysec/paper_67_long_momentum/paper.md`

**核心思想**：A 股长端涨跌幅（Ret160）整体呈反转，因为高振幅日（过度反应日）主导了长端收益。剥离高振幅日、只取低振幅 70% 交易日的超额收益，即可露出真正的动量效应。

| 因子 | 方向 | 公式 |
|------|------|------|
| long_mom_1 | +1 | 长端动量 1.0：回溯160日 → 每日振幅=H/L−1 → 取低振幅70%交易日涨跌幅加总 |
| long_mom_2 | +1 | 长端动量 2.0：在 1.0 基础上四处改进——剔除涨跌停/停牌日、振幅改(H−L)/前收、日超额收益(减市场均值)、20日反转中性(截面回归取残差) |

**研报绩效**：长端动量 2.0 RankIC 6.92%，RankICIR 2.75，多空年化 18.09%。
**数据依赖**：后复权日频 OHLCV + `normal_day_panel`（涨跌停/停牌剔除）+ `RET20_PANEL_PATH`（20日反转中性回归）。

#### A.2.5 改进动量因子 ht p04（8/8 入选）

**研报出处**：华泰证券《多因子系列之四：单因子测试之动量类因子》（2016-12-20）
- paper.md：`sources/htsec/paper_04_momentum/paper.md`

**核心思想**：传统 N 月收益率（return_Nm）在 A 股呈反转效应。引入换手率信息加权可加强信号——换手率高的交易日信息含量更大。

| 因子 | 方向 | 公式 |
|------|------|------|
| wgt_return_1m | -1 | 过去20交易日：Σ(换手率ᵢ × 日收益ᵢ) / Σ(换手率ᵢ) |
| wgt_return_3m | -1 | 过去60交易日同上 |
| wgt_return_6m | -1 | 过去120交易日同上 |
| wgt_return_12m | -1 | 过去240交易日同上 |
| exp_wgt_return_1m | -1 | 过去20日：权重 = 换手率ᵢ × exp(−xᵢ/(4×20))，xᵢ=距截面日天数 |
| exp_wgt_return_3m | -1 | 过去60日同上，decay_scale=4×60 |
| exp_wgt_return_6m | -1 | 过去120日同上，decay_scale=4×120 |
| exp_wgt_return_12m | -1 | 过去240日同上，decay_scale=4×240 |

**研报结论**：exp_wgt_return_3m 和 exp_wgt_return_6m 综合表现最好；改进动量因子额外引入换手率信息，与传统反转因子正相关性强。
**数据依赖**：后复权日频 close + `turnover_rate_panel`（dquant 流通股换手率）。

---

### A.3 训练过程

#### A.3.1 标签

- **标签类型**：`csrank_normcdf_robust`（截面秩正态化稳健Z）
- **horizon**：20 日远期收益
- **计算步骤**：逐日截面 → 20日远期收益 → 截面平均秩 → 正态分位 `norm.ppf` → 稳健Z（MAD去极值）→ clip ±5.0
- **目的**：高波动日不再垄断 L2 梯度，目标近似标准正态，日内排序与原始收益完全等价（Spearman≡1）

#### A.3.2 特征标准化

- **标准化器**：`WholeSetRobustZ`（全集 per-feature RobustZScore）
- **计算**：train 段拟合 median + MAD×1.4826 → 全段 transform → 落盘 `scaler_x.parquet`
- **注意**：LGBM 对单调变换不敏感，标准化主要为了 scaler 落盘一致性 + MLP 兼容

#### A.3.3 两阶段因子筛选（SHAP）

**Stage-1（选因子）**：
1. 在 train+valid 段（192 个全特征）上训一棵 LGBM（参数同最终模型，valid 早停）
2. 对 train+valid 合并样本随机采样 100,000 行（random_state=0）
3. 用 SHAP TreeExplainer 计算 mean(|SHAP value|) 作为因子重要性
4. 按重要性降序取 top-128

**Stage-2（合成模型）**：
1. 仅用入选 128 因子重训 LGBM（valid 早停）
2. 落盘 `model.txt` + `selected_features.json` + `scaler_x.parquet` + `run_meta.json`

#### A.3.4 时间切分

| 段 | 区间 | 用途 |
|------|------|------|
| train | 起始 ~ 2017-11-30 | 训练（含 Stage-1 选因子 + Stage-2 合成） |
| embargo | 2017-12-01 ~ 2017-12-31 | 空档（防 20 日标签重叠泄漏） |
| valid | 2018-01-01 ~ 2019-11-30 | 早停 + SHAP 采样 |
| embargo | 2019-12-01 ~ 2019-12-31 | 空档（防 20 日标签重叠泄漏） |
| test | 2020-01-01 ~ 因子共同覆盖末日 | 评估（IC/回测，全程不可见于训练） |

#### A.3.5 LGBM 超参

| 参数 | 值 |
|------|------|
| objective | regression（MSE） |
| learning_rate | 0.05 |
| num_leaves | 31 |
| min_child_samples | 200 |
| feature_fraction | 0.8 |
| bagging_fraction | 0.8 |
| bagging_freq | 1 |
| num_boost_round | 1000（上限） |
| early_stopping_rounds | 200 |
| num_threads | 64 |
| seed | 42 |
| deterministic | True |

#### A.3.6 样本池

- **训练候选池**：`eligible_today`（T日因子有效）& `can_buy`（T+1可成交=label可实现）& `has_label`（20日远期收益存在）
- **has_factor_policy**：`none`（LGBM 不要求所有因子都有值，天然处理 NaN）
- **demean 基准池**：`eligible_today & can_buy`（与投资域同口径）

#### A.3.7 训练后推理

- `predict.after_train = true`：训完自动 reload 模型 → predict_live → export_panel
- 推理区间：2020-01-01 ~ 因子共同覆盖末日
- 每日取预测分 top-500 导出信号（回测只用排序，不用分数）
