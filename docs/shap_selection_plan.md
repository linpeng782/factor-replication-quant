# SHAP 选股复现 执行计划

> 复现国金「Alpha掘金」九/十/十三：**SHAP 特征筛选 → GBDT(+NN) 选股 → 指数增强**。
> 在我们 221 因子库上做（alpha158 158 + cxl 22 + kysec 41）；增量价值 = 基本面(cxl) + 分钟微观结构(kysec) 能否进 SHAP top-K。
> 出处标注：九=20231212多模型指增；十=20240328全流程消融；十三=20240909特征工程。

## 一、全局设置（每条标注出处 = 防编造 + 可追溯）

| 环节 | 取值 | 出处 |
|------|------|------|
| 预测标签 | 未来 20 日**超额收益率**（个股−基准）| 十§一(超额>绝对)、十三§3.2(20日) |
| 特征预处理·GBDT | **整体数据集 RobustZScore**（截面处理会损害 LightGBM）| 十§一 图1-3 |
| 特征预处理·NN | 特征 RobustZScore + **标签 CSRank**(截面排序) | 十§一 图4-6 |
| 任务类型 | **回归**(MSE)，非分类 | 十§四、§五 |
| 损失函数 | **MSE**（IC/RankIC 损失无明显增益）| 十§五 |
| 树集成 | **DART** 优先（DART>GBDT>RF）| 十§六 图21 |
| 模型 | LightGBM + GRU(NN) | 九§1、十全篇 |
| 模型训练窗口 | **一次性 8年训+2年验**（优于滚动/扩展）| 十§三 图13 |
| 特征选择频率 | **年度滚动重选**（哪些因子重要会漂移）| 十三§3.4 |
| 随机种子 | **5 个取平均** | 十§一、十三§3.2 |
| 股票池 | 视基准：LGBM+沪深300→成分股；全A亦可。选择实验用全A | 十§二、十三§3.2 |
| SHAP 工具 | LightGBM→**TreeExplainer**；GRU→GradientExplainer | 十三§3.2 |
| SHAP 重要性 | **mean(\|SHAP\|)**，随机抽样 **10⁵** 样本 | 十三§3.2 |
| 选择规则 | 取重要性 **top-K**（报告 64/158）；可选 MMR 去冗余(效果有限) | 十三§3.2 |
| 中性化 | 因子 + 标签 都中性化（作为改进项）| 十三§四 |
| 回测 | 月频调仓、T+1、因子后推 1 天、手续费单边千二 | 十§7 |
| 指增组合优化 | 马科维茨均值方差，跟踪误差 ≤5% | 十§7.2 |

## 二、我们已有 vs 缺口

| 组件 | 现状 |
|------|------|
| 221 因子库 (raw/cleaned/neu, 网格对齐) | ✅ 有 |
| 绝对 forward_return 标签 (vwap) | ✅ 有 → ❌ **缺超额收益标签** |
| 指数成分股 membership (300/500/1000) | ❌ 缺 |
| RobustZScore 预处理 | ❌ 缺（现为 MAD+截面zscore，与十§一GBDT结论不符）|
| ML 数据集构建器 (因子+标签→矩阵) | ✅ 有 `ml/dataset.py`（203因子, 8+2+test, embargo） |
| LightGBM 训练 + gain importance 选 64 | ✅ 有，已实跑（`full_gbdt_es200` test IC≈+0.122，top_k=64 为效率拐点）|
| SHAP 计算 (`ml/select.py:select_by_shap`) | ⚠️ **代码已写好且接入 run.py，但 shap 未安装、从未实跑** |
| DART / GRU / 组合优化器 | ❌ 缺（Phase 5）|

## 三、分阶段执行计划

### Phase 0 — 补库 + 统一总账（前提，可暂用 203）
- (可选) 跑 kysec paper_33(18) → 221；重跑 `build_factor_inventory` → 全库 ic_series。
- 用户已说 paper_33 不急 → **可先用现有 203 因子起步**。

### Phase 1 — 标签 + 股票池（数据缺口，必做）
- **1a 超额收益标签**：`未来20日超额 = 个股20日收益 − 基准20日收益`。先选**一个基准（建议中证1000**，十中表现最好且全A训练 OK）。
  - 产出 `market-data/labels/excess_return_20d_csi1000.parquet`
- **1b 指数成分股**：rqdatac `index_components`(时点) → 历史 membership 面板。
  - 产出 `market-data/universe/csi1000_membership.parquet`

### Phase 2 — 数据集构建器 `build_ml_dataset.py`
- 输入：选定 universe + 区间；**raw** 因子（非 neu，预处理按十§一）；超额标签。
- 处理：停牌→NaN(十§一)；**特征整体 RobustZScore**(用训练集统计量, 防泄露, 十§一)；停牌/缺失按模型类型处理（GBDT 可留 NaN）。
- 切分：**一次性 8年训 + 2年验 + 测试**(十§三)。
- 产出：`(样本=日×股) × (221特征 + 超额标签)` 矩阵 + train/val/test 索引。

### Phase 3 — LightGBM baseline + SHAP 筛选（核心，框架已就位）
- **3a** ✅ 已完成：LightGBM(GBDT) 全 203 因子 → gain importance → top-64（`full_gbdt_es200`）。
- **3b** ⚠️ 待跑：`shap.TreeExplainer` → 抽样 10⁵ 行 → `mean(|SHAP|)` → 203 因子重要性排序。代码已在 `ml/select.py:select_by_shap`。
- **3c** 先**一次性**筛选（与我们 one-shot 训练一致）；年度滚动(十三§3.4) 待对照跑通后再权衡（成本高）。

### Phase 4 — SHAP vs gain 对照实验（可执行 · 本阶段重点）

**前置**：`pip install shap`（≥0.44，对 LightGBM NaN 支持稳定）。先小样本冒烟确认 `shap_values` 维度/无报错。

**命令**（与 gain 基线同口径，仅换 Stage 1 筛选方法）：
```bash
python -m ml.run --select-method shap --top-k 64 --date-sample 5 --max-features 60 --run-id shap_smoke  # 冒烟
python -m ml.run --select-method shap --top-k 64 --run-id shap_top64                                    # 全量
```

**三个看点**（对照基线 `full_gbdt_es200`）：

| # | 问题 | 判读标准 |
|---|---|---|
| Q1 | SHAP top-64 与 gain top-64 **重合度** | ≥56/64 → 与研报一致(gain 已够用)；列出仅 SHAP / 仅 gain 名单看是否仅尾部近义因子互换 |
| Q2 | SHAP 选的 64 因子重训 **test IC** vs gain +0.122 | 预期 **\|ΔIC\|<0.002 噪声级**。十三§3 原文：纯 GBDT 下 gain 已具特征选择力，SHAP 主要为喂 NN |
| Q3 | **★cxl 基本面 / kysec 分钟因子在 SHAP top-64 各占几席** | 直接回答「203 vs 国金 Alpha158-only 增量值不值」——本实验核心产出 |

> 预期管理：**Q2 打平是符合研报的正常结果，不是失败**；SHAP 真正增益在 NN 分支(Phase 5)。本实验价值在 Q3(增量证据) + Q1(方法论交叉验证)。

**可选增强 — MMR 去冗余**（比研报更有意义）：研报在 Alpha158 上称 MMR 效果有限，但 plan §8 实测我们 203 因子仅 ~34 个独立 alpha、高度冗余，纯 `mean(|SHAP|)` 会扎堆选近义因子。MMR(十三§3.2) 让 64 名额覆盖更多独立信息源：
```
MMR(Dᵢ) = mean(|SHAP|)(Dᵢ) × (1 − max_{Dⱼ∈已选} |Spearman(Dᵢ, Dⱼ)|)   # 贪心逐个入选
```
仅在上面对照跑通后再做，看 IC/ICIR 是否提升。

**执行清单**：
- [ ] `pip install shap` → 冒烟 `shap_smoke`
- [ ] 全量 `shap_top64`
- [ ] 填 Q1 重合度 + 仅 SHAP/仅 gain 名单
- [ ] 填 Q2 IC 对照（gain +0.122 vs SHAP）
- [ ] 填 Q3 cxl/kysec 占比（★核心）
- [ ] （可选）MMR 组 `shap_mmr64`
- [ ] 结论回写 `ml_pipeline_plan.md §6`

### Phase 5（扩展，可后做）
- 加 **GRU**（SHAP 用 GradientExplainer）；**GBDT+NN 合成**；**因子+标签中性化**(neu, 十三§四)；
- **马科维茨 TE≤5% 组合优化** → 指数增强策略；月频回测(手续费单边千二)。

## 四、MVP 最小可行路径（当前进度）
Phase 1（超额标签，等权 demean）→ Phase 2（数据集）→ Phase 3a（LightGBM + gain 选 64）**均已完成**，链路已跑通（test IC≈+0.122）。
**当前唯一待做 = Phase 4 的 SHAP 对照实验**（装包 + 实跑 + 填三个看点表），即可回答："SHAP 在 203 因子上选哪 top-64、cxl/kysec 占几席、与 gain 是否打平"。NN / 指增组合优化仍留 Phase 5。

## 五、落点
- 建议新建 modeling 仓 或 `factor-repilcation-quant/modeling/`。
- 新脚本：`excess_labels.py`(stock-data-fetching)、`build_ml_dataset.py`、`train_lgbm_shap.py`、`eval_selection.py`。
- 复用：221 neu/raw 因子、本地 minute/daily 数据、rqdatac(13522652015)。

## 六、与国金的差异（诚实标注）
- 国金只在 **Alpha158(158)** 上做；我们在 **221**（+cxl基本面 +kysec分钟）上做 → 增量看点在此。
- 国金标签=超额收益；我们现有绝对收益 → **Phase 1 必须补**。
- 预处理我们现用 MAD+截面zscore（旧, 之九口径）；十§一已更新为 **GBDT 整体 RobustZScore** → Phase 2 改。
