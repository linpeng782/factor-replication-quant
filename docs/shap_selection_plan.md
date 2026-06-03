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
| ML 数据集构建器 (因子+标签→矩阵) | ❌ 缺 |
| LightGBM/DART 训练 | ❌ 缺 |
| SHAP 计算 | ❌ 缺 |
| GRU / 组合优化器 | ❌ 缺（Phase 5）|

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

### Phase 3 — LightGBM(DART) baseline + SHAP 筛选（核心）
- **3a** 训练 LightGBM(`boosting='dart'`) 全 221 因子 → 超额收益(回归 MSE)，5 种子平均。
- **3b** `shap.TreeExplainer` → 抽样 10⁵ 行 → `mean(|SHAP|)` → 221 因子重要性排序。
- **3c** 选 **top-K**（扫 K=32/64/100）；**年度滚动**重选（十三§3.4）。
- 产出：`shap_importance_<year>.csv`、`selected_factors_<year>.csv`。

### Phase 4 — 验证：SHAP 选择 vs 全量 vs 我们的统计选择
- 用 top-K 重训 LightGBM(DART) → OOS：IC/ICIR/多头超额/多空。
- **对照 3 组**：① 全 221 baseline；② SHAP top-K；③ 我们之前的相关性/ICIR 统计选择集。
- 关键看点：**① SHAP 选的少因子能否 ≈ 全量？② cxl 基本面 / kysec 分钟因子 进没进 top-K？**（这是我们相对国金 Alpha158-only 的增量）

### Phase 5（扩展，可后做）
- 加 **GRU**（SHAP 用 GradientExplainer）；**GBDT+NN 合成**；**因子+标签中性化**(neu, 十三§四)；
- **马科维茨 TE≤5% 组合优化** → 指数增强策略；月频回测(手续费单边千二)。

## 四、MVP 最小可行路径（建议先做）
**Phase 1（超额标签 + 中证1000成分股）→ Phase 2（数据集）→ Phase 3（LightGBM+DART+SHAP）→ Phase 4（验证）**。
先不碰 NN / 指增组合优化。一条链跑通即可回答："SHAP 在我们 221 因子上选出哪 top-K、cxl/kysec 占几席、少因子能否打平全量"。

## 五、落点
- 建议新建 modeling 仓 或 `factor-repilcation-quant/modeling/`。
- 新脚本：`excess_labels.py`(stock-data-fetching)、`build_ml_dataset.py`、`train_lgbm_shap.py`、`eval_selection.py`。
- 复用：221 neu/raw 因子、本地 minute/daily 数据、rqdatac(13522652015)。

## 六、与国金的差异（诚实标注）
- 国金只在 **Alpha158(158)** 上做；我们在 **221**（+cxl基本面 +kysec分钟）上做 → 增量看点在此。
- 国金标签=超额收益；我们现有绝对收益 → **Phase 1 必须补**。
- 预处理我们现用 MAD+截面zscore（旧, 之九口径）；十§一已更新为 **GBDT 整体 RobustZScore** → Phase 2 改。
