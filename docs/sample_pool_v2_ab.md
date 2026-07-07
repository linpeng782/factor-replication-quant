# 样本池 v2（eligible_today）重定义与 A/B 验证

> 2026-07-07。ML 管线训练/预测池口径升级，单模型 A/B 重训 + topk100 回测验证。
> 结论：**test IC 持平，组合端年化 +3.38pp / 夏普 +0.13，7 个年份全部不输，去噪声样本确认有效。**

---

## 1. 背景与动机

dquant 数据迁移验证期间发现：旧池口径下，T 日停牌/ST 股票的「垃圾因子值」会进入训练与预测池。
典型案例（2026-07-01）：

- `000656.XSHE` 停牌日 volume=0 → VMA/VSTD 极端值，仍被打分进入预测池
- `002898.XSHE` 复牌大跌 → QTLD60/BETA5 极端值
- 旧预测池用 T+1 状态（can_buy）决定 T 日样本是否有效，隐含**未来信息**

## 2. 池口径变更（代码改动）

| 文件 | 改动 |
|---|---|
| `alpha_shared/cleaning/mask_loader.py` | 新增 `load_eligible_today_mask()`：T 日非 ST/非停牌/非新股，**无 shift、纯当日信息** |
| `ml_core/universe.py` | `Universe` 增加 `eligible_today` 字段 |
| `ml_core/pipeline.py` / `ml_core/rolling.py` | 训练池 = `eligible_today & can_buy & has_label`；预测池 = `eligible_today`；ExcessReturn demean 池 = `eligible_today & can_buy` |
| `ml_core/run.py` | `run_meta.json` 增加 `sample_pool` 溯源字段 |

效果：训练候选样本 13,534,573 → **13,485,408**（剔除 ~4.9 万个垃圾样本）；
预测池不再依赖 T+1 状态，停牌日直接 NaN 分（已验证 000656.XSHE 07-01 NaN、复牌后正常出分）。

## 3. A/B 实验设置

| 项 | A（旧） | B（新） |
|---|---|---|
| run_id | `lgbm_a158_p27_shap_dqlabels` | `lgbm_a158_p27_shap_dq_pool2` |
| 差异 | 仅样本池口径 | 仅样本池口径 |
| 其余 | LGBM seed=42 / shap top-64 / 切分 / 超参 / 数据源全部相同 | 同左 |

## 4. 结果

### 4.1 test IC（2020-01 ~ 2026-07，1550 天）—— 持平

| 指标 | 旧 | 新 |
|---|---|---|
| IC 均值 | +0.1210 | +0.1208 |
| ICIR | +1.197 | +1.206 |

分年 IC 差异全部在 ±0.0007 以内；入选因子 top-64 重合 60/64（换血的 4 个均为重要性 ~0.004 的尾部因子）；
每日 top500 重合度均值 86.8%，截面秩相关均值 0.982。

### 4.2 topk100 回测（netting / vwap_am / shift1 / interval5）—— 显著变强

| 指标 | 旧 | 新 | 变化 |
|---|---|---|---|
| 累计收益 | 336.55% | **414.57%** | +78.0pp |
| 年化收益 | 26.67% | **30.05%** | +3.38pp |
| 年化超额 | 24.58% | **28.25%** | +3.67pp |
| 夏普 | 0.993 | **1.124** | +0.131 |
| 信息比率 | 1.244 | **1.460** | +0.215 |
| 最大回撤 | 37.57% | 37.48% | 持平 |
| 换手率 | 34.75 | 34.62 | 持平 |

分年收益差（新-旧）：2020 +3.82pp / 2021 +0.51 / 2022 +3.36 / 2023 +4.04 / 2024 +0.30 / 2025 +9.85 / 2026H1 +0.11。
**7/7 年份不输，无一年退化。** 日收益差均值 +1.04bp/日，t=+2.15（1571 天），统计显著。

## 5. 关键洞察：为什么 IC 持平但组合明显变强

1. **IC 看全截面，组合只消费排序头部。** rank-IC 对 ~5000 只股票整体排序敏感，
   topk100 组合只取最顶端。垃圾样本（停牌股陈旧因子、极端 VMA/VSTD）最容易被模型打出
   **极端分数挤进头部**——污染的正是组合利润所在的那一小截，对全截面 IC 几乎无感。
2. **预测池换成 eligible_today 后，top500 信号不再混入 T 日停牌/ST 股**，
   回测候选池 300 补位配额不再被无效票占用，头部排序更干净。
3. 回撤/换手不变 → 提升不是靠加风险，是纯粹的选股质量改善。

**保留意见**：单次重训 A/B，尾部 4 因子换血带来一定重训噪声（2025 +9.85pp 中可能含此成分）；
但方向 7/7 一致 + 日频 t=2.15，结论站得住。

## 6. 产物路径

- 模型/预测：`/nfs/ofs-prediction/peterzhenglinpeng/ml/{models,predictions}/lgbm_a158_p27_shap_dq_pool2/`
- 回测：`daily-realtime-backtest-pipeline/results/lgbm_a158_p27_shap_{dqlabels,dq_pool2}_20200103_20260630_topk100_netting_vwapam_shift1_interval5/`
- 训练日志：`ml_core/logs/lgbm_a158_p27_shap_dq_pool2_20260707_124310.log`

## 7. 后续

- 滚动线（`ml_core/rolling.py`）已同步池口径，下次滚动重训自动生效
- 多 seed 集成（`run_seeds.py` / `ensemble_seeds.py`）默认引用 train_config，重跑即用新池
- 待办：V2 回归级验证（`regression_baseline_replication.py` + `regression_compare.py` 确认因子评估零漂移）、V4 增量日更演练
