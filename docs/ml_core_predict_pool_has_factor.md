# ml_core 推理预测池：can_buy vs can_buy & has_factor（LGBM）

> 记录一次关于「LGBM 推理时预测池该不该卡 has_factor」的调研与结论。
> 模型：`lgbm_a158_p27_top64`（alpha158-dquant + kysec-dquant/paper_27_microstructure，SHAP 选 top-64，horizon=20）。
> 复现脚本（只读）：`_tmp_predict_pool_check.py`（覆盖量化）、`_tmp_missing_trace.py`（缺失溯源）。

## 1. 问题

`ml_core.pipeline.predict_live` 里预测池 `base = u.can_buy`，只卡 T+1 可买（非 ST/停牌/新股），
**没有**像旧华泰 ml_ht 项目那样卡 `can_buy & has_factor`（要求全部入选因子非 NaN）。
两个问题：(1) 这样合理吗？(2) 缺因子的行 LGBM 还能出预测吗？

## 2. 关键代码事实：has_factor 没丢，是按模型策略分层施加

`base = u.can_buy` 只是**候选池**；`has_factor` 挪到了 `features.build_feature_matrix` 里按
`HasFactorPolicy` 分模型施加：

| 模型 | 策略 | 效果 | 有效预测池 |
|---|---|---|---|
| LGBM | `NONE` | 不卡完整性，NaN 原生喂模型 | `can_buy` |
| MLP  | `ALL`  | 任一因子 NaN 即剔行（稠密输入缺一不可） | `can_buy & has_factor` |

**所以 MLP 已经是 `can_buy & has_factor`；LGBM 是故意只用 `can_buy`。** 旧 ml_ht 用后者，
是因为那是全连接网络（MLP），口径其实一致。

## 3. LGBM 能对缺因子的行出预测吗？——能，一个都不落

LightGBM 原生支持缺失值：训练时每个分裂节点学一个「默认方向」（NaN 往哪边走，由损失决定）。
推理时一行不管缺几个因子（1 个 / 11 个 / 全缺），遇到 NaN 就走默认方向，必落到某个叶子 → 出预测分。
- 缺 1-2 个：几乎不影响，绝大多数分裂用有值的因子。
- 缺 6+ 个：仍出值，但越来越多分裂靠默认方向兜底，预测变钝（信息量下降，非报错）。

这就是 `predict_live` 用 NONE 能对整个 can_buy 池出满面板的底层保证。

## 4. 量化：两口径实质等价（推理区间累计）

`_tmp_predict_pool_check.py`，区间 2022-01-04~2026-06-26（1083 交易日）：

| 口径 | 累计样本 |
|---|---|
| `can_buy` | 5,021,665 |
| `can_buy & has_factor`（64 因子全非 NaN） | 4,978,920 |
| **改口径将丢弃** | **42,745（仅占 0.85%）** |

- 每日覆盖：can_buy 中位 4754 只/日；can_buy&has_factor 中位 4738 只/日 → **每天只差 ~16 只**。
- 每日留存比中位 99.23%，最差 96.82%；近端无骤降（尾部因子面板不陈旧）。
- can_buy 样本「缺几个因子」分布：**缺 0（全有）占 99.15%**；缺 1-2 占 0.76%；缺 3-5 占 0.08%；
  缺 6-10 占 0.01%；缺 11+ ≈ 0%。中位=0、均值=0.01、95 分位=0。

即被 has_factor 剔掉的 0.85% 里，近 90% 只缺 1-2 个因子——恰是 LGBM 处理 NaN 最可靠的情形。

## 5. 溯源：因子缺失是谁、缺哪些、什么场景

`_tmp_missing_trace.py`，区间 2020-01-02~2026-06-26（can_buy=6,749,530）。64 因子 = 52 alpha158 + 12 p27。

**逐因子 can_buy 缺失率 Top（缺失主力就几个）**：
- `valley_weighted_quantile` / `valley_relative_vwap`（p27）各 0.509%
- `valley_ridge_price_ratio__mp10`（p27）0.441%
- `CORD10`(0.052%) / `CORD5` / `VWAP0` / `CORR5` / `RSQR10` / `CORD30` / `CORD60`（alpha158 相关性/量能家族）

成因：
- **p27 `valley_*`**：需要分钟级"谷"结构，无清晰谷形/分钟数据薄时无定义 → 因子设计属性，非 bug。
- **alpha158 `CORD*`/`CORR*`/`RSQR*`/`VWAP0`**：都是价量相关/成交量派生，窗口内成交量近乎不变或为 0 时
  相关系数无定义、VWAP 除零 → 典型是低流动性/一字/近停牌日。

**高缺失桶（可忽略量级）**：
- 缺 6-10：587 行，494 只股票散布 449 天，单股最多 6 次 → **零散个股事件，非系统性**；缺的是 valley_* + 相关性家族叠加。
- 缺 11+：仅 30 行 / 27 只 / 16 天（6.5 年）。p27 占比 52%，且 12 个分钟因子几乎**全体同时缺**
  → 根因：**某可买股票当天分钟数据整体缺失/不可用，一整个分钟因子家族一起变 NaN**。

## 6. 结论与建议

1. **两口径对本 LGBM 实质等价**：alpha158+p27 量价因子全市场覆盖极高，can_buy 里 99%+ 天然 has_factor。
   与华泰 MLP 场景（158 稠密因子缺失多）不同。
2. **保持现状 `can_buy`（NONE）**：① 与训练同分布（训练即 NONE），无 train/serve skew；
   ② 白捡 ~16 只/日可买股票信号；③ 丢弃项主要缺 1-2 个，LGBM 处理稳。
3. **可选保险**（若想挡住极稀疏行）：不必全上 has_factor，设阈值「缺失 > 10 个因子才剔」即可——
   全区间只剔 ~30 行、几乎不引入 skew。当前未加，视需要再定。

## 附：清理

`_tmp_predict_pool_check.py` / `_tmp_missing_trace.py` 为一次性只读分析脚本，结论已沉淀于此，可删。
