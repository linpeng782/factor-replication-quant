# 开源证券·市场微观结构系列（33）

**研报**：高频价格跳跃的峰、岭、谷信息（2026-05-15，魏建榕等）
**共享方法论**：日内分钟**振幅** ≥ 1σ → 价格跳跃；前后 1 分钟振幅 → 局域情绪；前后 1 分钟价格区间是否重叠 → 缺口判定；峰/岭/谷取双特征并集（详见 paper.md L161-180）。

**与 paper_27 的关系**：续作，**因子框架同构**（参考 paper.md L209）；区别在于 paper_27 用"成交量喷发"，本文用"价格跳跃"——是**两套独立分类逻辑**，故落到独立算子 `minute_pricejump_aggregate`，因子前缀 `pj_`。

因子清单见 specs/ 子目录。详见 paper.md（研报原文）+ docs/<factor>.md（沉淀）。

---

## 复现范围：paper.md Table 7 全 17 个有效因子 + 1 mp10 变体

> paper.md L209："基于价格跳跃的峰、岭、谷划分，构建 11 个大类的 17 个有效因子"。
> 全部用 `pj_` 前缀避免与 paper_27 同名冲突（详见 docs/findings.md §6 12 因子相关矩阵）。

---

## 复现 vs 论文完整对照（5d 频率，2016-01-01 ~ 2025-12-31，**无中性化**）

> 来源：`factor_inventory/20260525_213929/inventory.parquet`。
> ⚠️ 与 paper（月频 10 分组 市值/行业中性）不可直接比较 ICIR 量级；日频 ICIR × √21 ≈ 月频 ICIR。
> miss_listed 是已上市域内的 NaN 比例（baseline ~10%）。

| paper# | 因子 | dir | 5d ICIR | LS Sharpe | miss_listed | paper RankICIR | 方向 |
|--------|------|-----|---------|-----------|-------------|---------------|------|
| p1  | `pj_peak_minute_count`         | +1 | 0.212 | 1.61 | 10.3% | +3.0 | ✅ |
| p2  | `pj_ridge_minute_count`        | -1 | 0.420 | 1.88 | 10.3% | -2.3 | ✅ |
| p3  | `pj_ridge_minute_return`       | -1 | **0.671** | 2.89 | 10.3% | -3.8 | ✅ |
| p4  | `pj_valley_relative_vwap`      | +1 | **0.673** | 2.47 | 11.4% | +4.0 | ✅ |
| p5  | `pj_valley_weighted_quantile`  | +1 | 0.169 | 0.45 | 11.4% | +3.5 | ✅ |
| p6  | `pj_peak_interval_std`         | -1 | 0.111 | 0.82 | 10.3% | -2.0 | ✅ |
| p7  | `pj_peak_interval_skew`        | +1 | 0.102 | 0.75 | 10.3% | +2.2 | ✅ |
| p8  | `pj_peak_interval_kurt`        | +1 | 0.101 | 0.79 | 10.3% | +2.4 | ✅ |
| p9  | `pj_ridge_interval_std`        | +1 | **0.635** | 2.98 | 15.5% | +3.0 | ✅ |
| p10 | `pj_ridge_interval_skew`       | -1 | **0.685** | 3.44 | 15.5% | -3.5 | ✅ |
| p11 | `pj_ridge_interval_kurt`       | -1 | **0.671** | 3.30 | 15.5% | -3.3 | ✅ |
| p12 | `pj_valley_ridge_price_ratio`  | +1 | 0.585 | 2.54 | **🔴 91.6%** | +2.9 | ✅ |
| p12'| `pj_valley_ridge_price_ratio__mp10` | +1 | **0.660** | 2.47 | 21.7% | – | mp10 变体 |
| p13 | `pj_peak_ridge_turnover_ratio` | +1 | 0.535 | 2.95 | 10.7% | +2.0 | ✅ |
| p14 | `pj_jump_followup_ratio`       | -1 | **0.650** | 3.28 | 10.3% | -3.5 | ✅ |
| p15 | `pj_jump_turnover_sensitivity` | -1 | **0.601** | 2.51 | 10.3% | -3.4 | ✅ |
| p16 | `pj_jump_turnover_corr`        | -1 | **0.686** | 3.24 | 10.3% | -3.8 | ✅ |
| p17 | `pj_peakridge_minute_corr`     | +1 | 0.244 | 0.60 | 10.3% | +2.1 | ✅ |

**核心观察**：

- **方向 17/17 与论文一致** ✅
- **强因子（5d ICIR ≥ 0.6，9 个）**：p3 / p4 / p9 / p10 / p11 / p12_mp10 / p14 / p15 / p16；ICIR ~0.6-0.69，约等于 paper RankICIR/5，与频率+中性化损耗预期一致
- **弱因子（ICIR < 0.3，6 个）**：p1 / p5 / p6 / p7 / p8 / p17；其中 paper Table 7 也确认 p1 / p6 / p7 / p8 是 paper_33 内最弱的几个（paper RankICIR ~2.0-2.4）
- **p12 触发预测中的 91.6% 雪崩**（paper.md L171 价岭定义比 paper_27 ridge 更稀疏 → ridge_vwap 在更多日 NaN）；mp10 变体降到 21.7% 且 **ICIR 反而提升**（0.585 → 0.660），说明放宽 min_periods 不仅救覆盖也救信号
- **`pj_valley_weighted_quantile` (0.169) 仍偏弱**：paper L420 明说要做 20d 反转中性化，v1 spec 没做（v2 待办）

---

## paper_33 vs paper_27 强弱对照（同名因子）

| 维度 | paper_27 强 | paper_33 强 | 评价 |
|------|------------|------------|------|
| Peak 类（孤立喷发 vs 价格跳跃峰） | 5d ICIR 0.30-0.38 | 5d ICIR 0.10-0.21（弱 50%） | volume-defined peak 信号更纯 |
| Ridge 类 | 5d ICIR 0.32-0.74 | 5d ICIR 0.42-0.69（持平/略弱） | 两者类比基本一致 |
| Cross-category 类 | 5d ICIR 0.59-0.82 | 5d ICIR 0.54-0.69（略弱） | paper_27 的 turnover_ratio 0.82 是头号 |

**结论**：paper_27 整体更强于 paper_33，与 paper Table 7 的 RankICIR 结论一致（paper_27 头部 ICIR 4.4-4.6，paper_33 头部 3.8-4.0）。但 paper_33 的 `pj_peak_minute_count` 与 paper_27 同名因子相关性仅 0.49（详见 docs/findings.md §6），**仍是真正的 incremental alpha**，建议保留。

---

## 算子设计要点

`minute_pricejump_aggregate`（与 paper_27 的 `minute_intraday_aggregate` 平行，独立 cache_key 命名空间，互不污染）：

- **振幅** = `(high - low) / close` 每分钟（paper 未明说，按标准实践，**单股冒烟测试时验证**）
- **σ 窗口** = 过去 20 日同时点 1σ（与 paper_27 平行；paper 未明说但论"过去 20 日"高度一致）
- **缺口判定** = `(t-1).high < (t+1).low` ∨ `(t+1).high < (t-1).low`（区间不重叠）
- **局域情绪** = 跳跃前后 1 分钟振幅是否 ≥ 1σ → 高高（高涨）/ 低低（低迷）/ 混合（适中）
- **价峰** = 「**非**高涨」∧ 「无缺口」 跳跃时点（即低迷 ∨ 适中 ∧ 无缺口）
- **价岭** = 「**非**低迷」∧ 「有缺口」 跳跃时点（即高涨 ∨ 适中 ∧ 有缺口）
- **价谷** = 振幅 < 1σ 的时点（非跳跃）
