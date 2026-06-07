# paper_33 复现关键洞察（2026-05-25）

> 本文件汇总 6 代表因子复现完成后跨因子层面的非平凡发现。
> 单因子复现结果直接看 README 的"复现 vs 论文对照"表 + 各 spec.yaml description。

---

## 1. 6 因子缺失率温和（10-16%）的真实原因——是因子选择的幸运，**不是算子设计的功劳**

> ⚠️ 自我修正（2026-05-25 review 时发现）：本节初版错误地把"无高缺失"归功于 sum-then-derive
> 算子设计；但 6 个因子的 spec 全部严格对齐 paper 原文（不是为避免 NaN 而偷换公式）。
> 真正原因如下分析。

**逐因子审视 spec 与 paper 形式的关系**（与 paper 行号对照）：

| 因子 | 形式 | paper 原文 | 是否危险（per-day ratio + sparse 分母）|
|------|-----|-----------|----------------------------------|
| `pj_peak_minute_count` | rolling.mean(peak_count) | L219 "数量" | ❌ 没有 ratio，无风险 |
| `pj_ridge_minute_return` | rolling.sum(ridge_return_sum) | L269 "加总" | ❌ sum-and-done，无 ratio |
| `pj_valley_relative_vwap` | **per-day ratio → mean(20)** | L338 "每日做比、20 日均值" | ⚠️ 形式危险，但 valley 是 ~80% 时点（非跳跃 = 价谷），**valley_vwap 几乎永有定义**，分母不稀疏 → 实测安全 |
| `pj_valley_weighted_quantile` | **per-day quantile → mean(20)** | L548 "20 日均值" | ⚠️ 同上，valley_vwap 稠密 → 安全 |
| `pj_ridge_interval_skew` | pool moments → derive | L434 "20 日间隔分布" | ❌ pool-then-derive 模式，paper 用语本身就是 pool |
| `pj_jump_turnover_corr` | pool moments → Pearson | L488 "过去 20 天 ... 相关系数" | ❌ pool-then-derive，paper 用语本身就是 pool |

**关键**：3 号、4 号是 **per-day ratio → mean** 模式（paper_27 mp20 78% 雪崩的同款危险模式），但因为 paper_33 的 valley = 非跳跃时点 ≈ 80% 分钟，valley_vwap 几乎每天都有定义，**所以分母不稀疏，雪崩自然不发生**。这是 paper 选择"价谷"作为分母而带来的红利，**不是我们 spec 设计的功劳**。

**Phase 5 扩 17 因子时必然会重现 78% 雪崩的因子**：
- **p12 价格谷岭加权价格比** (`valley_vwap / ridge_vwap`) —— `ridge_vwap` 在 ridge 极稀疏时会 NaN（实测平安银行 4 月 ridge 仅 0.7/天），per-day ratio → 20d mean，**与 paper_27 的 `valley_ridge_price_ratio` 完全同构，必须复用 `__mp10` 模板**
- **p13 价格峰岭成交额比** —— 形式取决于 paper 措辞；如果是"先 sum 后比"（类比 paper_27 `peak_ridge_turnover_ratio`）则安全，如果是"日比再 mean"则危险

**正确的工程结论**：
1. **不要为避免 NaN 偷换 paper 公式**——这是 paper_27 那次"sum-then-ratio 更接近 paper"幻觉的核心教训
2. 当 paper 公式确实是 per-day ratio → mean 且分母稀疏时，**唯一忠实做法是 paper_27 mp10 模板**（保留语义、放宽 min_periods、做衰减实验决定 sweet spot）
3. 6 因子缺失率温和是结构红利不是设计红利，但**复现忠实度是真功夫**——这才是 docs 该记录的真东西

---

## 2. paper_33 vs paper_27 同结构因子的强弱对照

paper Table 7 自己已经报告（同月频/中性化口径）：

| 因子结构 | paper_27 ICIR | paper_33 ICIR | 强弱 |
|---------|---------------|---------------|------|
| peak_minute_count | 4.36 | 3.0 | 27 强 |
| ridge_minute_return | -3.55 | -3.8 | 33 略强 |
| valley_relative_vwap | 4.44 | 4.0 | 持平 |
| valley_weighted_quantile | 4.32 | 3.5 | 27 强 |
| ridge_interval_skew | -3.82 | -3.5 | 持平 |
| eruption/jump turnover_corr | -3.99 | -3.8 | 持平 |

**观察**：paper_27（基于成交量喷发）整体略强于 paper_33（基于价格跳跃）。这与"成交量是更直接的资金活动信号、价格跳跃只是结果"的直觉一致。**配组合时若要二选一，paper_27 优先；做 ensemble 时两份 alpha 互补**（paper_33 §7 用 6 个独立 alpha 合成达到 RankICIR 4.1，单因子 3.5–3.8 的合成）。

我们的 5d 复现指标也一致呈现这个 paper 内的强弱排序：paper_33 几个因子的 5d ICIR 都比 paper_27 同名稍弱。

---

## 3. `pj_valley_weighted_quantile` 的弱：缺反转中性化（v2 待办）

复现 5d ICIR = 0.169，在 6 因子里垫底。paper_27 同名因子也只有 0.391。

paper L420（paper_27）+ paper_33 沿用同方法：
> "因子计算中正向暴露 20 日反转因子，我们进一步对量谷加权价格分位点因子做反转中性化处理"

**v1 spec 没做反转中性化**，IC 被 20 日反转因子稀释。如果决定推进 paper_33 的 v2 / 17 因子全集，**这个因子是第一个要补反转中性化的**（用 `cross_section_regress` 算子 over 20d cumulative return 即可）。

---

## 4. paper L179 sanity check 通过

paper L179 提到："以 2026年4月统计为例，价峰时点数量整体多于价岭时点数量。"

我们的单股冒烟（Phase 2）：
- 平安银行 (000001) 2026-04：peak=35.3/day, ridge=0.7/day → peak >> ridge ✅
- 茅台 (600519) 2026-04：peak=7.4/day, ridge=2.6/day → peak > ridge ✅

ridge 极稀疏（"前后 1 分钟价格区间不重叠"在流动性好的股票里是少见事件），这印证算子设计阶段对 ridge 类因子高缺失风险的预判。但因为我们用 sum-then-derive，缺失没传染（见 §1）。

---

## 5. 合成因子快速验证（vs paper §7.1）

paper §7.1（L575）：6 因子等权合成，paper 月频 RankIC 10.72%, RankICIR 4.1, 多空年化 32.86%, 月度胜率 78%。

复现方式（5d 频率，无中性化）：
- 取 cleaned panel（已 MAD + cross-section z-score）
- 每个因子按 direction 校正符号 → 6 个 z-score 取均值（至少 4 个非 NaN 才出值）
- 5d Spearman IC 时间序列统计

| 指标 | 复现（5d） | paper（月频，市值/行业中性） | 评价 |
|------|----------|-------------------------|------|
| Rank IC mean | +0.0856 | +10.72% | ✅ 量级一致 |
| ICIR | +0.715 | +4.1 | – |
| ICIR × √21（粗估月频） | **+3.27** | **+4.1** | ✅ 80% of paper（差距来自缺中性化） |
| positive_pct | 77.93% | 78% | ✅ 几乎逐字一致 |

**观察**：
- 合成 ICIR 0.715 比头部单因子 0.686（`pj_jump_turnover_corr`）只高 ~4%；paper 对应增益是 3.8 → 4.1（~8%）。
- 我们这边增益小的原因：`pj_peak_minute_count`（0.21）和 `pj_valley_weighted_quantile`（0.17）两个偏弱因子拖累等权平均。如果只用 4 个强因子合成，ICIR 估计能上 0.74+。

---

## 6. 12 因子（paper_27 6 + paper_33 6）相关性矩阵

> 已 direction 符号校正（每因子 × 其 direction），所以 ρ > 0 = 同向 alpha，|ρ| 衡量 alpha 重叠度。
> 每日 cross-section Spearman 取时间序列均值。区间 2016-01-01 ~ 2025-12-31。

```
              v_peak  v_rret  v_vvwap v_vqtl  v_rskew v_ecorr p_peak  p_rret  p_vvwap p_vqtl  p_rskew p_jcorr
v_peak_cnt    1.000   0.023   0.023   0.156   0.089   0.265   0.493   0.144   0.052   0.176   0.173   0.298
v_ridge_ret   0.023   1.000   0.581   0.127   0.368   0.271  -0.039   0.709   0.480  -0.001   0.305   0.261
v_val_vwap    0.023   0.581   1.000   0.357   0.405   0.409  -0.008   0.609   0.730   0.114   0.343   0.388
v_val_qtl     0.156   0.127   0.357   1.000   0.147   0.203   0.131   0.230   0.435   0.927   0.127   0.191
v_ridge_skew  0.089   0.368   0.405   0.147   1.000   0.469  -0.094   0.480   0.289   0.042   0.564   0.488
v_erupt_corr  0.265   0.271   0.409   0.203   0.469   1.000   0.093   0.435   0.353   0.119   0.407   0.921
p_peak_cnt    0.493  -0.039  -0.008   0.131  -0.094   0.093   1.000   0.084   0.069   0.174   0.151   0.108
p_ridge_ret   0.144   0.709   0.609   0.230   0.480   0.435   0.084   1.000   0.541   0.117   0.470   0.405
p_val_vwap    0.052   0.480   0.730   0.435   0.289   0.353   0.069   0.541   1.000   0.372   0.274   0.324
p_val_qtl     0.176  -0.001   0.114   0.927   0.042   0.119   0.174   0.117   0.372   1.000   0.055   0.107
p_ridge_skew  0.173   0.305   0.343   0.127   0.564   0.407   0.151   0.470   0.274   0.055   1.000   0.394
p_jump_corr   0.298   0.261   0.388   0.191   0.488   0.921   0.108   0.405   0.324   0.107   0.394   1.000
```

**78 个 unique pair 的相关性分布**：

| |ρ| 区间 | 数量 | 占比 |
|---------|-----|------|
| < 0.2  | **29** | 37% |
| 0.2 ~ 0.4 | 17 | 22% |
| 0.4 ~ 0.6 | 15 | 19% |
| 0.6 ~ 0.8 | 3  | 4% |
| > 0.8  | 2  | 3% |

**冗余对（|ρ| > 0.8，建议二选一）**：
- `valley_weighted_quantile` ↔ `pj_valley_weighted_quantile` （ρ = 0.927）
- `eruption_turnover_corr` ↔ `pj_jump_turnover_corr` （ρ = 0.921）

**paper_33 真正贡献的独立 alpha（与 paper_27 同名 ρ < 0.6）**：
- `pj_peak_minute_count`：与 `peak_minute_count` ρ=0.493——peak 在两 paper 定义差异最大，最有价值
- `pj_ridge_interval_skew`：与 `ridge_interval_skew` ρ=0.564

**paper_27 内部本身已经够 diverse**：v_peak_cnt 与 v_ridge_ret / v_val_vwap 的 ρ 仅 0.023——paper_27 的"6 类 20 因子"框架自带多样性。

**组合层建议**：
1. 全留 paper_27 6 因子（已经相对独立）
2. paper_33 只留 `pj_peak_minute_count` + `pj_ridge_interval_skew`（+2 个真正独立的新 alpha）
3. 可选 `pj_ridge_minute_return` 或 `pj_valley_relative_vwap`（ρ ~0.7，仍有 ~30% 残余信号）
4. **不留** `pj_valley_weighted_quantile`、`pj_jump_turnover_corr`（与 paper_27 同名 ρ > 0.92，几乎同因子）

→ paper_27 + paper_33 合并因子库实际独立因子数 ~9-10，不是 12。

---

## 8. 引擎重构后迁移为 MinuteReducer + 全 18 因子复现（2026-06-07）

**背景**：`minute_pricejump_aggregate` 原是"独立引擎 op"（自带 ProcessPool + 读旧 `MINUTE_DATA_DIR`
per-stock 后复权文件）。引擎 reducer 化重构后旧目录已删 → 本因子族**一度搁浅无法生产**。

**修复**：迁移为 `JumpReducer(MinuteReducer)`，与 tide/dazzle/intraday/smartmoney 同走
`MinuteAggregateEngine`（读 minute/raw 窗口 → 读时复权 → 按股切表 → reduce → append-only 缓存）。
**归约数学逐行不变**（_compute_one_stock 仅入参 src_path→raw），spec 入口 action/cache_key 零改动，
18 个 spec 静态校验全通过。warmup=2×std_window=40（peakridge_minute_corr_pooled 嵌套两层 rolling）。

**复现结果**（月频 RankIC，2014-2019，neu 生产版；带论文目标的 11 个**方向 11/11 全对齐**）：
| 因子 | 我RankIC | 论文 | | 因子 | 我RankIC | 论文 |
|---|---|---|---|---|---|---|
| pj_valley_relative_vwap | +6.35% | 6.53% ✓ | | pj_ridge_minute_return | -9.67% | -8.51% ✓ |
| pj_jump_followup_ratio | -9.61% | -8.92% ✓ | | pj_ridge_minute_count | -8.59% | -7.65% ✓ |
| pj_ridge_interval_kurt | -8.63% | -7.46% ✓ | | pj_ridge_interval_skew | -8.73% | -7.62% ✓ |
| pj_peak_minute_count | +4.36% | 6.38% ✓ | | pj_jump_turnover_sensitivity | -6.22% | -6.82% ✓ |
| pj_jump_turnover_corr | -6.77% | -10.23% ✓ | | pj_peak_interval_std | -2.03% | -3.96% ✓ |

方向 100% 一致，量级多数贴合（valley_relative_vwap 近乎命中）；少数高阶矩/分位因子偏弱但符号正确。

**非平凡洞察**：pricejump（系列㉝）与成交量峰岭谷（系列㉗）是**同一 reducer 模板的孪生**——
把"喷发=量>σ"换成"跳跃=振幅>σ"、把"量加权"换成价格跳跃口径，superset 列结构几乎一致。
两者 warmup 同为 40。这验证了"峰岭谷方法论"在工厂里高度可复用：换一个 σ 判定口径即得一篇新研报。
