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
