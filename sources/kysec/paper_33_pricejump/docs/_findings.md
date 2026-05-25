# paper_33 复现关键洞察（2026-05-25）

> 本文件汇总 6 代表因子复现完成后跨因子层面的非平凡发现。
> 单因子复现结果直接看 README 的"复现 vs 论文对照"表 + 各 spec.yaml description。

---

## 1. 算子设计的红利：没有 paper_27 mp20 那种 78% 高缺失陷阱

paper_27 当时三个因子（`ridge_relative_vwap` 等）miss_listed 高达 78%，根因是 `先日 ratio → 20d mean`：
- ridge_vwap 在 0-ridge 日是 NaN
- rolling.mean 默认 min_periods=20 → 任一天 NaN 整窗 NaN
- 后来用 `__mp10` 变体降到 ~10%

paper_33 的 6 因子全部 24–29% raw NaN，**根因不同**：
- 没用"先日 ratio → 20d mean"模式
- 涉及 ridge 的因子（`pj_ridge_minute_return`、`pj_ridge_interval_skew`）走"先 sum moments → 算 skew"
- 涉及 jump 的因子走"先 sum 6 阶矩 → 算 Pearson"
- moments / sums 在 0-ridge 日是 0（不是 NaN），rolling.sum 不被 NaN 传染

**结论**：写新因子 spec 时优先用"sum-then-derive"模式，避开"日 ratio → mean"的 NaN 雪崩。这条已经在 AGENTS.md §4 因子变体约定里有引用（mp10 案例），将来设计同类因子时该作为 default。

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
