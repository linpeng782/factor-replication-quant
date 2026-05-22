# roic_ttm_all_rnk8 — 全市场 ROIC 排名 8 期最小值

> **状态**：v1 全市场跑通，**噪声**（ICIR < 0.04，单调性貌似好但 IC 无信号）
> **复现 IC/ICIR**：5d 0.001/+0.006, 10d 0.002/+0.019, 20d 0.004/+0.032
> **LongShort Sharpe** +0.36，**单调性** +0.89（误导）
> **基准**：PROGRESS.md 中无 roic_ttm 系列基准

---

## 1. 因子定义

```
roic_ttm_all_rnk8 = min(过去 8 期 ROIC_TTM 全市场截面排名)
```

与 [[roic_ttm_ind_rnk8]] 唯一差别：**排名分组不带行业，仅按交易日截面**。

- `ascending=False, rank_method='min', pct=False` 输出整数排名 1~5500
- 在 ROIC_TTM 变化日上 `rolling(8, min_periods=4).min()`，然后 ffill
- **方向 -1**（spec 写）：但评估自动翻成 +1（详见 §4）

---

## 2. 计算链路（3 步，比 ind_rnk8 少 merge+industry）

```yaml
1. fetch:  return_on_invested_capital_ttm
2. rank:   group_by=[date], ascending=False, rank_method='min', pct=False
           →  roic_ttm_all_rank
3. rolling: agg=min, window=8, min_periods=4,
           change_on=return_on_invested_capital_ttm,
           fill_method=ffill  →  roic_ttm_all_rnk8
```

---

## 3. 复现结果

```
覆盖：(2430, 5549)，非空 8.81M / 13.5M = 65%

IC (Spearman):
  5d:  ic_mean=+0.001,  icir=+0.006   ← 几乎为 0
  10d: ic_mean=+0.002,  icir=+0.019
  20d: ic_mean=+0.004,  icir=+0.032

分层 (5 组 × 5 日):
  G1: +1.72%   G2: +3.64%   G3: +3.78%   G4: +3.67%   G5: +5.12%
  LongShort: ann=+3.51%, Sharpe=+0.36
  monotonicity: +0.891   ← 看着漂亮但具有误导性
```

---

## 4. ⭐ 关键洞察：单调性 0.89 ≠ 因子有效

这是这组三因子里**最值得记录的工程教训**：

`monotonicity = Pearson(组号 1~5, 各组 ann_return)` = +0.89 看着像非常单调的因子，但 **5d ICIR = 0.006 几乎是纯噪声**。

为什么？

1. **G1~G5 跨度仅 3.4pp**（1.72% → 5.12%），整体在 +2~+5% 区间漂移
2. **截面信息几乎不可分**——5d Spearman IC 0.001 说明任意两支股票之间排序与未来收益基本无关
3. **5 组分层把噪声当成信号"塑形"**：在 5500 只股票上分 5 组，每组 1100 只，组内平均把噪声平均掉，**残留的微弱漂移就被分层放大成假单调性**

更要命的：spec 写 `direction=-1`（小排名值越好），但评估检测 raw IC 为正自动翻成 +1。**这意味着研报的经济直觉在 A 股全市场尺度上反过来了**——"从未辉煌"反而比"曾经辉煌"更有微弱预测力。但这个微弱信号 ICIR 0.006 不足以采用。

**结论**：用 ICIR 而不是单调性判定有效性。单调性 +0.89 + ICIR < 0.05 = 几乎肯定是分层伪信号。

---

## 5. 与同组因子对比

| 因子 | ICIR | 单调性 | LongShort Sharpe | 评价 |
|---|---|---|---|---|
| `roic_ttm_ind_rnk8` | +0.044 | -0.37 | -0.30 | 失效 |
| **`roic_ttm_all_rnk8`** | **+0.006** | +0.89 | +0.36 | **噪声**，单调性是分层伪信号 |
| `roic_ttm_dev_std8` | +0.222 | +0.72 | +0.76 | 有效（详见 [[roic_ttm_dev_std8]]）|

`ind_rnk8` 和 `all_rnk8` 失效原因高度同源：8 期 min 操作在公司基本面季度间高相关的前提下，几乎不改变当前排名分布。`min(40, 38, 42, 39, 41) ≈ 37` ≈ 当前排名 40。详见 [[roic_ttm_ind_rnk8]] §4。

---

## 6. v2 候选改进路径

| 优先级 | 方向 | 预期 |
|---|---|---|
| P1 | 改用"排名变化量"：当前排名 - 8 期 min | 提取"赶超"信号 |
| P1 | 中证 800 + 总市值前 100 底池（研报原文） | 大票池子里 ROIC 变化更有意义 |
| P2 | window 扩到 20 期 | 增加排名变化幅度 |
| P3 | 直接用 ROIC_TTM 截面分位 | 跳过 8 期 min，回归基础 quality 因子 |

短期看 P1.1 + P1.2 应该联动做：在大票池子里取"赶超"信号，避开小盘噪声 + 排名滞涨双重失效。
