# roic_ttm_ind_rnk8 — 行业内 ROIC 排名 8 期最小值

> **状态**：v1 全市场跑通，**失效**（ICIR < 0.07，单调性负）
> **复现 IC/ICIR**：5d 0.006/+0.044, 10d 0.009/+0.066, 20d 0.008/+0.059
> **LongShort Sharpe** -0.30，**单调性** -0.37
> **基准**：PROGRESS.md 中无 roic_ttm 系列基准

---

## 1. 因子定义

```
roic_ttm_ind_rnk8 = min(过去 8 期 ROIC_TTM 行业内截面排名)
```

- 每个交易日按中信一级行业分组，对 ROIC_TTM 从高到低排名（`ascending=False, method='min'`）
- 在 ROIC_TTM 变化日上 `rolling(8, min_periods=4).min()`，然后 ffill
- **方向 -1**：因子值越**小** = 历史上行业内曾达到过越靠前的排名 → "曾经辉煌"

经济直觉：龙头基因。即使当前业绩下滑，曾达到过行业前列的公司，恢复潜力比从未优秀过的更大。

---

## 2. 计算链路（5 步）

```yaml
1. fetch:  return_on_invested_capital_ttm
2. fetch:  __internal__zx2019_industry  →  industry 表（自动日频化）
3. merge:  industry 的 first_industry_name 列 → data
4. rank:   group_by=[date, first_industry_name], ascending=False,
           rank_method='min', pct=False  →  roic_ttm_ind_rank
5. rolling: agg=min, window=8, min_periods=4,
           change_on=return_on_invested_capital_ttm,
           fill_method=ffill  →  roic_ttm_ind_rnk8
```

---

## 3. 复现结果

```
覆盖：(2430, 5549)，非空 8.65M / 13.5M = 64%

IC (Spearman):
  5d:  ic_mean=+0.006,  icir=+0.044
  10d: ic_mean=+0.009,  icir=+0.066
  20d: ic_mean=+0.008,  icir=+0.059

分层 (5 组 × 5 日):
  G1: ann=+3.98%   G2: +4.41%   G3: +2.82%   G4: +4.67%   G5: +2.83%
  LongShort: ann=-2.51%, Sharpe=-0.30
  monotonicity: -0.37
```

**所有有效性指标都不及格**：ICIR < 0.07、单调性负、LongShort Sharpe 负。

---

## 4. ⭐ 失效根因分析

### 4.1 行业内排名尺度不可比

银行业 ~40 只，机械 ~500 只，**同样的"排名 40"在两个行业里语义完全不同**——银行 40 名是垫底，机械 40 名是中上游。MAD+zscore 在这种异质分布下被极端值压扁，截面分布右偏 + 尖峰。

### 4.2 8 期 min 操作几乎失效

公司基本面季度间高度相关，过去 8 期排名通常变化在 ±5 名内。`min(40, 38, 42, 39, 41, 40, 37, 39) ≈ 37` ≈ 当前排名。**取最小没有提取出"曾经辉煌"信号，因子退化为"当前行业内排名"**。

### 4.3 方向自动翻转

spec 写 `direction=-1`（小排名值越好），但评估系统检测 raw IC 为负数，自动设 direction=-1 后 LongShort 仍为负。说明经济直觉"曾经辉煌"在 A 股 2016-2025 数据上不成立。

可能的原因：
- A 股质量因子在样本期内整体失效（小盘/概念主导）
- 评估周期偏短（5/10/20 日），质量因子可能需月度调仓
- 行业内排名信息早已 priced in，没有 alpha 残留

---

## 5. 与同组因子对比

| 因子 | ICIR | 单调性 | 评价 |
|---|---|---|---|
| **`roic_ttm_ind_rnk8`** | +0.044 | -0.37 | 行业内排名尺度+8 期 min 双重失效 |
| `roic_ttm_all_rnk8` | +0.006 | +0.89 | 噪声 |
| `roic_ttm_dev_std8` | +0.222 | +0.72 | 有效（详见 [[roic_ttm_dev_std8]]）|

---

## 6. v2 候选改进路径

| 优先级 | 方向 | 预期 |
|---|---|---|
| P1 | 用 `pct=True` 行业内分位排名（消行业尺度差） | 改善尺度但 8 期 min 失效仍在 |
| P1 | 改用"排名变化量"：当前排名 - 8 期 min 排名 | 提取真实"上升"信息 |
| P2 | 中证 800 + 总市值前 100 底池（研报原文） | 大票池子里行业排名更稳定 |
| P3 | window 扩到 20 期 | 增加排名变化机会但牺牲覆盖率 |

短期看 P1.2（排名变化量）最值得试，把"曾经辉煌"改成"正在赶超"——经济直觉更强、信号噪声比更可能正向。
