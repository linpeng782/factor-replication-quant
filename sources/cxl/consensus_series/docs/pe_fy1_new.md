# pe_fy1_new — 滚动一致预期 PE（前瞻价值因子）

> 类别：价值 | 方向：−1 | 数据源：`get_consensus_comp_indicators(report_range=3)`

## 1. 因子定义
分子 = 总市值；分母 = **forward-12m 滚动一致预期净利润**。

设某交易日 D 落在日历年 Y，`report_year_t = ry`（最近已披露年报年度）。一致预期净利润
`comp_con_net_profit_t1/t2/t3` 分别对应年度 ry+1 / ry+2 / ry+3。前瞻 12 个月窗口
[D, D+1yr] 跨 Y 与 Y+1，按当年剩余天数线性插值：

```
w = (Dec31(Y) − D).days / 365        # 落在年 Y 的比例
forward_np = w·NP(Y) + (1−w)·NP(Y+1)
NP(Y)   = t{Y−ry}      NP(Y+1) = t{Y−ry+1}   （offset 越界置 NaN）
pe_fy1_new = 总市值 / forward_np      （要求 forward_np > 0）
```

研报原文用"距去年 12/31 天数 / 365 作 fy1 权重，余量给 fy2"。本实现等价于标准
forward-12m 滚动（economically = 前瞻 12 月盈利收益率倒数），消解了原文 fy1/fy2 标签的
方向歧义；用 vendor 口径的 report_year_t 自动处理 4 月年报披露时的预测期切换。

## 2. 关键工程决策
- **`report_range=3`（不考虑补录入）**：保证 PIT——某日的一致预期只含当日前已入库的研报，
  历史值不被后续补录修正。`report_range=0` 虽带现成 `comp_con_net_profit_ftm`，但含补录入
  → 有前视风险，弃用，改自行插值（恰好就是研报定义的滚动逻辑）。
- 市值复用预计算 `market_cap_panel`（单位**亿元**，×1e8 → 元）与净利润（元）对齐。
- 覆盖：comp 数据仅含有分析师覆盖的股票（5168/5551），面板天然比财务因子稀疏，符合预期。

## 3. 复现结果（20160101–20251231, 2430 天, n=5×g=5, 全市场）
| 版本 | rankIC 5/10/20d | ICIR 5/10/20d | 单调性 | LongShort 年化/Sharpe | 方向 |
|---|---|---|---|---|---|
| neu(生产) | 0.027/0.032/0.037 | 0.32/0.35/0.39 | +0.955 | 10.5% / 1.33 | −1 |
| cleaned | 0.029/—/— | — | +0.955 | 7.6% / 0.51 | −1 |

## 4. 非平凡洞察 ⭐
**前瞻一致预期 PE ≈ 历史 PE 的 2 倍信息量。** 同口径全市场下 `pe_ttm_new`（trailing PE）
rankIC≈0.015 / ICIR≈0.15，而 `pe_fy1_new`（forward consensus PE）rankIC≈0.027 / ICIR≈0.32。
信息量翻倍但量级仍落在价值因子常态区间（未出现 IC>0.1 的前视污染特征）——这是 `report_range=3`
PIT 口径正确的旁证：若误用 `report_range=0` 的补录入 ftm，IC 大概率被前视抬高到不合理水平。

**经济直觉**：分析师对未来 12 个月盈利的预期，比已实现的 TTM 盈利更贴近"市场为之定价的
盈利锚"，故 forward E/P 的截面区分度更强；neu 版 LongShort Sharpe 1.33、单调性 0.955
说明排序信息干净。

## 5. 引用
- 计算：`scripts/consensus_factors.py::build_pe_fy1_new` / `compute_forward_np`
- 数据：`data_fetching/consensus.py::load_or_fetch_comp_indicators`
- 评估图：`output/cxl/consensus_series/pe_fy1_new/evaluation_20160101_20251231__{cleaned,neu}.png`
