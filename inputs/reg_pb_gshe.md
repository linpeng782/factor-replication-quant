# 因子描述：reg_pb_gshe（经过调整的 PB 估值因子）

经过调整的 PB 估值因子。属于价值类因子。

## 计算流程

1. 在中证全指范围内，取 PB（pb_ratio_lf）、单季度 ROE（net_profit_mrq_0 / total_equity_mrq_0）、
   单季度 PE（market_cap_3 / net_profit_mrq_0），计算 EP = 1/PE = net_profit_mrq_0 / market_cap_3
2. 剔除 PB ≤ 0 和 ROE ≤ 0 的个股
3. 对每只股票，求 EP 过去一年（252 个交易日）的中位数
4. 每个交易日截面内，对 EP 中位数做 MAD 去极值，然后分成 10 组
5. 每个交易日截面内，对 ROE 和 log(PB) 分别做 MAD 去极值
6. 在每个 (交易日, EP 分组) 内，做截面 OLS 回归：log(PB) = β₀ + β₁·ROE + ε
7. 残差 ε 即为因子值 reg_pb_gshe

## 因子方向
direction = -1（反向）：残差为正表示实际 PB 高于 ROE 所能解释的水平（高估），预期未来收益低。

## 基准
PROGRESS.md 记录基准 IC=0.025375，ICIR=0.530611。
