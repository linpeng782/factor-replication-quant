# 因子描述：reg_pe_hist（经过历史增速变化调整的历史 PE 变化）

经过历史增速变化调整的历史 PE 变化。属于价值类因子。

## 原始变量
1. 单季度净利润同比 = `log(net_profit_mrq_0 / net_profit_mrq_4)`
   - 取 log 比率形式（避免负值问题，保留对称性）
   - 要求 `net_profit_mrq_0 > 0` 和 `net_profit_mrq_4 > 0`
2. 净利润 TTM = `log(net_profit_ttm)`，要求 `net_profit_ttm > 0`
3. PE_TTM = `log(pe_ratio_ttm)`，要求 `pe_ratio_ttm > 0`

## 计算流程
对三个 log 变量分别求 60 个交易日的差分（diff）：
- `delta_log_yoy = Δ60(log_yoy)`
- `delta_log_npttm = Δ60(log_npttm)`
- `delta_log_pe = Δ60(log_pe)`

然后做嵌套截面回归：
1. **第一步**：以 `delta_log_yoy` 为 y、`delta_log_npttm` 为 x，按交易日做截面回归，
   取残差 `residual_1`（"单季同比变化中独立于 TTM 变化的部分"）
2. **第二步**：以 `delta_log_pe` 为 y、`residual_1` 为 x，按交易日做截面回归，
   取残差 `reg_pe_hist`（"PE 变化中无法被独立利润动量解释的部分 = 估值情绪"）

## 因子方向
direction = -1（反向）：PE 异常上涨（无利润支撑）→ 估值高估 → 预期未来收益低。

## 基准
PROGRESS.md 记录基准 IC=0.063，ICIR=0.85。
