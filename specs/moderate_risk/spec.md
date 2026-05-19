
---

## Spec 文档：moderate_risk

## 1. 因子基本信息

- **因子名称（英文）**：moderate_risk
- **别名**：适度冒险因子、Moderate Adventure
- **因子大类**：复合因子
- **因子标签**：量价 / 行为金融
- **合成方式**：月耀眼波动率（50%）+ 月耀眼收益率（50%）等权合成

## 2. 因子定义

通过考察个股日内成交量激增时段的价格波动与价格变动，衡量投资者对信息反应"适度程度"的因子。反应不足（成交量激增但价格几乎不动）和反应过度（成交量激增导致价格剧烈波动或大幅涨跌）均为非理性行为，因子旨在捕捉"适度反应"的股票。

## 3. 计算公式

**核心逻辑（三步走）：**

**Step A：识别成交量激增时刻**
$$
\text{volume\_increase}_t = \text{volume}_t - \text{volume}_{t-1}
$$
$$
\text{mean} = \text{mean}(\text{volume\_increase}), \quad \text{std} = \text{std}(\text{volume\_increase})
$$
$$
\text{surge\_moment} = \{ t \mid \text{volume\_increase}_t > \text{mean} + \text{std} \}
$$

**Step B：构造"耀眼5分钟"**
对于每个 surge_moment，取该分钟及随后4分钟，共5分钟：
$$
\text{dazzling\_5min}(t) = [t, t+1, t+2, t+3, t+4]
$$

**Step C-1：月耀眼波动率分支**
$$
\text{dazzling\_vol}_i = \text{std}(\text{minute\_returns} \in \text{dazzling\_5min}_i)
$$
$$
\text{daily\_dazzling\_vol} = \text{mean}(\text{dazzling\_vol}_i)
$$
$$
\text{moderate\_daily\_vol} = |\text{daily\_dazzling\_vol} - \text{cross\_sectional\_mean}|
$$
$$
\text{monthly\_mean\_vol} = \text{mean}_{20D}(\text{moderate\_daily\_vol})
$$
$$
\text{monthly\_std\_vol} = \text{std}_{20D}(\text{moderate\_daily\_vol})
$$
$$
\text{monthly\_dazzling\_vol} = \frac{\text{monthly\_mean\_vol} + \text{monthly\_std\_vol}}{2}
$$

**Step C-2：月耀眼收益率分支**
$$
\text{dazzling\_ret}_i = \text{minute\_return}(\text{surge\_moment}_i)
$$
$$
\text{daily\_dazzling\_ret} = \text{mean}(\text{dazzling\_ret}_i)
$$
$$
\text{moderate\_daily\_ret} = |\text{daily\_dazzling\_ret} - \text{cross\_sectional\_mean}|
$$
$$
\text{monthly\_mean\_ret} = \text{mean}_{20D}(\text{moderate\_daily\_ret})
$$
$$
\text{monthly\_std\_ret} = \text{std}_{20D}(\text{moderate\_daily\_ret})
$$
$$
\text{monthly\_dazzling\_ret} = \frac{\text{monthly\_mean\_ret} + \text{monthly\_std\_ret}}{2}
$$

**Step D：最终合成**
$$
\text{moderate\_risk} = \frac{\text{monthly\_dazzling\_vol} + \text{monthly\_dazzling\_ret}}{2}
$$

## 4. 数据对齐（★ 最重要 ★）

### 4.1 数据源与字段

| 变量 | 米筐字段 | 表/接口 | 频率 | 数据类型 | 关键备注 |
|------|---------|--------|------|---------|---------|
| 分钟收盘价 | `close` | `get_price` | 1分钟 | 时点值 | 日内分钟行情 |
| 分钟成交量 | `volume` | `get_price` | 1分钟 | 累计/增量 | 需计算增量 |
| 市值 | `market_cap` | `get_factor` | 日频 | 时点值 | 正交化用 |
| 行业分类 | `citics_industry` | `get_factor` | 日频 | 类别 | 正交化用 |

### 4.2 数据处理要点

- **剔除开盘和收盘**：只保留日内连续竞价时段（9:30-11:30, 13:00-14:56），剔除开盘前集合竞价和收盘集合竞价
- **成交量增量**：每分钟成交量 = 该分钟累计成交量 - 上一分钟累计成交量
- **激增时刻判定**：日内所有分钟的成交量增量 > 日内均值 + 1倍标准差
- **耀眼5分钟**：激增时刻及其后4分钟，共5分钟
- **20日滚动**：最近20个交易日（非自然日）

### 4.3 正交化处理

测试时需对因子进行**市值和行业正交化**：
1. 用市值对因子回归，取残差
2. 用行业虚拟变量对残差回归，取残差

## 5. 计算步骤（按执行顺序）

**Step 1：获取分钟行情数据**
- 输入：股票池、日期范围
- 操作：调用 `get_price(frequency='1m', fields=['close', 'volume'])`
- 输出：`minute_data`（分钟收盘价、分钟成交量）

**Step 2：剔除开盘收盘**
- 输入：`minute_data`
- 操作：过滤时间，只保留 09:30-11:30 和 13:00-14:56
- 输出：`minute_data_filtered`

**Step 3：计算成交量增量**
- 输入：`minute_data_filtered`
- 操作：`volume_increase = volume.diff()`
- 输出：`minute_data_with_increase`

**Step 4：识别激增时刻**
- 输入：`minute_data_with_increase`
- 操作：按日分组，计算 mean 和 std，标记 volume_increase > mean + std 的分钟
- 输出：`surge_moments`

**Step 5：构造耀眼5分钟**
- 输入：`surge_moments`
- 操作：对每个激增时刻，扩展为 [t, t+1, t+2, t+3, t+4]
- 输出：`dazzling_5min`

**Step 6：计算分钟收益率**
- 输入：`minute_data_filtered`
- 操作：`minute_return = close.pct_change()`
- 输出：`minute_returns`

**Step 7：计算日耀眼波动率**
- 输入：`dazzling_5min`, `minute_returns`
- 操作：对每个耀眼5分钟计算收益率标准差，再求日均值
- 输出：`daily_dazzling_vol`

**Step 8：计算适度日耀眼波动率**
- 输入：`daily_dazzling_vol`
- 操作：`|daily_dazzling_vol - cross_sectional_mean|`
- 输出：`moderate_daily_vol`

**Step 9：计算月耀眼波动率**
- 输入：`moderate_daily_vol`
- 操作：最近20个交易日的均值和标准差，等权合成
- 输出：`monthly_dazzling_vol`

**Step 10：计算日耀眼收益率**
- 输入：`surge_moments`, `minute_returns`
- 操作：取激增时刻的分钟收益率，求日均值
- 输出：`daily_dazzling_ret`

**Step 11：计算适度日耀眼收益率**
- 输入：`daily_dazzling_ret`
- 操作：`|daily_dazzling_ret - cross_sectional_mean|`
- 输出：`moderate_daily_ret`

**Step 12：计算月耀眼收益率**
- 输入：`moderate_daily_ret`
- 操作：最近20个交易日的均值和标准差，等权合成
- 输出：`monthly_dazzling_ret`

**Step 13：合成适度冒险因子**
- 输入：`monthly_dazzling_vol`, `monthly_dazzling_ret`
- 操作：等权合成 `(vol + ret) / 2`
- 输出：`moderate_risk`

## 6. 回测参数

| 参数 | 设置 | 来源 |
|------|------|------|
| 股票池 | 全A股（非ST、非停牌、上市满60日） | 研报基准测试 |
| 备选池 | 沪深300 / 中证500 / 中证1000 | 研报扩展测试 |
| 调仓频率 | 月频 | 研报设定 |
| 测试区间 | 2013年4月 ~ 2022年2月 | 研报区间 |
| 因子方向 | 反向（值越小越好） | Rank IC = -8.89% |
| 处理方式 | 市值和行业正交化 | 研报设定 |
| 分组数 | 10组 | 研报设定 |

## 7. 预期效果

| 指标 | 预期值 | 说明 |
|------|--------|------|
| Rank IC | -8.89% | 全A样本，月频 |
| Rank ICIR | -4.84 | |
| 多空年化收益率 | 37.46% | |
| 信息比率 | 4.10 | |
| 月度胜率 | 87.74% | |
| 正交化后Rank IC | -3.18% | 剔除风格因子后 |
| 正交化后Rank ICIR | -1.89 | |

## 8. 风险提示

| 风险点 | 严重程度 | 说明 |
|--------|---------|------|
| 分钟数据质量 | 🔴 高 | 分钟行情数据量大，缺失/异常值处理复杂 |
| 成交量定义差异 | 🔴 高 | 米筐分钟成交量可能是"该分钟成交量"或"累计到该分钟的成交量"，需确认 |
| 开盘收盘剔除 | 🟡 中 | 不同数据源对开盘/收盘时间定义可能不同 |
| 正交化实现 | 🟡 中 | 市值和行业正交化方法（WLS/OLS）影响结果 |
| 参数敏感性 | 🟡 中 | 20日窗口、1倍标准差阈值、等权合成均为研报设定 |
| 历史区间外推 | 🟡 中 | 研报区间2013-2022，近年市场结构变化可能使因子失效 |

## 9. 与原研报的差异记录

| 研报原文 | Spec 理解 | 偏差说明 |
|---------|---------|---------|
| "剔除开盘和收盘数据" | 只保留09:30-11:30和13:00-14:56 | 假设米筐数据不含集合竞价，需验证 |
| "分钟频成交量的增加量的均值和标准差" | 日内所有分钟成交量增量的截面统计 | 标准定义 |
| "均值+一倍标准差" | mean + 1×std | 严格按研报阈值 |
| "耀眼5分钟" | 激增时刻及随后4分钟 | 共5分钟 |
| "最近20个交易日" | 20日滚动窗口 | 非自然日，需处理节假日 |
| "市值和行业正交化" | WLS回归取残差 | 研报未明确具体方法，需假设 |

---

