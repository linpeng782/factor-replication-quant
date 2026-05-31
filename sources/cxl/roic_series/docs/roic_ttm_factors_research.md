# ROIC_TTM 系列因子研究文档

> 研究对象：`roic_ttm_ind_rnk8`、`roic_ttm_all_rnk8`、`roic_ttm_dev_std8`  
> 研究日期：2026-05-18  
> 研究方法：API 调用验证 + 小范围全流程实验  

---

## 1. 因子定义拆解

### 1.1 roic_ttm_ind_rnk8（行业内排名最小值）

| 要素 | 内容 |
|------|------|
| 名称 | 过去8期ROIC_TTM行业内排名的最小值 |
| 核心指标 | ROIC_TTM（投入资本回报率，Trailing Twelve Months） |
| 计算范围 | **全市场个股**（`all_instruments("CS")`，约5548只），行业内排名 |
| 时间窗口 | 过去 **8个财报期**（约2年） |
| 聚合方式 | 8个财报期行业内排名的 **最小值**（数值越小排名越靠前） |
| 因子方向 | 正向（排名最小值越小，预期收益越好） |

### 1.2 roic_ttm_all_rnk8（全市场排名最小值）

与 `roic_ttm_ind_rnk8` 的唯一区别：**排名范围是全市场，不分行业**。

### 1.3 roic_ttm_dev_std8（稳定性因子）

| 要素 | 内容 |
|------|------|
| 公式 | `1 / 过去8期ROIC_TTM的标准差` |
| 含义 | ROIC_TTM在过去8个财报期中越稳定（标准差越小），因子值越大 |
| 方向 | 正向 |

---

## 2. 核心 API 调用清单

### 2.1 股票池获取

```python
import rqdatac as rq
rq.init()

# 全市场A股（Common Stock）
all_stocks = rq.all_instruments(type='CS')['order_book_id'].tolist()
# → 约 5548 只（2024年数据）
```

### 2.2 ROIC_TTM 获取

```python
# 方案A：批量时间序列（推荐）
df = rq.get_factor(
    all_stocks,
    'return_on_invested_capital_ttm',
    start_date='2016-01-01',
    end_date='2025-12-31'
)
# → MultiIndex DataFrame: (order_book_id, date) × value

# 方案B：单日查询（用于验证）
df = rq.get_factor(all_stocks, 'return_on_invested_capital_ttm', date='2024-01-02')
```

**关键发现**：米筐已预计算 `return_on_invested_capital_ttm` 字段，**无需手动计算** `operating_profitTTM / invested_capital_ttm`。

### 2.3 中信行业分类（kdcj 方法）

```python
# 获取中信一级行业映射（含历史变化）
industry_map_dict = rq.client.get_client().execute('__internal__zx2019_industry')
df_ind = pd.DataFrame(
    industry_map_dict,
    columns=['first_industry_name', 'order_book_id', 'start_date']
)
df_ind['start_date'] = pd.to_datetime(df_ind['start_date'])
df_ind = df_ind.sort_values(['order_book_id', 'start_date'])

# 转为宽表，按 start_date 做前向填充
df_ind_wide = df_ind.pivot(
    index='start_date',
    columns='order_book_id',
    values='first_industry_name'
).ffill()

# 查询某交易日的行业分类
date_dt = pd.Timestamp('2024-01-02')
valid_dates = df_ind_wide.index[df_ind_wide.index <= date_dt]
industry_today = df_ind_wide.loc[valid_dates[-1]]  # 取最新生效的行业
```

**行业数量**：33个中信一级行业。

---

## 3. 数据特性研究

### 3.1 ROIC_TTM 的更新频率

ROIC_TTM 是**日频数据**，但值只在**财报发布日**更新。在非财报发布日，值保持不变。

**实测：万科（000002.XSHE）2020-2024 年财报期**

| 财报期日期 | ROIC_TTM | 距上期天数 |
|-----------|----------|-----------|
| 2020-01-02 | 0.124410 | - |
| 2020-03-18 | 0.119062 | 76 |
| 2020-04-28 | 0.114780 | 41 |
| 2020-08-28 | 0.109032 | 122 |
| 2020-10-30 | 0.108519 | 63 |
| 2021-03-31 | 0.108275 | 152 |
| ... | ... | ... |
| 2024-08-30 | 0.000373 | 122 |
| 2024-10-31 | -0.016328 | 62 |

**统计**：
- 2015-2024 年共 **41个财报期**
- 平均间隔 **89.7天**，中位数 **78.5天**
- 最短 **23天**，最长 **154天**
- 每年约 **4~5个财报期**

### 3.2 财报期不同步（关键发现）

**不同股票的 ROIC_TTM 变化日不完全一致**：

| 股票 | 2024年财报期变化日 |
|------|-------------------|
| 万科 000002.XSHE | 01-02, 03-29, 04-30, 08-30, 10-31 |
| 茅台 600519.XSHG | 01-02, 04-03, 04-26, 08-09, 10-25 |

→ **结论**：不能用统一财报期，必须**逐股票识别**自己的 ROIC_TTM 变化日。

### 3.3 "8期"的精确定义

实验对比了两种实现方式：

- **方式A**：`rolling(window=504, min_periods=1).min()`（504个交易日 ≈ 8个季度）
- **方式B**：识别每只股票的8个财报期 → `rolling(window=8, min_periods=1).min()`

**对比结果**（2020-2024年，3只测试股）：

| 股票 | 方式A vs 方式B 完全一致率 | 平均差异 | 最大差异 |
|------|-------------------------|---------|---------|
| 万科 000002.XSHE | **36.6%** | 1.29 | 5.0 |
| 茅台 600519.XSHG | **91.8%** | 0.08 | 1.0 |
| 五粮液 000858.XSHE | **77.6%** | 0.24 | 3.0 |

**分析**：
- 万科差异大的原因是财报期间隔极不均匀（最短28天，最长154天）
- `rolling(504)` 会在长间隔期"浪费"窗口空间，导致更早的排名被挤出
- 方式B（8个财报期）严格符合因子名称中"8期"的含义

→ **结论**："8期"应理解为 **8个财报期**（ROIC_TTM变化的日期），而非504个交易日。

### 3.4 银行类股票无 ROIC_TTM

| 股票 | 2024-01-02 ROIC_TTM |
|------|---------------------|
| 平安银行 000001.XSHE | **NaN** |
| 浦发银行 600000.XSHG | **NaN** |
| 工商银行 601398.XSHG | **NaN** |
| 建设银行 601939.XSHG | **NaN** |

银行类股票的 `return_on_invested_capital_ttm` **始终为 NaN**，因为银行业务模式不适用ROIC指标。这些股票的因子值自然为 NaN，会在清洗阶段被 mask 处理。

### 3.5 行业分类历史变化

中信行业分类不是静态的——股票可能因业务转型而被重新分类。

**示例数据**（`__internal__zx2019_industry` 原始返回）：

| first_industry_name | order_book_id | start_date |
|---------------------|---------------|------------|
| 钢铁 | 000932.XSHE | 2019-12-02 |
| 基础化工 | 600141.XSHG | 2019-12-02 |
| 房地产 | 000002.XSHE | 2019-12-02 |
| ... | ... | ... |

→ 必须使用 `pivot + ffill` 按 `start_date` 处理历史变化，不能只用最新行业分类。

---

## 4. 计算链路详解

### 4.1 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: 获取全市场ROIC_TTM时间序列                               │
│  API: rq.get_factor(stocks, 'return_on_invested_capital_ttm')   │
│  Output: long格式 DataFrame (order_book_id, date, roic_ttm)     │
│  Shape: ~13,000,000 rows (5500 stocks × 2400 days)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: 获取中信行业分类（含历史变化）                            │
│  API: rq.client.get_client().execute('__internal__zx2019_industry')│
│  Processing: pivot(start_date × order_book_id) → ffill           │
│  Output: 每个交易日每只股票的最新行业分类                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: 逐日计算行业内排名                                       │
│  Logic: groupby(['date', 'industry'])['roic_ttm'].rank()        │
│  Method: rank(ascending=False, method='min')                    │
│  Note: 数值越小排名越靠前（第1名 = 最高ROIC_TTM）                 │
│  Output: 排名宽表 (date × order_book_id)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: 识别每只股票的财报期（ROIC_TTM变化日）                    │
│  Logic: 对每只股票的ROIC_TTM序列，找出值变化的日期                  │
│  Output: 每只股票的财报期日期列表（约20~40个/10年）               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: 提取财报期排名并计算过去8期最小值                          │
│  Logic: report_ranks.rolling(window=8, min_periods=1).min()     │
│  Output: 每只股票的8期排名最小值序列（仅财报期有值）                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 6: 前向填充到所有交易日                                       │
│  Logic: factor.reindex(all_trading_dates).ffill()               │
│  Reason: 因子值为日频，非财报发布日沿用最近财报期的值                │
│  Output: 最终因子宽表 (date × order_book_id)                     │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 详细伪代码

```python
import pandas as pd
import rqdatac as rq

# ============ Step 1: 股票池 ============
all_stocks = rq.all_instruments(type='CS')['order_book_id'].tolist()

# ============ Step 2: 获取ROIC_TTM时间序列 ============
# 返回 MultiIndex DataFrame: (order_book_id, date) -> roic_ttm
roic_raw = rq.get_factor(
    all_stocks,
    'return_on_invested_capital_ttm',
    start_date='2016-01-01',
    end_date='2025-12-31'
)
roic_long = roic_raw.reset_index()
roic_long.columns = ['order_book_id', 'date', 'roic_ttm']

# 转为宽表: date × order_book_id
roic_wide = roic_long.pivot(index='date', columns='order_book_id', values='roic_ttm')

# ============ Step 3: 获取中信行业分类 ============
industry_map = rq.client.get_client().execute('__internal__zx2019_industry')
ind_df = pd.DataFrame(
    industry_map,
    columns=['first_industry_name', 'order_book_id', 'start_date']
)
ind_df['start_date'] = pd.to_datetime(ind_df['start_date'])
ind_df = ind_df.sort_values(['order_book_id', 'start_date'])
ind_wide = ind_df.pivot(
    index='start_date', columns='order_book_id', values='first_industry_name'
).ffill()

# ============ Step 4: 逐日计算行业内排名 ============
trading_dates = roic_wide.index.tolist()
rank_wide = pd.DataFrame(index=trading_dates, columns=roic_wide.columns, dtype=float)

for date in trading_dates:
    date_dt = pd.Timestamp(date)
    
    # 获取该日最新的行业分类
    valid_ind_dates = ind_wide.index[ind_wide.index <= date_dt]
    if len(valid_ind_dates) == 0:
        continue
    ind_today = ind_wide.loc[valid_ind_dates[-1]]
    
    # 获取该日ROIC_TTM
    roic_today = roic_wide.loc[date]
    
    # 合并
    df_day = pd.DataFrame({
        'roic_ttm': roic_today,
        'industry': ind_today
    }).dropna()
    
    # 行业内排名（从高到低）
    df_day['rank'] = df_day.groupby('industry')['roic_ttm'].rank(
        method='min', ascending=False
    )
    
    # 写回宽表
    rank_wide.loc[date, df_day.index] = df_day['rank'].values

# 前向填充排名（非财报发布日排名不变）
rank_wide = rank_wide.ffill()

# ============ Step 5: 逐股票识别财报期 + 过去8期最小值 ============
# 注意：必须逐股票处理，因为不同股票财报期不同步
factor = pd.DataFrame(index=trading_dates, columns=roic_wide.columns, dtype=float)

for stock in roic_wide.columns:
    roic_series = roic_wide[stock]
    
    # 识别该股票的财报期（ROIC_TTM值变化的日子）
    changes = roic_series[roic_series != roic_series.shift(1)]
    report_dates = changes.dropna().index.tolist()
    
    if len(report_dates) == 0:
        continue
    
    # 提取这些日期的排名
    report_ranks = rank_wide.loc[rank_wide.index.isin(report_dates), stock]
    
    # 过去8期排名最小值
    min8 = report_ranks.rolling(window=8, min_periods=1).min()
    
    # 构建完整时间序列（仅财报期有值，其余NaN）
    factor_series = pd.Series(index=trading_dates, dtype=float)
    factor_series[min8.index] = min8.values
    
    # 前向填充到所有交易日
    factor[stock] = factor_series.ffill()

# factor 即为最终因子值宽表 (date × order_book_id)
```

### 4.3 实测案例：万科（000002.XSHE）

**房地产行业 2020-2024 年完整计算过程**：

| 财报期 | ROIC_TTM | 行业内排名 | 过去8期最小值 |
|--------|----------|-----------|--------------|
| 2020-01-02 | 0.124410 | 10 | **10** |
| 2020-03-18 | 0.119062 | 12 | **10** |
| 2020-04-28 | 0.114780 | 13 | **10** |
| 2020-08-28 | 0.109032 | 13 | **10** |
| 2020-10-30 | 0.108519 | 14 | **10** |
| 2021-03-31 | 0.108275 | 12 | **10** |
| 2021-04-23 | 0.104235 | 11 | **10** |
| 2021-08-30 | 0.095843 | 12 | **10** |
| 2021-10-29 | 0.088198 | 15 | **11** ← 窗口滑动，10被挤出 |
| 2022-03-31 | 0.063839 | 29 | **11** |
| ... | ... | ... | ... |
| 2024-03-29 | 0.032187 | 44 | **17** |
| 2024-04-30 | 0.028665 | 38 | **17** |
| 2024-08-30 | 0.000373 | 66 | **17** |
| 2024-10-31 | -0.016328 | 79 | **17** |

**解读**：
- 2020-2021年，万科在房地产行业内ROIC_TTM表现较好，最好排名为第10名
- 2022年后行业下行，排名逐渐恶化至第79名
- 到2024年10月，"过去8期排名最小值"为 **17**，意味着最近8个财报期中最好的表现是第17名
- 因子方向为正向，即该值越小（历史上行业内排名越靠前），预期收益越好

---

## 5. 性能评估

### 5.1 各步骤耗时（实测）

| 步骤 | 操作 | 数据规模 | 耗时 |
|------|------|---------|------|
| 1 | 获取全市场ROIC_TTM（2年） | 5548只 × 726天 = 268万行 | **112秒** |
| 2 | 获取中信行业分类 | 9477条映射记录 | **19.5秒** |
| 3 | 行业内排名计算 | 每日~5000只 × 33行业 | **数秒**（内存操作） |
| 4 | 逐股票识别财报期+rolling(8).min() | 5500只 × 20期 | **1-2分钟** |
| | **合计（2年数据）** | | **~2.5分钟** |

### 5.2 全历史估算（2016-2025，约10年）

| 步骤 | 估算耗时 |
|------|---------|
| 获取ROIC_TTM（10年） | **~500-600秒**（8-10分钟） |
| 行业+排名+8期min | **~1-2分钟** |
| **总计** | **~10分钟/因子** |

### 5.3 性能瓶颈

**主要瓶颈**：`rq.get_factor(all_stocks, 'return_on_invested_capital_ttm', ...)` 的批量时间序列获取。

- 第一次调用有预热时间（~50秒），后续缓存加速
- 逐日调用 `get_factor(stocks, ..., date=date)` 更慢（~1-5秒/天）
- **推荐方案**：一次性用 `start_date + end_date` 获取完整时间序列

---

## 6. 三个相关因子对比

| 维度 | roic_ttm_ind_rnk8 | roic_ttm_all_rnk8 | roic_ttm_dev_std8 |
|------|-------------------|-------------------|-------------------|
| **排名范围** | 行业内 | 全市场 | 不涉及排名 |
| **计算公式** | `min(过去8期行业内排名)` | `min(过去8期全市场排名)` | `1 / std(过去8期ROIC_TTM)` |
| **对行业的依赖** | 高（需行业分类） | 无 | 无 |
| **对ROIC_TTM变化日的依赖** | 高（需逐股票识别财报期） | 高 | 高 |
| **因子含义** | 历史上行业内最好排名 | 历史上全市场最好排名 | ROIC_TTM的稳定性 |
| **实现复杂度** | 高 | 中 | 低 |

### 6.1 共用计算模块

三个因子可以**共用**以下计算结果：
1. 全市场ROIC_TTM时间序列（Step 1）
2. 财报期识别（Step 4）

差异仅在最后一步的聚合方式：
- `ind_rnk8`: 行业内排名 → rolling(8).min()
- `all_rnk8`: 全市场排名 → rolling(8).min()
- `dev_std8`: 直接对ROIC_TTM序列 → rolling(8).std() → 取倒数

---

## 7. 待确认问题

### 7.1 问题1：新股的 min_periods 处理 ✅ 已确认

**结论**：`min_periods = 4`

**依据**：
- 新股 mask 过滤 252 个交易日（≈ 379 日历天，约 1.04 年）
- 抽样验证：62% 的股票在 252 天解除时刚好有 4 个财报期，29% 有 3 个，9% 有 5 个
- `min_periods=4` 与 mask 基本同步，避免 mask 解除后还要等 2-5 个月才有因子值
- `min_periods=5` 会导致 91% 的股票出现额外空窗期，浪费有效数据

### 7.2 问题2：行业内排名的具体方法 ✅ 已确认

**结论**：`method = 'min'`

**依据**：
- `method='min'` 是量化领域最通用的做法，简单稳定
- 与现有清洗/评估 pipeline 一致
- `average` 会产生小数排名（2.5），对后续"取最小值"操作不友好
- `dense` 会压缩排名范围，不同行业间可比性变差

### 7.3 问题3：排名方向 ✅ 已确认

**结论**：`ascending = False`（ROIC_TTM 越高 → 排名数值越小 → 排名越靠前）

**经济含义**：
- 因子值 = `min(过去8期排名)`，数值越小 = 历史上行业内表现越好
- 方向为正向：曾经做到过行业顶尖的公司（如第4名），比从未进过前20的公司更有投资价值
- 即使当前业绩下滑，"曾经辉煌过"的龙头基因意味着复苏潜力更大

### 7.4 问题4：行业分类缺失的处理 ✅ 已确认

**结论**：**选项 A — 保持 NaN**

**依据**：
- 实测：约 4.7% 股票（260只）无中信行业分类
- 缺失行业分类的股票主要是新股、退市股、北交所股票等，本来就会被 mask 过滤
- 强行归入"综合"行业会污染该行业的排名分布

### 7.5 问题5：银行股的处理 ✅ 已确认

**结论**：**选项 A — 保持 NaN**

**依据**：
- 银行类股票（000001.XSHE、600000.XSHG、601398.XSHG 等）的 `return_on_invested_capital_ttm` 始终为 NaN
- 银行业务模式（高杠杆、利息收入为主）不适用 ROIC 指标
- 自然为 NaN 是财务常识，清洗阶段会被 mask 处理

### 7.6 问题6：ROIC_TTM 与 ROIC（非TTM）的区别 ✅ 已确认

**结论**：因子名称明确为 `roic_ttm`，使用 `return_on_invested_capital_ttm`。无需考虑非 TTM 版本。

---

## 8. 附录：关键实验代码

### 8.1 验证 "8期" = 8个财报期

```python
# 获取目标股票ROIC_TTM时间序列
stock = '000002.XSHE'
df = rq.get_factor([stock], 'return_on_invested_capital_ttm',
                   start_date='2020-01-01', end_date='2024-12-31')
df = df.reset_index()
df.columns = ['order_book_id', 'date', 'roic_ttm']

# 识别财报期（值变化的日子）
changes = df[df['roic_ttm'] != df['roic_ttm'].shift(1)].dropna()
report_dates = changes['date'].tolist()
print(f'财报期数量: {len(report_dates)}')
# → 21个财报期（2020-2024）

# 相邻财报期间隔
changes['days_diff'] = changes['date'].diff().dt.days
print(changes['days_diff'].describe())
# → mean=89.7, std=46.1, min=23, max=154
```

### 8.2 获取中信行业分类

```python
industry_map_dict = rq.client.get_client().execute('__internal__zx2019_industry')
df_ind = pd.DataFrame(
    industry_map_dict,
    columns=['first_industry_name', 'order_book_id', 'start_date']
)
print(f'总记录数: {len(df_ind)}')
print(f'唯一股票数: {df_ind.order_book_id.nunique()}')
print(f'行业数: {df_ind.first_industry_name.nunique()}')
# → 总记录数: 9477, 唯一股票数: 5754, 行业数: 33
```

### 8.3 对比两种 "8期" 实现

```python
# 方式A: rolling(504日)
method_a = rank_wide[stock].rolling(window=504, min_periods=1).min()

# 方式B: 按财报期rolling(8)
roic_series = roic_wide[stock]
changes = roic_series[roic_series != roic_series.shift(1)].dropna()
report_dates = changes.index.tolist()
report_ranks = rank_wide.loc[rank_wide.index.isin(report_dates), stock]
min8 = report_ranks.rolling(window=8, min_periods=1).min()
method_b = min8.reindex(rank_wide.index).ffill()

# 对比
diff = (method_a - method_b).abs()
print(f'完全一致率: {(diff == 0).mean()*100:.1f}%')
print(f'平均差异: {diff.mean():.2f}')
print(f'最大差异: {diff.max():.2f}')
```

---

## 9. 下一步工作

1. **确认待确认问题**（Section 7）
2. **编写 Spec 文档**（`specs/roic_ttm_ind_rnk8/spec.md` + `spec.yaml`）
3. **实现 YOLO 计算链路**（可能需要扩展 `rolling` 和 `groupby_rank` 算子）
4. **批量复现**三个ROIC_TTM相关因子


---

## 10. 实现层面的细节

### 10.1 排名时 NaN 的处理

**规则**：行业内排名**只在该期有 ROIC_TTM 值的股票中**进行，`NaN` 的股票不参与排名，直接给 `NaN`。

**原因**：
- 如果让 NaN 参与排名，pandas 默认会把 NaN 排在最后（或根据 `na_option` 参数处理）
- 这会导致有值的股票排名被人为"抬高"或"压低"
- 正确的做法是先 `dropna()`，再对有效值做排名

**代码示意**：
```python
df_day = df_day.dropna(subset=['roic_ttm', 'industry'])
df_day['rank'] = df_day.groupby('industry')['roic_ttm'].rank(method='min', ascending=False)
```

### 10.2 8期窗口中某期为 NaN 的处理

**规则**：`rolling(8, min_periods=4)` 且**只统计有效排名期**。

**场景**：某只股票在过去8个财报期中，有2期 ROIC_TTM 为 NaN（如停牌、未发布财报），实际只有6期有效排名。

**处理**：
- 对这6个有效排名取最小值（满足 `min_periods=4`）
- **不是**要求8期都必须有值，也不是把 NaN 当作一个排名值参与比较
- 本质上：`report_ranks.dropna().rolling(8, min_periods=4).min()`

### 10.3 三个因子共用计算模块

`ind_rnk8`、`all_rnk8`、`dev_std8` 可以共用以下中间结果：

| 中间结果 | 计算耗时 | 复用方式 |
|---------|---------|---------|
| 全市场 ROIC_TTM 时间序列 | ~10分钟 | 三个因子直接复用同一宽表 |
| 逐股票财报期识别 | ~1分钟 | 复用 `report_dates` 字典 |

差异仅在最后一步聚合：
- `ind_rnk8`：行业内排名 → `rolling(8, min_periods=4).min()`
- `all_rnk8`：全市场排名 → `rolling(8, min_periods=4).min()`
- `dev_std8`：直接对 ROIC_TTM → `rolling(8, min_periods=4).std()` → 取倒数

→ **建议一次计算，三个因子同时产出**。

---

## 11. 算子拓展需求分析

### 11.1 现有算子能力盘点

| 算子 | 已有功能 | 缺失功能 |
|------|---------|---------|
| `fetch` | `get_pit_financials_ex`、`get_factor` | 不支持自定义 API（如 `__internal__zx2019_industry`） |
| `compute` | 列运算公式（`pd.eval`） | 不支持 `groupby`、`rolling` 等复杂操作 |
| `filter` | `df.query(condition)` | — |
| `rank` | `pct=True` 的百分比排名，支持 `industry_rank` | **不支持 `pct=False` 的顺序排名**；不支持 `ascending`、`method` |
| `transform` | `diff_quarterly`、`yoy`、`qoq`、`zscore` | **不支持 `ffill`、`rolling` 等** |

### 11.2 复现 roic_ttm_ind_rnk8 所需的最小算子拓展

#### 拓展 1：扩展 `rank` 算子

**当前限制**：
- 只支持 `pct=True`（返回 0~1 的百分比排名）
- 不支持 `ascending` 参数（默认 `ascending=True`）
- 不支持 `method` 参数（默认 `average`）

**所需能力**：

```yaml
action: rank
method: industry_rank
rank_column: roic_ttm
group_column: industry
pct: false           # 新增：顺序排名（1, 2, 3...）而非百分比
ascending: false     # 新增：ROIC_TTM 越高排名越靠前
method: min          # 新增：并列取最小排名
output: ind_rank
```

**实现改动量**：小。在 `op_rank` 中把写死的 `pct=True, ascending=True` 改为从 `step` 中读取参数即可。

#### 拓展 2：新增 `rolling` 算子

**用途**：对时间序列做滚动窗口聚合，支持逐股票（`group_by`）的 rolling。

**所需能力**：

```yaml
action: rolling
input: ind_rank        # 输入：排名宽表 (date × order_book_id)
window: 8              # 窗口大小：8个财报期
min_periods: 4         # 最少有效值：4期
agg: min               # 聚合函数：min / max / mean / std / sum
group_by: order_book_id # 按股票逐只 rolling
output: ind_rank_min8
```

**核心逻辑**：
```python
def op_rolling(ctx, step, fetcher):
    df = _resolve_input(ctx, step.get("input"))
    window = step.get("window", 8)
    min_periods = step.get("min_periods", 1)
    agg = step.get("agg", "min")
    group_by = step.get("group_by", "order_book_id")
    
    # 如果是宽表 (date × stock)，按列（逐股票）rolling
    if group_by in df.columns:
        # long 格式
        result = df.groupby(group_by).rolling(window=window, min_periods=min_periods).agg(agg)
    else:
        # wide 格式：逐列 rolling
        result = df.rolling(window=window, min_periods=min_periods).agg(agg)
    
    ctx[output_name] = result
    return result
```

**实现难点**：
- 需要区分输入是 **long 格式**（有 `date` 和 `order_book_id` 列）还是 **wide 格式**（date 为 index，stock 为 columns）
- wide 格式下，`df.rolling(window=8)` 默认是按**行**（时间方向）rolling，这正是我们需要的
- 但 `min_periods=4` 在 wide 格式下会同时作用于所有列，也是正确的

#### 拓展 3：新增 `ffill` 算子（或扩展 `transform`）

**用途**：把财报期的因子值前向填充到所有交易日。

**方案 A：独立 `ffill` 算子**

```yaml
action: ffill
input: factor          # 仅财报期有值的宽表
by: order_book_id      # 逐列 ffill
output: factor_daily
```

**方案 B：扩展 `transform` 算子**

```yaml
action: transform
method: ffill
input: factor
output: factor_daily
```

**推荐方案 B**：`ffill` 属于数据转换操作，归入 `transform` 更统一。实现极简单：

```python
def _compute_ffill(df: pd.DataFrame, step: Dict, ctx: Dict) -> pd.DataFrame:
    """前向填充"""
    value_cols = [c for c in df.columns if c not in ("order_book_id", "date", "quarter")]
    for col in value_cols:
        df[col] = df[col].ffill()
    return df
```

#### 拓展 4：扩展 `fetch` 算子支持自定义 API

**用途**：获取 `__internal__zx2019_industry` 等行业分类数据。

**方案**：在 `fetch` 算子中新增 `api: custom` 模式：

```yaml
action: fetch
api: custom
command: "__internal__zx2019_industry"
output: industry_map
```

**实现**：
```python
elif api == "custom":
    command = step.get("command", "")
    raw = fetcher._rq.client.get_client().execute(command)
    df = pd.DataFrame(raw, columns=step.get("columns", []))
```

**替代方案**：新增独立的 `fetch_industry` 算子，更语义化：

```yaml
action: fetch_industry
type: zx  # 中信一级行业
output: industry_map
```

### 11.3 算子拓展优先级

| 优先级 | 算子 | 工作量 | 复用性 |
|--------|------|--------|--------|
| P0 | 扩展 `rank`（支持 `pct/ascending/method`） | 小（加3个参数） | 高（所有排名类因子） |
| P0 | 扩展 `transform`（新增 `ffill` method） | 极小（1个函数） | 极高（所有低频→日频因子） |
| P0 | 扩展 `fetch`（支持自定义 API）或新增 `fetch_industry` | 小 | 高（所有需要行业分类的因子） |
| P1 | 新增 `rolling` 算子 | 中（需处理 long/wide 格式） | 极高（所有滚动窗口因子） |

### 11.4 替代方案：不走 YOLO，直接写脚本

如果算子拓展工作量较大，可以先写一个**独立脚本**实现该因子：

```python
# scripts/compute_roic_ttm_factors.py
# 直接调用 rqdatac，输出到 raw_factor/ 目录
```

**权衡**：
- **YOLO 方案**：可复用、可配置、符合项目规范，但需拓展算子
- **脚本方案**：开发快、一次性，但不符项目规范，无法自动化批量复现

**建议**：先拓展算子（P0 的三项工作量都很小），再走 YOLO 流程。`rolling` 算子（P1）可以先在脚本中实现，后续再抽象。

---

## 12. 下一步工作

1. **拓展算子**（P0 三项，预计 1-2 小时）
2. **编写 Spec 文档**（`specs/roic_ttm_ind_rnk8/spec.md` + `spec.yaml`）
3. **YOLO 执行 + 评估**（全历史 2016-2025）
4. **批量复现** `roic_ttm_all_rnk8` 和 `roic_ttm_dev_std8`（复用中间结果）
