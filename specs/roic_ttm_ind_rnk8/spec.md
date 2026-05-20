# roic_ttm_ind_rnk8 — 单季度 ROIC_TTM 行业内排名最小值

## 因子定义

**名称**：`roic_ttm_ind_rnk8`  
**中文名**：过去8期ROIC_TTM行业内排名的最小值  
**类别**：质量 / 财务  
**方向**：正向（排名最小值越小，预期收益越好）

## 经济含义

ROIC（Return on Invested Capital，投入资本回报率）衡量公司使用投入资本的效率。

本因子不关注当前ROIC_TTM的绝对水平，而是关注**过去8个财报期中，该股票在行业内曾经达到过的最好排名**。核心直觉是：**"曾经辉煌过"的公司比"从未好过"的公司更有投资价值**——即使当前业绩暂时下滑，曾经的龙头基因意味着更强的复苏潜力。

## 计算公式

```
roic_ttm_ind_rnk8 = min(过去8个财报期的行业内排名)
```

其中：
- **财报期**：ROIC_TTM 值发生变化的日期（不同股票不同步）
- **行业内排名**：在每个交易日，对同一行业内的股票按 ROIC_TTM 从高到低排名（`rank(method='min', ascending=False)`）
- **8期**：8个财报期（约2年），`min_periods=4`

## 计算链路

### 数据输入

| 数据 | 来源 | 频率 |
|------|------|------|
| ROIC_TTM | `rq.get_factor(stocks, 'return_on_invested_capital_ttm')` | 日频（财报发布日更新） |
| 中信行业分类 | `rq.client.get_client().execute('__internal__zx2019_industry')` | 历史变更记录 |

### 计算步骤

```
Step 1: 获取全市场 ROIC_TTM 时间序列
        → long 格式 (order_book_id, date, return_on_invested_capital_ttm)

Step 2: 获取中信行业分类
        → 原始格式 (first_industry_name, order_book_id, start_date)
        → 自动转换为日频 long 格式 (order_book_id, date, first_industry_name)

Step 3: 行业内排名
        → 按交易日 + 行业分组，ROIC_TTM 从高到低排名
        → 添加 ind_rank 列（数值越小排名越靠前）

Step 4: 过去8期排名最小值（变化日 rolling）
        → 逐股票识别 ROIC_TTM 值变化的日期（财报期）
        → 在这些日期上对 ind_rank 做 rolling(8, min_periods=4).min()
        → 前向填充到所有交易日

Step 5: 输出
        → long 格式自动 pivot 为 wide 格式 (date × order_book_id)
        → 保存为 raw_factor/roic_ttm_ind_rnk8.parquet
```

### 关键处理规则

| 场景 | 处理方式 |
|------|---------|
| 新股不足4期财报 | `min_periods=4`，不足则为 NaN |
| 并列排名 | `method='min'`，并列取最小排名 |
| 排名方向 | `ascending=False`，ROIC_TTM 越高排名越靠前（第1名=最好） |
| 无行业分类的股票 | 因子值为 NaN（约4.7%） |
| 银行股（ROIC_TTM始终NaN） | 因子值为 NaN |
| 行业内某期仅1只股票有值 | 该股票排名=1 |
| 变化日之间非变化日 | 沿用最近变化日的 rolling 结果（ffill） |

## 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 股票池 | 全市场 `all_instruments("CS")` | 约5548只 |
| 时间范围 | 2016-01-01 ~ 2025-12-31 | 评估区间 |
| 行业分类 | 中信一级行业（2019版） | 33个行业 |
| 滚动窗口 | 8个财报期 | 约2年 |
| 最小期数 | 4期 | 与新股mask（252天≈4期财报）对齐 |
| 排名方法 | `method='min'` | 并列取最小排名 |

## 相关因子

| 因子 | 区别 |
|------|------|
| `roic_ttm_all_rnk8` | 全市场排名（不分行业） |
| `roic_ttm_dev_std8` | `1 / 过去8期ROIC_TTM标准差`（稳定性因子） |

## 复用说明

三个因子共用：
1. ROIC_TTM 时间序列（Step 1）
2. 中信行业分类（Step 2）
3. 财报期识别（Step 4 中 implicit）

差异仅在 Step 3~4 的聚合方式。
