# 国金证券 Alpha 掘金系列之十：均值方差组合优化梳理

> 研报路径：`/nfs/ofs-prediction/peterzhenglinpeng-code/research-paper-reproduction/parsed_output/国金证券/20240328-国金证券-Alpha掘金系列之十：机器学习全流程重构_细节对比与测试`

---

## 一、研报原文中的组合优化描述

研报第 7.2 节仅在 636~652 行给出了组合优化的高层描述：

> 为进一步贴近投资实际，我们此处构建了基于上述机器学习模型的指数增强策略。通过**马科维茨的均值方差优化模型**，对投资组合的**跟踪误差进行限制**，并**控制个股偏离程度以减少策略波动水平**，最大化预期超额收益率。

> 在本篇报告中，我们将年化跟踪误差控制为最大不能超过 5%。使用优化器对投资组合权重进行优化，回测期为 2015 年 2 月 1 日至 2023 年 9 月 30 日，以每月第一个交易日的收盘价进行月频调仓，假定手续费率为单边千二。

研报中的公式是：

$$
\max_{w} \ w^{\mathsf{T}} f
$$

$$
\sqrt{(w - w_{bench})^{\mathsf{T}} \Sigma (w - w_{bench})} \le target\_TE
$$

$$
|w - w_{bench}| \le 1
$$

其中：

- $w \in \mathbb{R}^{n}$：投资组合权重向量
- $w_{bench} \in \mathbb{R}^{n}$：基准指数成分股权重向量
- $f \in \mathbb{R}^{n}$：模型预测信号（预期超额收益）
- $\Sigma \in \mathbb{R}^{n \times n}$：股票收益率协方差矩阵
- $target\_TE$：目标年化跟踪误差

---

## 二、组合优化问题的完整数学形式

### 2.1 目标函数

最大化预期超额收益：

$$
\max_{w} \quad w^{\mathsf{T}} f
$$

等价于最小化负预期收益：

$$
\min_{w} \quad -w^{\mathsf{T}} f
$$

### 2.2 跟踪误差约束

组合相对基准的主动风险（年化跟踪误差）不超过 $target\_TE$：

$$
\sqrt{(w - w_{bench})^{\mathsf{T}} \Sigma (w - w_{bench})} \le target\_TE
$$

两边平方后：

$$
(w - w_{bench})^{\mathsf{T}} \Sigma (w - w_{bench}) \le target\_TE^{2}
$$

### 2.3 个股偏离约束

研报原文写的是 $w - w_{bench} \le 1$，结合上下文应为个股相对基准的主动偏离上限：

$$
|w_i - w_{bench,i}| \le \delta, \quad i = 1, 2, \dots, n
$$

其中 $\delta$ 通常取 1% 或 5%（研报未明确）。

### 2.4 完整 QP 形式

补全工程上必需的权重和约束与非负约束：

$$
\begin{aligned}
\max_{w} \quad & w^{\mathsf{T}} f \\
\text{s.t.} \quad & (w - w_{bench})^{\mathsf{T}} \Sigma (w - w_{bench}) \le target\_TE^{2} \\
& |w_i - w_{bench,i}| \le \delta, \quad i = 1, 2, \dots, n \\
& \sum_{i=1}^{n} w_i = 1 \\
& w_i \ge 0, \quad i = 1, 2, \dots, n
\end{aligned}
$$

这是一个标准的**带线性约束和二次约束的凸优化问题**（QCQP），当 $\Sigma$ 半正定时可行。

---

## 三、关键参数的研报设定

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| 目标年化跟踪误差 | $\le 5\%$ | 主动风险上限 |
| 调仓频率 | 月频 | 每月第一个交易日收盘价 |
| 回测区间 | 2015-02-01 ~ 2023-09-30 | 约 8 年半 |
| 手续费 | 单边千二 | $0.2\%$ 每笔 |
| 股票池 | 沪深 300 / 中证 500 / 中证 1000 成分股 | 按指数分别训练 |

---

## 四、研报未披露的关键工程细节

研报只给了高层框架，以下工程细节均未说明：

### 4.1 预测信号 $f$ 的构造

- 是直接用 GBDT/NN 的原始预测值？
- 还是截面排序/标准化后的得分？
- 是否经过行业市值中性化？
- 是否做截断（winsorize）？

### 4.2 协方差矩阵 $\Sigma$ 的估计

常见做法：

| 方法 | 公式 | 适用场景 |
| --- | --- | --- |
| 样本协方差 | $\hat{\Sigma} = \frac{1}{T-1} R^{\mathsf{T}} R$ | 数据充足、股票数少 |
| Ledoit-Wolf 压缩 | $\hat{\Sigma} = \lambda F + (1-\lambda) S$ | 股票数多、样本有限 |
| 因子模型 | $\Sigma = B \Sigma_F B^{\mathsf{T}} + D$ | 指数增强常用 |
| 指数加权 | $\hat{\Sigma}_{ij} = \frac{1 - \lambda}{1 - \lambda^T} \sum_{t=1}^{T} \lambda^{T-t} r_{i,t} r_{j,t}$ | 更重视近期数据 |

研报未说明使用哪种方法，也未说明估计窗口长度。

### 4.3 个股偏离上限 $\delta$

- 若 $\delta = 1$ 表示 $100\%$ 偏离，则约束几乎无意义
- 若 $\delta = 1\%$，则个股相对基准最多偏离 1 个百分点
- 从换手率 88%~142% 看，$\delta$ 应该在 $1\% \sim 5\%$ 之间

### 4.4 其他约束

- 是否允许做空？
- 是否有行业偏离上限？
- 是否有个股权重上限（如 $w_i \le 10\%$）？
- 是否约束最低持仓数量？
- 是否处理 ST/停牌/涨停/跌停股票？

### 4.5 优化与回测的耦合

- 优化是否每月滚动重新估计 $\Sigma$？
- 信号 $f$ 是否每月更新？
- 交易成本是否在优化目标中考虑？

---

## 五、与当前项目的关联

当前 `factor-replication-quant-new/ml_core` 只负责**输出信号**（$yhat$ 面板），不直接做组合优化。研报中的指数增强策略属于下游回测/执行层职责。

相关项目边界：

- `ml_core`：输出日频股票信号 $f$
- `backtest_engine`：消费信号 txt，做等权 top-N 回测
- 组合优化层：目前缺失，若要做指数增强需新增

---

## 六、最小可复现实现（Python + cvxpy）

```python
import cvxpy as cp
import numpy as np

n = len(f)                    # 股票数量
w = cp.Variable(n)            # 组合权重
w_bench = ...                 # 基准权重
Sigma = ...                   # 协方差矩阵
f = ...                       # 模型预测信号

target_TE = 0.05              # 年化跟踪误差 5%
delta = 0.01                  # 个股偏离上限 1%

objective = cp.Maximize(w @ f)
constraints = [
    cp.sum(w) == 1,
    w >= 0,
    cp.quad_form(w - w_bench, Sigma) <= target_TE ** 2,
    cp.abs(w - w_bench) <= delta,
]

problem = cp.Problem(objective, constraints)
problem.solve()
optimal_weights = w.value
```

注意：协方差矩阵 $\Sigma$ 需半正定，否则 `quad_form` 可能报错。实际中建议用 Ledoit-Wolf 或因子模型估计。

---

## 七、结论

研报第七节只给出了均值方差组合优化的**高层数学框架**，对工程落地关键细节（信号构造、协方差估计、个股偏离上限、行业约束等）披露不足。因此：

- 该研报**无法直接复现**组合优化部分
- 研报前六节的因子模型结论（RobustZScore、回归、MSE、一次性训练、DART 等）可以复现
- 若要在当前项目中实现指数增强，需自行补全组合优化模块
