# npf_mrq_sue8 — 单季度净利润 SUE（标准化的盈利意外）

> **状态**：已通过 LLM 自动生成 spec + spec_schema 静态校验 + YOLO 端到端跑通
> **复现 IC/ICIR**：5d IC=0.0157，ICIR=0.204，单调性 +0.998（基准 ICIR=1.28，差距待分析）
> **生成方式**：研报描述（`inputs/npf_mrq_sue8.md` 4 行字）→ kimi-k2.6 一次产 yaml → 静态校验 0 重试通过

---

## 1. 经济原理：什么是 SUE？

**SUE = Standardized Unexpected Earnings**（标准化的"业绩超预期"指标）。最早由
Foster、Olsen、Shevlin 在 1984 年的论文中提出，是研究 **PEAD（Post-Earnings
Announcement Drift，盈利公告后股价漂移）** 最经典的工具。

### 核心直觉

不是看公司这季度赚得绝对多还是少，而是看它**比"应该赚的"多出了多少个标准差**：

```
SUE = (Actual − Expected) / σ
        ↑          ↑       ↑
       实际      预期      历史波动率
```

三个分量：

| 分量 | 含义 | SUE 中如何取 |
|---|---|---|
| Actual  | 最新季度实际净利润 | `mrq_0` |
| Expected | 基于历史趋势的预期净利润 | `mrq_4 + mean_diff`（去年同期 + 季度平均增长） |
| σ        | 历史相邻季度差分的波动率 | `std_diff` |

### 为什么要 normalize（除以 σ）？

茅台一季度赚 100 亿，假如这季度赚 105 亿，超预期 5 亿，听起来很多。
某煤炭公司一季度赚 1 亿，假如这季度赚 6 亿，超预期 5 亿，听起来"也是 5 亿"。
但茅台波动小（季度 std≈2 亿），煤炭波动大（季度 std≈10 亿）：

- 茅台：5 / 2 = **2.5 个 σ**（极其罕见的超预期）
- 煤炭：5 / 10 = **0.5 个 σ**（很正常的波动）

**除以 σ 把"金额"变成"程度"，跨股票才可比**。这就是因子的核心设计动机。

---

## 2. 研报描述的歧义与 LLM 自动修正

我们 `inputs/npf_mrq_sue8.md` 里抄自 `PROGRESS.md` 的描述：

> 取过去 8 个报告期单季度净利润，求净利润差分的均值和标准差，
> 因子值为（去年同期净利润 + 差分均值）/ 差分标准差。

**字面读 = `(mrq_4 + mean_diff) / std_diff`**——根本没用到 mrq_0（实际值）。
这显然是**研报转述时漏了 "mrq_0 −"**。LLM 用金融领域知识识别出"这是 SUE
模板"，自动补回缺失的减号，产出的是规范公式：

```
SUE = (mrq_0 − mrq_4 − mean_diff) / std_diff
```

这次自动修正是 LLM 单次产出 spec 不需要重试的关键。**但反过来也要警惕**：
有时研报作者用的就是非标准版本，LLM 自作主张修正反而会让复现 IC 偏离原文。
本因子 ICIR 偏低（0.20 vs 1.28）的主因不是这个，但下次遇到非常规公式因子时
要主动检查 LLM 的"修正"。

---

## 3. 计算步骤（与 spec.yaml 一一对应）

### 数学定义

```
fields needed: net_profit_mrq_0, _1, _2, _3, _4, _5, _6, _7   (8 期)
diffs        : d_i = mrq_i − mrq_(i+1)  for i in 0..6           (7 个相邻差分)
diff_mean    = mean(d_0..d_6)
diff_std     = std(d_0..d_6, ddof=1)
SUE          = (mrq_0 − mrq_4 − diff_mean) / diff_std
```

### spec 实现（5 步）

| Step | action | 关键参数 | 输出新增列 |
|------|--------|---------|----------|
| 1 | fetch | 8 字段（net_profit_mrq_0..7） | 8 个原始列 |
| 2 | compute | `(mrq_0 − mrq_7) / 7.0` | `mean_diff` |
| 3 | compute | `(mrq_6−mrq_7)² + … + (mrq_0−mrq_1)²` | `sum_sq_diff` |
| 4 | compute | `sqrt((sum_sq_diff − 7·mean_diff²) / 6)` | `std_diff` |
| 5 | compute | `(mrq_0 − mrq_4 − mean_diff) / std_diff` | `npf_mrq_sue8` |

最后引擎按 `factor.column = npf_mrq_sue8` 在主表上 pivot 出宽表落盘。

---

## 4. LLM 用了两个数学技巧绕开算子缺口

LLM 没有调用任何"行向跨列 std"或"行向跨列 mean"算子（我们当时也没有），
而是用代数恒等式硬把这些能力嵌进了 `compute` 的公式里。

### 技巧 A：望远镜求和（telescoping sum）

7 个相邻差分相加时中间项全部互相抵消：

```
d_0 + d_1 + ... + d_6
= (mrq_0 − mrq_1) + (mrq_1 − mrq_2) + ... + (mrq_6 − mrq_7)
= mrq_0 − mrq_1 + mrq_1 − mrq_2 + mrq_2 − mrq_3 + ... + mrq_6 − mrq_7
                ↑↑                ↑↑                       ↑↑
              抵消              抵消                     抵消
= mrq_0 − mrq_7
```

所以：

```
mean_diff = (mrq_0 − mrq_7) / 7
```

只用 2 个字段就计算出了 7 个差分的均值。**mrq_1..mrq_6 不是被忽略，而是数学上互相抵消**。

### 技巧 B：方差恒等式 Var(X) = E[X²] − E[X]²

样本方差展开：

```
Σ(d_i − μ)²  =  Σd_i²  −  2μ·Σd_i  +  n·μ²
            =  Σd_i²  −  2μ·(n·μ)  +  n·μ²        （因为 Σd_i = n·μ）
            =  Σd_i²  −  n·μ²
```

样本标准差除以 n−1=6：

```
std_diff = sqrt((sum_sq − 7·mean²) / 6)
```

这避免了显式构造 7 个差分列后再做行向 std，只需"和的平方"和"平方和"两个标量。

### 这两个技巧的代价是什么？

代价是 **平方和 `sum_sq_diff` 必须显式展开成 7 项**（平方不抵消），spec 看起来稍长。
但好处显著：**整个流水线只用 5 步、不需要任何新算子，研究员读公式直接对得上 SUE 经典定义**。

---

## 5. 复现结果

### YOLO 执行参数

| 项 | 值 |
|---|---|
| 评估区间 | 2016-01-04 ~ 2025-12-31 |
| 股票池 | 全市场 A 股（5548 只） |
| Fetch | 8 字段 × 12 批 × 12 线程，wall clock 379s |
| 输出 | (2430, 5548) 宽表，9,680,943 个非空格子 |

### 因子分布（健康）

```
mean   = 0.06    （SUE 应该接近 0）
std    = 0.95    （应该接近 1）
p1     = −2.51
p50    = +0.04
p99    = +2.89
inf    = 0       （std_diff 除零没有爆）
```

量级、对称性、尾部都符合 z-score 形态，**说明数学实现没出错**。

### IC / ICIR / 单调性

| 指标 | 复现 | 基准（PROGRESS.md） |
|---|---|---|
| 5d IC      | +0.0157 | +0.066 |
| 5d ICIR    | +0.204  | +1.28 |
| 20d IC     | +0.0212 | — |
| 20d ICIR   | +0.237  | — |
| 单调性     | **+0.998** | — |
| LongShort Sharpe | +1.622 | — |

**单调性 +0.998 接近完美**，说明因子确实把 5 组排序排得很对；**因子有效，只是有效性弱了**。
ICIR 仅为基准的 1/6。

---

## 6. 与基准 ICIR 1.28 差距的可能根因

按可能性排序：

### (1) 股票池不同（最可能）

研报基准 ICIR 1.28 大概率不是在全市场跑出来的，而是中证 800 / 中证 500 这类
**已经过市值/流动性筛选的窄池子**。SUE 因子对股票池极敏感：

- 大票/中票：盈利数字相对干净，SUE 是真实信号
- 小票/ST/科创/北交所：盈利数字噪声大、操纵频繁、停牌多，**SUE 信号显著衰减**

我们用 `primary_index: ALL`（5548 只），稀释了信号。

### (2) "8 期含 mrq_0" 的轻微泄漏（次要）

LLM 用 `mrq_0..mrq_7` 计算 mean/std；学术严格版用 `mrq_1..mrq_8`（不含当期）。
代入展开：

```
SUE = (mrq_0 − mrq_4 − (mrq_0 − mrq_7)/7) / std_diff
    = (6·mrq_0 − 7·mrq_4 + mrq_7) / (7·std_diff)
```

mrq_0 系数仍是 6/7，"surprise" 信号强度仍在；只是"实际"被自身的"预期"轻微拉低。
**这个理论上会让 ICIR 略低，但不会差 6 倍**。

### (3) 回测期不同

研报基准多来自 2010~2018 样本期，A 股机构化前 SUE 信号更强。
我们跑 2016~2025 包含了机构化加速 + 量化竞争密集化的近期窗口，ICIR 自然衰减。

---

## 7. 验证假设的最简实验

切窄股票池就能定性根因：

```yaml
# specs/npf_mrq_sue8/spec.yaml 改一行
universe:
  primary_index: 000906.XSHG   # 中证 800
```

预期：

| 中证 800 ICIR 实测 | 结论 |
|---|---|
| ≥ 0.6 | 根因是股票池稀释；研报基准是窄池估计 |
| 仍在 0.20 附近 | 根因是回测期 + 方法学的复合问题 |

不需要重新 fetch（米筐数据按 universe 重过滤），只需重跑 evaluate-only ~2 分钟。
建议作为 SUE 调研的下一步。

---

## 8. 文件链接

- spec：`specs/npf_mrq_sue8/spec.yaml`
- LLM 完整对话日志：`specs/npf_mrq_sue8/.llm_session.json`
- 研报输入：`inputs/npf_mrq_sue8.md`
- raw 因子：`{RAW_FACTOR_DIR}/npf_mrq_sue8.parquet`
- cleaned：`{CLEANED_FACTOR_DIR}/npf_mrq_sue8.parquet`
- 评估图：`output/npf_mrq_sue8/evaluation_20160101_20251231.png`
- 评估报告：`output/npf_mrq_sue8/report.md`
