# peak_interval_kurt — 量峰间隔峰度（开源_微观_27 因子 f10）

> **状态**：v1 全市场跑通；**信号方向 + 量级正确**，多空收益略低于论文
> **复现 IC/ICIR**：5d IC=+0.0278 / ICIR=+0.345，t=+17.28，positive 64.7%
> **复现回测**：LongShort 年化 **+16.02%**，**Sharpe 2.26**，单调性 **+0.910**
> **论文基准**：多空年化 23.30%，IR 3.39，RankICIR 4.63

---

## 1. 因子定义

```
对每只股票、每个交易日 t：
  当日相邻量峰之间的分钟间隔 = sorted(peak_minute_indices) 的 np.diff
  
  pooled 跨 20 日的所有间隔：X = concat([当日间隔 for 每日 ∈ t-20..t-1])
  
  因子 = excess_kurtosis(X) = E[(X-μ)⁴] / σ⁴ - 3
  方向 +1（峰度高 → 间隔分布尖瘦 → 量峰簇集明显）
```

**实现思路**：算子对每一天产出 5 阶矩 (n=Σ1, m1=Σ, m2=Σ², m3=Σ³, m4=Σ⁴)，
spec 后续做 20 日 rolling sum 各阶矩，再代数还原 mean / var / 中心化 4 阶矩 / kurt。
这样不必在算子里持有"当日间隔列表"这种变长结构，全程矢量化。

---

## 2. 实现 spec（11 步：5 阶矩聚合 + 5 个 rolling sum + 4 个 compute）

```yaml
- minute_intraday_aggregate features=[peak_interval_n, ..., peak_interval_m4]   # 共享底座
- rolling sum × 5    # pi_n_20, pi_m1_20, pi_m2_20, pi_m3_20, pi_m4_20
- compute pi_mean    = pi_m1_20 / pi_n_20
- compute pi_var     = pi_m2_20 / pi_n_20 - pi_mean²
- compute kurt_raw   = (Nμ4) / (Nσ⁴) - 3  ← 见数学推导
- compute kurt_gate  = (pi_n_20 >= 5) * (pi_var > 1e-12)
- compute peak_interval_kurt = kurt_raw * (gate / gate)   # divide-by-mask 把 gate=False 变 NaN
```

完整 yaml: `specs/peak_interval_kurt/spec.yaml`。

---

## 3. ⭐ 数学：从 5 阶矩到 excess kurtosis

设 X 是 N 个间隔，μ = m1/N，σ² = m2/N - μ²。

中心化 4 阶矩：
```
μ₄ = E[(X-μ)⁴] = m4/N - 4μ·m3/N + 6μ²·m2/N - 3μ⁴
```

两边乘 N：
```
N·μ₄ = m4 - 4μ·m3 + 6μ²·m2 - 3N·μ⁴
```

excess kurtosis = μ₄/σ⁴ - 3 = (N·μ₄)/(N·σ⁴) - 3。最后形式让分子分母都"有 N"，
对应 spec 里的 compute 公式：

```python
(pi_m4_20 - 4*pi_mean*pi_m3_20 + 6*pi_mean**2*pi_m2_20 - 3*pi_n_20*pi_mean**4) \
    / (pi_n_20 * pi_var**2) - 3
```

**Plan agent 一度误判分子要除一次 N**（说我的公式错）。其实**分子已经是 N·μ₄、分母是 N·σ⁴，N 自然约掉**。我在 spec 注释里把数学推导写齐，避免后续误读。

---

## 4. ⭐ pd.eval 的 `where()` 不可用 → divide-by-mask 替代

`compute.py` 用 `pd.eval`，**它不支持 `where(cond, a, b)` 函数式调用**——尽管
`_BUILTINS` 里挂着 "where"（那是给标识符校验用的，不是 eval 时实际可调用）。

替代惯用法：
```yaml
formula: kurt_raw * (gate / gate)   # gate=0 → 0/0=NaN；gate=1 → 1/1=1
```

把"样本不足 / var≈0"位置变 NaN，避免被算入排序后污染分组。**实现时踩到这个坑**，
log 报错 `"where" is not a supported function`。

---

## 5. 复现指标 vs 论文

| 指标 | 我（5 分组 × 5 日）| 论文（10 分组 × 月频）|
|---|---|---|
| LongShort 年化 | +16.02% | +23.30% |
| Sharpe / IR | 2.26 | 3.39 |
| IC 5d | +2.78% / ICIR 0.345 | RankIC 7.19% / ICIR 4.63 |
| 单调性 | +0.910 | / |

**收益量级正确（同符号、同数量级）**，但比论文低 ~30%。

---

## 6. 与论文差距分析（按可能性排序）

### (1) 分组数 + 调仓频率（**最可能贡献 50%+ 差距**）
- 论文：10 分组、约 21 日调仓
- 我：5 分组、5 日调仓
- 10 分组 G1/G10 是更极端的"长尾"，多空年化天然更大
- 月频换手率低 → 持仓更稳 → IR 高

### (2) 中性化（论文做了 Barra 风格 + 行业中性化）
表 16 显示 peak_interval_kurt 与流动性 -35%、波动率 -25% 相关；中性化后 LongShort
应小幅下降但 IC 系数差异显著。**我没做中性化** → IC 可能低估。

### (3) 间隔定义边界差异
"两个量峰之间的间隔"——我严格用 `np.diff(sorted(peak_minutes_in_day))`。
论文若处理午休边界（11:30 和 13:01 之间是否算"相邻"）方式不同，会有微小差异。
我的 minute_of_day 编码 `(h*60+m) - 9*60-30` 自然让午休是个"大 gap"，与正常间隔混合
进入分布。

### (4) "pooled 20 日" vs "每日峰度均值"两种解读
论文文字"过去 20 日同日前后两个量峰之间的时间间隔分布的峰度"——我解读为
**pool 20 日所有间隔 → 一个 kurt**。另一种解读是**每日先算 kurt → 20 日均值**。
两种数值不同（pool 法用更多样本 → kurt 估计更稳）。我选了 pool 法因为表达更紧凑。

---

## 7. 工程经验沉淀

### 7.1 高阶矩 → kurt 的"5 阶矩 + rolling sum + 代数"模式

是这个算子族的通用范式。任何"pooled N 日 X 的 kurt/skew/var/mean"都能通过
"per-day 输出 raw moments → rolling sum → compute 还原"实现，**不需要新算子**。

类比：单因子 npf_mrq_sue8 用的"望远镜求和 + 方差恒等式"也是同一思路——把
"显式持有 N 个值"换成"持有 N 个标量阶矩"。

### 7.2 守门保护放在 spec 末尾，不是算子内

算子产出 raw moments，n、var 都暴露给下游。**守门（n>=5、var>1e-12）放在
spec 末尾的 compute step**，而不是在 minute_intraday_aggregate 里偷偷做。
原因：不同因子可能有不同 robust 准则（kurt 要 n>=5，skew 要 n>=3，corr
要 n>=10），都塞进算子会让算子参数化失控。

### 7.3 cache 命中后 step#1 = 8 秒（vs 首次 62 秒）

5455 只股票全部 cache hit + concat + slice **只用 8 秒**。peak_minute_count 跑过
之后，peak_interval_kurt 完全免费。论文剩 18 个因子都将享用这次缓存（除非
bump `_FEATURES_SUPERSET_VERSION` 触发重建）。

---

## 8. 文件链接

- spec: `specs/peak_interval_kurt/spec.yaml`
- raw 因子: `RAW_FACTOR_DIR/peak_interval_kurt.parquet`（2528×5363, 969 万非空）
- cleaned: `CLEANED_FACTOR_DIR/peak_interval_kurt.parquet`
- 评估图: `output/peak_interval_kurt/evaluation_20150101_20250531.png`
- 算子: `core/operators/minute_intraday_aggregate.py`（共享，与 peak_minute_count 同源）
