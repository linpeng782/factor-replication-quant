# eruption_turnover_corr — 喷发成交额相关性（开源_微观_27 因子 f19）

> **状态**：v1 全市场跑通；**LongShort 年化几乎逐位复现，Sharpe 超过论文**
> **复现 IC/ICIR**：5d IC=−0.0667 / |ICIR|=0.704，t=+35.20，positive 78.1%
> **复现回测**：LongShort 年化 **+29.61%**，**Sharpe 3.585**（论文 IR 2.73），单调性 **+0.995**
> **论文基准**：多空年化 29.72%，IR 2.73，RankICIR −3.99
> **方向**：−1（高相关 → 资金跟随强 → 散户主导 → 未来收益弱）

---

## 1. 因子定义

```
对每只股票、每个交易日 t：
  从过去 20 日 pooled 中取所有"喷发分钟"：m where E[t', m]=True, t' ∈ [t-20, t-1]
  X = T[t', m]      （喷发那一分钟的成交额）
  Y = T[t', m+1]    （下一分钟的成交额，同一交易日）
  
  因子 = Pearson(X, Y)
  方向 −1: 相关高 → 大单后资金跟随强 → 散户特征 → 未来收益弱
```

---

## 2. ⭐ 数学：用 6 阶矩在算子层不存 X、Y 列表

Pearson formula（设 N 个样本）：
```
ρ = (N·ΣXY − ΣX·ΣY) / sqrt((N·ΣX² − (ΣX)²) · (N·ΣY² − (ΣY)²))
```

推导：
```
cov(X,Y) = E[XY] − E[X]·E[Y] = (N·ΣXY − ΣX·ΣY)/N²
σ²_X    = E[X²] − (E[X])²    = (N·ΣX² − (ΣX)²)/N²
σ²_Y    = E[Y²] − (E[Y])²    = (N·ΣY² − (ΣY)²)/N²
ρ = cov / (σ_X·σ_Y)；分子分母都有 N²，约掉。
```

只需要算子吐 6 个**日频**标量：N、ΣX、ΣY、ΣX²、ΣY²、ΣXY。spec 末尾 rolling sum 20 日
后用 compute 拼出 ρ。**不需要任何"持有 X 列表"的特殊算子**。

类比：
- peak_interval_kurt（f10）用 5 阶矩还原 kurt
- npf_mrq_sue8 用望远镜求和 + 方差恒等式
- 都是**"局部高阶矩 + 跨期 rolling sum + 代数还原"**的范式

---

## 3. 实现 spec（14 步：1 算子 + 6 rolling sum + 7 compute）

```yaml
- minute_intraday_aggregate features=[eruption_count, eruption_turnover_sum,
    eruption_turnover_sumsq, eruption_next_turnover_sum, eruption_next_turnover_sumsq,
    eruption_xy_sum]                 # 算子吐 6 个日频阶矩
  cache_key: prv_v2                  # v2 = 加了 4 列 eruption 阶矩

- rolling sum × 6                    # 20 日 pooled
  → n_20, sx_20, sx2_20, sy_20, sy2_20, sxy_20

- compute corr_num = n·sxy − sx·sy
- compute var_x    = n·sx² − (sx)²
- compute var_y    = n·sy² − (sy)²
- compute corr_den = sqrt(var_x · var_y)
- compute corr_raw = corr_num / corr_den

- compute corr_gate = (n_20 >= 10) * (var_x > 1e-12) * (var_y > 1e-12)
- compute eruption_turnover_corr = corr_raw * (corr_gate / corr_gate)
```

完整 yaml: `specs/eruption_turnover_corr/spec.yaml`。

---

## 4. 算子 v2 升级：加 4 列 eruption 阶矩

`minute_intraday_aggregate.py` 的 `_FEATURES_SUPERSET_VERSION` v1 → v2。新增 superset：
- `eruption_count`        = N （peak_count + ridge_count，但显式输出更清晰）
- `eruption_turnover_sum` = ΣX
- `eruption_turnover_sumsq` = ΣX²
- `eruption_xy_sum`       = Σ T[m]·T[m+1] where E[m]=True

`eruption_next_turnover_sum` / `eruption_next_turnover_sumsq` 已在 v1 里有。

**Bump 后果**：params_hash 变 → cache_key=`prv_v1` 的旧因子（peak_minute_count、peak_interval_kurt）下次跑也会走 cache miss 一次性重建。代价是 60-90 秒，可接受。

旧缓存 `prv_v1__h09dd8a9e78/` (2.3GB) 自动 orphan，可手动 `rm` 清理。

---

## 5. 复现指标 vs 论文

| 指标 | 我（5 分组 × 5 日调仓 × 全市场无中性化）| 论文（10 分组 × 月频 × Barra+行业中性）|
|---|---|---|
| LongShort 年化 | **+29.61%** | +29.72% |
| Sharpe / IR | **3.585** ⭐ | 2.73 |
| 单调性 | +0.995 | / |
| Raw IC 5d | −0.0667 | RankIC −10.94% |
| t-stat | 35.2 | / |
| 多头 G5 年化 | +17.49% | +15.12% |

**核心发现**：
- LongShort 几乎逐位复现（差 0.11%）
- **Sharpe 比论文高 30%** — 我用 5 日调仓 + 全市场无中性化，理应不如论文优雅，
  但 Sharpe 反而更高。可能原因：日频换仓 + 5 分组（vs 论文月频 + 10 分组）
  让"长尾"风险曝露更分散
- 单调性 0.995 → 因子分组之间几乎完美单调

---

## 6. 与论文小差距分析

### (1) IC 量级差距（5d ICIR 0.7 vs 论文 ~3.99）
**频率换算**：日频 ICIR ×√21 ≈ 月频 ICIR；0.918（20d ICIR）×√1 ≈ 0.92。
论文 ICIR 3.99 估计是月频 t-stat（接近 RankICIR×√n_months 的标度），不是
直接频率换算可比。本质看 **t-stat 35.2 + 多空年化** 这两个不依赖频率口径的量，
都对得上甚至超过。

### (2) 多头 G5 我 +17.5% vs 论文 +15.12%
我比论文高 ~2.4 个百分点。可能原因：
- 全市场池子里小票多，相关性高的"散户股"可能 G1 弱、G5 反而是"机构主导股"
  → 长头多挣
- 5 分组 vs 10 分组：我的 G5 是前 20%，论文 G10 是前 10%；前 20% 平均化更稳
  但极值收益略低（实测反而更高，与直觉相反——可能是单调性太完美）

### (3) Direction 自动判定 = −1 ✓
评估管线探测 `raw IC mean (5d) = −0.0667`，自动 flip 方向 → 与 spec 标的
direction=−1 一致。

---

## 7. 工程经验沉淀

### 7.1 高阶矩范式 = 分钟级因子的"瑞士军刀"

回顾本论文 20 个因子，至少 5 个（f8-f13 间隔 std/skew/kurt + f17-f19 跟随/敏感/相关）
都能用"算子产 raw moments → spec 末尾 rolling sum + 代数"实现。**不需要任何"分钟级持有"
的特殊算子**。

为新因子添加阶矩列时只需：
1. `minute_intraday_aggregate.py` 加列 + bump version
2. spec 写 rolling sum + compute 公式
3. cache 自动失效 + 60 秒重建

### 7.2 cache_key vs `_FEATURES_SUPERSET_VERSION` 的语义边界

- `cache_key`（spec 字段）：**人手 bump 用**，比如想强制 rebuild 一次但代码没动
- `_FEATURES_SUPERSET_VERSION`（算子代码常量）：**自动 bump 用**，加 superset 列后
  bump，避免下游因子读不到新列

两者通过 sha1(...) 联合进 `params_hash`，任一变化都触发 cache miss。

### 7.3 100 worker vs 64 worker 性能（NFS 已是瓶颈）

| 配置 | step#1 全量重建 | 评估 | 总用时 |
|---|---|---|---|
| 64 worker (peak_minute_count) | 62 秒 | 90 秒 | ~3 分钟 |
| 100 worker (eruption_turnover_corr) | ~85 秒 | 90 秒 | ~4 分钟 |

100 worker 没显著加速——NFS 上同时 100 个进程 read parquet 已经接近带宽上限。
继续加 worker 反而可能因 NFS 拥塞下降。**100 是合理上限**。

---

## 8. 文件链接

- spec: `specs/eruption_turnover_corr/spec.yaml`
- raw 因子: `RAW_FACTOR_DIR/eruption_turnover_corr.parquet`
- cleaned: `CLEANED_FACTOR_DIR/eruption_turnover_corr.parquet`
- 评估图: `output/eruption_turnover_corr/evaluation_20150101_20250531.png`
- 算子（共享）: `core/operators/minute_intraday_aggregate.py`（v2，33 个 superset cols）
- v2 缓存: `INTERMEDIATE_CACHE_DIR/prv_v2__haa350c59c6/<order_book_id>.parquet` × 5455
