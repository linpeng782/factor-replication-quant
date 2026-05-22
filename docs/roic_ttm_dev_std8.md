# roic_ttm_dev_std8 — ROIC_TTM 稳定性因子

> **状态**：v1 全市场跑通，**有效**
> **复现 IC/ICIR**：5d 0.018/**0.222**, 10d 0.022/**0.249**, 20d 0.023/**0.268**
> **LongShort Sharpe** +0.760，**单调性** +0.720，G1/G5 跨度 6.3pp
> **基准**：PROGRESS.md 中无 roic_ttm 系列基准 IC/ICIR，本因子作为新建立基线

---

## 1. 因子定义

```
roic_ttm_dev_std8 = 1 / std8(ROIC_TTM)
```

- **ROIC_TTM** = 米筐 `return_on_invested_capital_ttm`（已预算）
- **std8** = 过去 **8 个财报期** ROIC_TTM 的样本标准差
- **方向 +1**：std 越小（公司投入资本回报越稳定）→ 因子值越大 → 预期收益越高

经济直觉：**质量稳定性溢价**。在 ROIC 水平相近的股票里，长期波动小的公司说明经营管理 / 商业模式稳定，市场愿意给溢价。

---

## 2. 研报字面歧义 + 工程选择

PROGRESS.md 原文："1 ROIC_TTM / 过去8期ROIC_TTM的标准差" —— 字面有歧义：

| # | 歧义点 | 工程选择 | 理由 |
|---|---|---|---|
| 1 | "1 ROIC_TTM"是什么 | **`1 / std8`** | 用户决策；忠于"稳定性因子"字面 |
| 2 | "8 期"语义 | **8 个财报期**（变化日 rolling） | 504 交易日 rolling 在不规则财报间隔下会丢早期数据 |
| 3 | min_periods | **4** | 与新股 mask 252 天 ≈ 4 期财报对齐 |

**未来 v2 候选**：`ROIC_TTM / std8`（信噪比形式，类 Sharpe），高 ROIC + 低波动才高分，比纯稳定性更有经济解释力。

---

## 3. 计算链路（3 步）

```yaml
1. fetch:    return_on_invested_capital_ttm  (米筐 get_factor)
2. rolling:  agg=std, window=8, min_periods=4
             change_on=return_on_invested_capital_ttm   ← 仅在值变化日采样
             fill_method=ffill                          ← 变化日间沿用最近值
             → roic_ttm_std8
3. compute:  roic_ttm_dev_std8 = 1 / roic_ttm_std8
```

变化日 rolling 是关键算子：每只股票自动识别自己的财报发布日（ROIC_TTM 值变化的日子），在那些日子上做 8 期窗口 std，再 ffill 到所有交易日。

---

## 4. 复现结果（2016-01-04 ~ 2025-12-31，全市场 5549 只）

```
覆盖：(2430, 5549)，非空 8.8M / 13.5M = 65%（剔除银行+新股）

IC (Spearman):
  5d:  ic_mean=+0.018,  icir=+0.222,  positive_pct=61.5%,  t=10.7
  10d: ic_mean=+0.022,  icir=+0.249,  positive_pct=63.4%,  t=12.1
  20d: ic_mean=+0.023,  icir=+0.268,  positive_pct=64.3%,  t=13.0

分层 (5 组 × 5 日调仓):
  G1: ann=-1.38%   sharpe=-0.06   (low std → unstable)
  G2: ann=+5.14%   sharpe=+0.24
  G3: ann=+3.57%   sharpe=+0.17
  G4: ann=+5.51%   sharpe=+0.27
  G5: ann=+4.93%   sharpe=+0.26   (high std → stable)
  LongShort: ann=+5.53%, vol=7.28%, Sharpe=+0.760, turnover=6.5%
  monotonicity: +0.720
```

**有效因子的全部特征都齐**：ICIR > 0.2、单调性 > 0.7、LongShort Sharpe > 0.7、t-stat 双位数。在 PROGRESS.md 已复现的 19 个因子里能进前 5。

---

## 5. ⭐ 关键事件：partial-data 静默 bug

**第一次跑出 (2430, 1500) 只 1500 只股票**——以为是数据稀疏，实际是 fetcher 静默吞掉了 8/12 失败批次（米筐 "connection number exceeds"）。

诊断过程：
- 缺失股票 `000001~000990` + `600486~990018` 两段连续区间 → 不是稀疏，是批次丢失
- 重看日志：8 批 0.2~2.5s 内被拒绝，3 批最终成功，1 批 84s 超时
- fetcher 只 `logger.warning + continue`，partial concat 落盘
- 巧合的是 partial 上 ICIR 0.226 ≈ full 0.222——所以**信噪比相似但 LongShort Sharpe 0.46 vs 0.76**，因为 Long 端 G5 大票被截掉了

**修复**（应用到 `core/yolo_engine.py`）：
1. `_thread_get_factor` 加重试，瞬时错误最多 3 次，指数 backoff + jitter
2. `_fetch_factor_parallel/sequential` 任一批最终失败就 `RuntimeError`，永不接受 partial data
3. 日志统一 `batch#{idx}` 命名，不再混淆 done/batch_idx

**教训**：ICIR 看着合理时也要核对 raw_factor 的 shape 是否符合 universe 大小，partial-data 在边缘信号上可能完全等价于完整数据，但 LongShort 表现会差。

---

## 6. 与三因子组合内对比

| 因子 | 5d ICIR | LongShort Sharpe | 单调性 | 评价 |
|---|---|---|---|---|
| `roic_ttm_ind_rnk8` | +0.044 | -0.30 | -0.37 | 失效，行业内排名尺度+8 期 min 失效双重打击 |
| `roic_ttm_all_rnk8` | +0.006 | +0.36 | +0.89 | 噪声，单调性是直方图人造的，IC 无信号 |
| **`roic_ttm_dev_std8`** | **+0.222** | **+0.760** | **+0.720** | **唯一有效**，稳定性是真信号 |

为什么稳定性 work 而排名不 work？rank min8 把 ROIC 折成排名再取 8 期最小值，把绝对值信息和波动信息**双重压缩**到一个排名标量上；而 dev_std8 直接保留 ROIC 的二阶矩，没有信息损失。

---

## 7. v2 / v3 改进路径

| 优先级 | 方向 | 预期收益 |
|---|---|---|
| P1 | `ROIC_TTM / std8`（SNR 形式）| 加入水平信息，可能 ICIR +20~30% |
| P1 | 中证 800 + 总市值前 100 池子 | 复现研报真实底池，去除小盘噪声 |
| P2 | window 改 12 / 16 期 | 降噪但牺牲覆盖率 |
| P3 | log(1/std)：减弱右尾极端值 | std 接近 0 时因子爆炸的风险控制 |

未来若做 SNR 形式（`ROIC/std`），需要新建一个 spec 文件（不修改 dev_std8 现有产物）。
