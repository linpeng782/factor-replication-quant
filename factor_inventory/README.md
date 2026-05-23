# Factor Inventory

所有 panel 因子在统一评估配置下的"总账"——一张 parquet 表 × 一组 PNG，让你一眼看清"我手上有什么因子，哪些好"。

## 用途场景

| 场景 | 怎么用 |
|---|---|
| **LGBM 特征选择** | `df.query("icir_5d.abs() > 0.4").index.tolist()` 一行代码拿候选因子 |
| **找冗余因子** | 对 inventory 取 top-N 后做 cross-section corr matrix |
| **复现进度追踪** | `df.groupby('producer').size()` 看每个 producer 当前几个因子 |
| **质量监控** | 周/月跑一次，diff 上一份 inventory 看 IC 突变 |
| **写报告 / 论文** | `df.to_markdown()` / `df.to_latex()` 直接出表格 |

## 目录结构

```
factor_inventory/
├── README.md                           本文件
├── 20260523_224933/                    每次 refresh 一个时间戳目录（保留历史）
│   ├── inventory.parquet               ⭐ 197 行 × 36 列核心表（进 git）
│   ├── inventory.csv                   human-readable 镜像（进 git）
│   ├── run_meta.json                   本次 refresh 的元信息
│   └── plots/
│       ├── alpha158/                   158 张 PNG（不进 git）
│       └── spec/                       39 张 PNG（不进 git）
└── latest -> 20260523_224933/          软链：方便下游脚本写死 latest 路径
```

**约定**：
- 每次 `python scripts/build_factor_inventory.py` 新建一个时间戳目录，旧目录保留
- `latest` 自动指向最新——下游永远 `pd.read_parquet("factor_inventory/latest/inventory.parquet")`
- `inventory.parquet` 进 git（小，~50KB）；`plots/*.png` 不进 git（体积大，可重生成）

## inventory.parquet schema (36 列)

| 类别 | 列 |
|---|---|
| **身份** | `producer` (alpha158/spec), `direction` (±1) |
| **IC 指标 × 4 horizons** | `ic_mean_{2,5,10,20}d`, `ic_std_{2,5,10,20}d`, `icir_{2,5,10,20}d`, `ic_t_{2,5,10,20}d`, `pct_positive_{2,5,10,20}d`, `n_days_{2,5,10,20}d` |
| **分层指标** | `monotonicity`, `ann_return_G1`, `ann_return_G5`, `ann_return_LongShort`, `ann_vol_LongShort`, `sharpe_LongShort`, `mean_turnover_LongShort` |
| **元数据** | `time_window_start`, `time_window_end`, `eval_timestamp` |

Index 是 `factor`（因子名）。

## 使用示例

```python
import pandas as pd

inv = pd.read_parquet("factor_inventory/latest/inventory.parquet")

# 1. 总览
print(inv.shape)                                  # (197, 36)
print(inv.groupby("producer").size())              # alpha158 158, spec 39

# 2. Top 30 by 5d ICIR (绝对值)
top30 = inv.reindex(inv.icir_5d.abs().sort_values(ascending=False).index).head(30)
print(top30[["producer", "direction", "icir_5d", "icir_20d", "monotonicity", "sharpe_LongShort"]])

# 3. 强因子 + 高单调性
robust = inv[(inv.icir_5d.abs() > 0.5) & (inv.monotonicity.abs() > 0.85)]

# 4. 仅 spec producer 的 top 10
spec_top = inv[inv.producer == "spec"].nlargest(10, "icir_5d", keep="all")

# 5. 准备 LGBM 特征清单
feature_list = inv.query("icir_5d.abs() > 0.3").index.tolist()
```

## 评估配置（写在脚本里）

| 项 | 值 |
|---|---|
| 时间窗 | `2016-01-01 ~ 2025-12-31`（10 年） |
| IC horizons | `(2, 5, 10, 20)` |
| Primary horizon | 5 |
| Layer groups | 5 |
| Layer rebalance | 5 |
| Producers | `alpha158, spec`（**跳过 mars**：用户判断质量较低） |
| N workers | 100（128C 机器） |
| forward_returns 源 | `my-alpha-engine/labels/forward_return_*d.parquet`（PIT canonical） |

改配置：编辑 `scripts/build_factor_inventory.py` 顶部参数区。

## Refresh 流程

```bash
source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate
python scripts/build_factor_inventory.py     # ~2 分钟（100 workers，197 因子）
```

完事会自动：
1. 在 `factor_inventory/<时间戳>/` 落 inventory.parquet + plots
2. 把 `latest` 软链指向新时间戳目录

## 常见问题

**Q: `inventory.parquet` 跟 `evaluation-reports/batch_*/ic_summary.csv` 是不是重复？**

A: 不是。alpha-engine 的 `run_evaluation.py` 输出 batch 报告（含两个分散 CSV）；inventory 是合并 + 加 producer 元信息后的**单一查询入口**，且按时间戳归档支持 diff。`run_evaluation.py` 仍然可用做"探索性批量评估"。

**Q: 为什么跳过 mars？**

A: 用户判断 mars 因子集质量较低，暂不纳入 LGBM 训练池。需要纳入时把 `PRODUCERS` 改成 `("alpha158", "mars", "spec")` 即可。

**Q: 历史时间戳目录会不会越积越多？**

A: 每次 refresh 都新建。建议每月清一次老的，只保留最近 3-6 份。`inventory.parquet` 进 git 留档，老的 plots 删了不可惜（脚本能重生成）。
