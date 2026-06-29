# Alpha158 数据源迁移 rq → dquant — 完整设计

> 状态：**方案稿**（待 §9 待确认项拍板后转执行）。
> 目标：把 alpha158 的两层输入数据（L0 复权因子 / L1 日频原始 OHLCV）从 **rqdatac** 切换到 **dquant**，
> 价格走 `source="rq"`（与现有米筐数据对齐），后复权因子从 rq `ex_factor` 换为 **jy `adj_factor`**。
> 设计准则：**最小改造面、A/B 可回退、对齐解耦**——只换数据源，下游计算/增量/对账一行不改。
> 关联：`alpha158_incremental_design.md`（日更铁律，本文复用）/ `minute_incremental_design.md`。

---

## 1. 核心洞察（决定方案的简洁性）

现有 alpha158 是**严格三层数据流**：

```
L0  daily/stock-ex-factors/<股>.parquet   逐股稀疏 ex_cum_factor      (data_fetching/ex_factors.py, rq)
L1  daily/stock-ohlcv/<股>.parquet        逐股原始不复权 OHLCV        (data_fetching/raw_ohlcv.py, rq)
        │ 读时复权：价×ffill(cum_factor)，量÷cum_factor   (core/producers/alpha158/adjusted_panels.py)
        ▼
L3  factors/raw/alpha158/<group>/<factor>.parquet   WIDE date×stock, 158 因子
        (build_alpha158.py 全量 / alpha158_daily_update.py 增量)
```

**L3 计算、build 脚本、增量算法都只依赖 L1/L0 的「逐股 parquet 契约」，与数据是 rq 还是 dquant 无关。**

> **结论**：迁移 = 只换 L0、L1 的**取数数据源**，保持逐股 parquet layout 不变
> → **L3 计算、增量算法（`alpha158_daily_update.py`）、对账脚本（`smoke_alpha158_truncate_replay.py`）100% 不动**。
> 这是最小、最安全的改造面。

---

## 2. 三个难点 × 对策

| 难点 | rq 现状 | dquant 约束 | 对策 |
|---|---|---|---|
| **股票池** | `all_instruments(type="CS")` 一次拿**全历史**（含退市） | `all_instruments(None,D,D)` 只给**当日**在册池 | 按日取当日池 → 逐股 append（新股自动建文件，与现有 `raw_ohlcv.py` 的 groupby-append 同构）；全量回填的历史并集策略见 §9 待确认 |
| **存储结构** | rq 直接逐股落盘 | dquant 自然**按天**返回长表 | 取数按天 → 写盘 `groupby(order_book_id)` 落**逐股**（维持 `alpha158_incremental_design.md` 决策#1「输入维持按股」契约），**不**引入持久化长表 |
| **复权源** | rq `get_ex_factor` → `ex_cum_factor` | — | 换 jy `get_adj_factor(source="jy")`（累计 `adjfactor`，ricequant 兜底），物化成 L0 兼容格式（价×adjfactor 语义一致） |

---

## 3. 目标数据流（仅 L0/L1 数据源变化，L3 不变）

```
L0  daily/stock-ex-factors-jy/<股>.parquet   逐股 adjfactor       ← dquant get_adj_factor(source="jy")
L1  daily/stock-ohlcv-dquant/<股>.parquet    逐股原始不复权 OHLCV  ← dquant get_price(source="rq", 1d)
        │ 读时复权（adjusted_panels.py 逻辑不变，仅列名兼容）
        ▼
L3  factors/raw/alpha158/...                 build_alpha158 / alpha158_daily_update 完全不变
```

**新目录独立**（不覆盖现有 rq 目录）→ 可 A/B 并行对账、随时回退。

### dquant 取数口径（来自 stock-data-fetching/core/dquant_daily.py 已验证用法）

- 日频行情：`get_price(order_book_ids=codes, start_date, end_date, frequency="1d", source="rq")` → 列含 `open/high/low/close/volume/amount`（`amount`→重命名 `total_turnover`）
- 复权因子：`get_adj_factor(order_book_ids=codes, trade_date=D, source="jy")` 主，`source="ricequant"` 兜底（覆盖互补）；列 `adjfactor`
- 涨跌停（如需）：`get_limit(codes, D, D)` → `limit_up/limit_down`
- 股票池：`all_instruments(None, D, D)` 筛 `type=="CS"`

---

## 4. 详细任务表（主交付）

| # | 模块 | 改动 | dquant API / 输入 | 输出 layout | 口径/格式差异 | 对齐校验 |
|---|---|---|---|---|---|---|
| **T0** | 公共取数 helper | 新建 `data_fetching/dquant_source.py` | `all_instruments(None,D,D)` 筛 CS；交易日历 | — | 当日池；回溯改交易日切片 | 与 rq 池逐日比对股票数 |
| **T1** | L1 日频 OHLCV | 新建 `raw_ohlcv_dquant.py`（仿现有 `raw_ohlcv.py` 结构） | `get_price(codes,s,e,"1d",source="rq")` | `daily/stock-ohlcv-dquant/<股>.parquet`，index=date，列 open/high/low/close/volume/total_turnover | `amount`→`total_turnover`；长表 groupby 落逐股；不复权 | **逐值比对** dquant raw vs 现有 rq L1（容差 1e-6，应 bit 级） |
| **T2** | L0 复权因子 | 新建 `ex_factors_jy.py` | `get_adj_factor(codes,trade_date=D,source="jy")`（兜底 ricequant） | `daily/stock-ex-factors-jy/<股>.parquet`，稀疏 `ex_cum_factor`(=adjfactor) | jy 累计因子，基准归一可能异于 rq；列名对齐 loader | jy vs rq 累计因子比值在除权日附近的差异分布 |
| **T3** | loader 适配 | `adjusted_panels.py` 读 L0 列名兼容（`adjfactor`/`ex_cum_factor`） | 读 T1+T2 | 后复权宽表（内存） | 仅列名兼容，复权公式不变 | 单股后复权序列 dquant vs rq |
| **T4** | 配置开关 | `core/config.py` 增 L1/L0 的 dquant 变体路径（env 切换） | — | — | 默认指 rq，env 切 dquant，可回退 | — |
| **T5** | alpha158 产出对齐 | **不改** `build_alpha158.py` | 跑 dquant 源 | 落独立对比目录 | — | **158 因子 dquant vs rq 逐格 `max_rel`，NaN 模式一致** |

---

## 5. 对齐策略（关键：解耦两个变量）

同时换「数据源 rq→dquant」和「复权源 rq→jy」会让两个变量耦合，差异难定位。**必须分两步**：

| 步 | 配置 | 期望结果 | 验证目的 |
|---|---|---|---|
| **Step 1** 数据源正确性 | dquant-rq L1 **+ 原 rq L0**（ex_cum_factor 不变） | alpha158 vs 现有 = float-epsilon（`max_rel < 1e-6`） | 证明 rq→dquant **取价无损**（都是米筐原始价） |
| **Step 2** 复权政策切换 | dquant-rq L1 **+ jy L0** | 除权日附近**可控偏差**，其余吻合 | 这是要的新标准（jy 复权），偏差是**预期**非 bug，需量化记录 |

> **为何 alpha158 对复权基准不敏感**：158 因子全是**比率 / 滚动统计**（尺度不变量）。
> 复权因子整体乘一个常数（基准归一差异）**不改变因子值**；唯一可能差异来自 jy 与 rq
> 在**除息当日的累计因子比值**不同（现金分红处理口径差异）。Step 2 专门量化这一处。

---

## 6. 增量更新设计（直接复用现有铁律）

`alpha158_incremental_design.md` 的增量算法（warmup=65 交易日、append+dedup+原子写、新股 concat 列并集）
**完全不依赖数据源**，因此：

- **L1/L0 dquant 日更**：每交易日 `get_cs_codes(D)` + `get_price(codes,D,D)` / `get_adj_factor(trade_date=D)`
  → 逐股 append + dedup + 原子写（与现有 `raw_ohlcv.py` 同构，新股自动建文件，退市股自然停更）。
- **L3 日更**：`alpha158_daily_update.py` **一行不改**，只是上游读 dquant 目录。
- **编排**：`daily_update.sh` 把数据线 1–2（ex_factors / raw_ohlcv）换成 dquant 版步骤。
- **验收**：现有 `smoke_alpha158_truncate_replay.py`（截断重放对账）直接复用，验证 dquant 日更 == dquant 全量。

---

## 7. 分期实施 & 验收判据

| 阶段 | 内容 | 验收判据 |
|---|---|---|
| **P1** | T0+T1：dquant 全量回填 L1（逐股） | dquant L1 vs rq L1 逐值 `max|Δ| < 1e-6` |
| **P2** | T3+T4+Step1：原 rq 复权下跑 alpha158 | 158 因子 vs 现有 `max_rel < 1e-6` + NaN 模式一致 |
| **P3** | T2+Step2：换 jy 复权重跑 | 偏差量化报告（除权日附近），确认可接受 |
| **P4** | 增量：L1/L0 dquant 日更 + truncate-replay | 日更 == 全量（活区 rel<1e-6 + IPO 吻合） |
| **P5** | 切 `daily_update.sh` 编排，稳态日更 | reconcile IC 复现 |

---

## 8. 关键文件索引（现状，改造时参照）

| 角色 | 路径 |
|---|---|
| 配置（路径/字段/区间） | `core/config.py`（`RAW_OHLCV_DIR` / `EX_FACTORS_DIR` / `PRICE_FIELDS` / `VOLUME_FIELDS`） |
| L1 取数（rq，待仿写 dquant 版） | `data_fetching/raw_ohlcv.py` |
| L0 取数（rq，待换 jy 版） | `data_fetching/ex_factors.py` |
| 读时后复权 | `core/producers/alpha158/adjusted_panels.py` |
| L3 全量 / 增量 | `scripts/build_alpha158.py` / `scripts/alpha158_daily_update.py` |
| 对账（可复用） | `scripts/smoke_alpha158_truncate_replay.py` / `scripts/verify_alpha158_reproduction.py` |
| dquant 单日取数范例 | `../stock-data-fetching/core/dquant_daily.py` |

---

## 9. 待确认项（开工前拍板）

1. **A/B 目录**：dquant 落独立新目录（`stock-ohlcv-dquant` / `stock-ex-factors-jy`），不覆盖 rq，验证通过后再切配置。（倾向：是）
2. **全量回填粒度**：「逐日」（每日取当日池+get_price，与增量路径语义完全一致）vs「年批量」（用历史股池并集一次拉一年区间，更快、调用少）。（待拍板）
3. **退市股历史来源**：dquant `rq_price` 是否含退市股历史？
   - 含 → 股池取「逐日 `all_instruments` 并集」；
   - 不含 → 直接复用现有 L1 文件名（rq 时代全历史名单，~5508 只）作回填股池。（需先验证）
4. **limit_up/down 是否一并取**：alpha158 的 158 因子**不用**涨跌停（仅 OHLCV+vwap），可暂不取以简化 T1；
   但现有 rq L1 存了 limit 供其他消费端。确认 dquant L1 是否补 `get_limit` 保持字段齐全。（待拍板）

---

## 10. 决策记录（随推进补充）

| # | 决策 | 取向 | 状态 |
|---|---|---|---|
| 1 | 改造面 | 只换 L0/L1 数据源，下游不动 | 已定 |
| 2 | 存储结构 | 维持逐股 parquet（不建长表） | 已定 |
| 3 | 价格源 | dquant `source="rq"`，目标 bit 级对齐现有 rq L1 | 已定 |
| 4 | 复权源 | rq `ex_factor` → jy `adj_factor`（ricequant 兜底） | 已定 |
| 5 | 对齐 | 解耦两步（先验数据源、再验复权政策） | 已定 |
| 6 | A/B 目录 | 独立新目录 + config 开关，可回退 | 待确认(§9.1) |
| 7 | 回填粒度 | 逐日 / 年批量 | 待确认(§9.2) |
| 8 | 退市股股池 | 逐日并集 / 复用现有文件名 | 待确认(§9.3) |
| 9 | limit 字段 | 取 / 不取 | 待确认(§9.4) |
