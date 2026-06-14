# cxl 基本面因子增量更新 — 整体设计

> 状态：**阶段1-3 已落地**（L1 PIT 基础层 + 数据线 + fetch 读本地 + L3 增量验收；2026-06-14）。剩阶段4 编排收尾（§13）。
> 目标：给 cxl 基本面因子补上**本地 PIT 基础数据层**，实现**工业级 append-only 日更**，与分钟/alpha158 三胞胎统一同一套铁律。
> 设计准则：**简单、清晰、鲁棒**——复用已建好的增量内核（`incremental_append`）+ run.py 自动增量，新增只有"一层存储 + 一个数据线 + fetch 改读本地"。
> 关联：`minute_incremental_design.md` / `alpha158_incremental_design.md`（孪生）/ `daily_update.md`（编排）。
> 上手提示：先读 §1 业务本质 + §2 为何必须存；§6 是增量算法（与三胞胎同构）。

---

## 1. 业务本质（先记住这两条，后面全由它推导）

1. **cxl 基本面因子 = 「点位字段 → 纯截面计算」**：`fetch(get_factor 取 mrq/ttm/估值点位字段)` →
   `compute/rank/regress`（roe、yoy/qoq、SUE8、截面回归）。raw 阶段**逐股逐日独立**，无跨股交互
   （rank/中性化在 cleaned/neu 才发生）。"8 季"类（sue8/accs8）是 **fetch `net_profit_mrq_0..7` 八个点位字段
   后跨列 `row_aggregate`**——**不是时序 rolling**，仍无状态。
2. **最大回看**：绝大多数因子 **W2=1**（无时序依赖）；唯 `cross_section_regress`（`reg_pb_gshe`/`reg_pe_hist`：
   EP 过去 **252 日**中位数）**W2=252**。W2 由 `core.spec_resolver.max_rolling_window` **按 spec 自动解析**（禁硬编码）。

---

## 2. 为何「必须存」L1（核心决策，不可妥协）

cxl 与 alpha158 的根本差异：alpha158 有本地 `raw_ohlcv`（按股、增量维护），**cxl 没有本地基础数据**——
每次 `get_factor` 现场拉 API。直接拉的致命问题：

- **历史漂移 / 前视偏差**：直接拉时，历史某天 D 的值 = rqdatac **今天对过去的说法**。若厂商把财报**重述**
  灌回历史 → 回测引入**前视**（用未来才知道的重述值解释过去），且历史随时间漂移、**不可复现**。
- **破"历史冻结"铁律**：增量 append 的前提是历史不变；直接拉无法保证。

**结论：存本地每日 PIT 快照**——某天 D 拉到的值**当天落盘冻结、永不改写**（append-only）。这是 PIT 数据库金标准：
**as-first-reported、免疫重述、零前视、完全可复现**。与"存原始 + 读时复权"是同一条铁律的基本面版。

---

## 3. 三条铁律（与三胞胎统一）

1. **存 PIT 点位 + 历史冻结**：磁盘存每日 `get_factor` 快照；**只 append 当天、过去永不改写**。
2. **Append-only + 幂等**：每天追加 T 日；重跑 T 得到相同 T（`dedup(keep last)` + 计算确定性）。
3. **warm-up 自动推导**：W2=`max(spec rolling.window)`，按 spec 解析（无状态=1 / reg=252），禁硬编码。

---

## 4. 数据流总览（新增 L1 + 因子线改读本地）

```
L1 基本面PIT基础  market-data/fundamentals/<field>.parquet   (date×stock, 18字段, 新建)
       │  数据线 data_fetching/fundamentals.py 每日 append 当天快照（历史永不改写）
       │  因子线 fetch 算子改读本地（切 [start,end] → long）
       ▼
[计算]  compute / rank / filter / row_aggregate / cross_section_regress / rolling(252)  (纯截面, 不变)
       ▼
L3 因子面板  factors/raw/cxl/<group>/<factor>.parquet   (WIDE date×stock, 22个, 已有基线)
```

**关键：没有持久化"统一长表"**（同 alpha158 §3）；long 表只活在因子计算的内存里。

---

## 5. 各层数据契约

| 层 | 路径 | 结构 | 规模 | 增量 | 新股/新字段 |
|---|---|---|---|---|---|
| **L1 基本面PIT** | `market-data/fundamentals/<field>.parquet` | WIDE index=date, columns=stock | 18 字段 × ~90MB ≈ **1.6GB** | **本方案新建**（数据线 append） | 新股=concat 列并集自动；新字段=新文件 |
| **L3 因子面板** | `factors/raw/cxl/<group>/<factor>.parquet` | WIDE index=date, columns=stock | 22 文件 | run.py 自动增量（L3 改造已白送） | concat 列并集自动 |

**字段集（18，扫 spec get_factor 并集，可扩展）**：
`net_profit_mrq_0..8`、`net_profit_ttm_0`、`total_equity_mrq_0/1/4`、
`cash_flow_from_operating_activities_ttm_0`、`return_on_invested_capital_ttm`、
`market_cap_3`、`pb_ratio_lf`、`pe_ratio_ttm`。

> 布局选 **按字段 WIDE** 的理由：①截面消费**免 pivot**（直接 date×stock）；②文件少（18）；
> ③**直接复用 `core.yolo_engine.incremental_append`**（每日 append 一行=当天全市场快照、列并集纳新股、dedup、原子写）；
> ④1.6GB 整张重写秒级（同 alpha158 10GB 满文件重写范式）；⑤与因子面板同构。

---

## 6. 增量算法（与三胞胎同构：定起点 → 读 warmup+新日 → 只算新日 → append+dedup+原子写）

### 6.1 数据线（L1，`data_fetching/fundamentals.py`）
```
last = 各字段面板 max(date) 的最小值（保守对齐）；T = 最新就绪交易日
fetch get_factor(全市场, 18字段, start=last+1交易日, end=T)        # 只拉新日
逐字段：incremental_append(field_panel, 当日快照宽表, last)         # append > last, dedup, 原子写
```
- **零 warmup**（数据线只管存原始；warmup 是因子线的事）。
- **绝不回写历史**（`> last`）——冻结 PIT、防漂移的命门。
- 多天 catch-up（如基线 05-27 → 最新）：`get_factor` 区间一次拉齐 (last, T]，逐日 append。

### 6.2 因子线（L3，复用已落地的 run.py 自动增量）
```
fetch 算子（api=get_factor）：读本地 fundamentals 面板 → 切 ctx.[start,end] → 转 long   # 改这里
run.py：面板已存在→自动增量，fetch_start = last −(W2+10)交易日，W2 自动解析
compute/.../rolling → pivot → incremental_append（切 (last,T] append, 列并集, dedup, 原子写）
```
- **spec 一行不用改**（仍写 `api: get_factor`）；只改 fetch 算子实现 → 因子线零 API、可离线。
- L3 增量内核 + W2 解析 + run.py 自动检测 **已在分钟侧落地复用**（commit `6f84bb6`）。

---

## 7. 一次性历史 bootstrap（诚实说明）

- 首建：`get_factor` 全史 `[2010, T]` 拉一次建 18 个面板（一次性，~1.6GB）。
- ⚠️ **bootstrap 拉的是 rqdatac 今天对历史的说法**（老季度可能已含重述）——这是任何 PIT 库起步的现实
  （用厂商历史播种）。**真正的 as-first-reported 从上线日起、每天冻结快照逐步积累**；从今天起历史不再漂移。
- 与现有 cxl 基线面板（已到 05-27）的关系：bootstrap 建好 L1 后，因子线改读本地重算/续跑即与 L1 对齐。

---

## 8. 鲁棒性约束（与三胞胎统一 checklist）

| 约束 | 做法 |
|---|---|
| 原子写 | tmp + `os.replace`（复用 `incremental_append`）|
| 幂等 | `dedup(keep last)`；重跑 T 不变 |
| 历史冻结 | 数据线只 append `> last`，**绝不回写**（PIT 命门）|
| warmup 自动推导 | `W2=max(rolling.window)` 按 spec 解析（无状态=1 / reg=252）|
| 新股自动 | concat 列并集（L1 字段面板 + L3 因子面板）|
| 零手动日期 | last=面板 max+1；T=最新就绪交易日 |
| **重述审计**（非修正）| 周期比对"本地存 vs API 当前"，**只告警/记录、绝不覆盖**（覆盖即破坏冻结 PIT；与分钟 §8 reconcile 语义相反）|

---

## 9. 验收（服务器全量）

1. **读本地 == 读 API**：因子计算从本地 L1 vs 直接 API，同一窗口逐格 `max_rel<1e-6`（确认无缝切换）。
2. **truncate-replay**（源现在是本地文件，与分钟/alpha158 完全同构，复用 `incremental_append`）：
   砍因子面板尾 K 天 → 从未截断 L1 逐日重放 → 活区 `max_rel<1e-6` + NaN 模式逐格一致 + 真实 IPO 列吻合。
3. **历史冻结**：数据线重跑 / 多天后，历史段指纹不变（同分钟 step1 指纹法）。

---

## 10. 编排：接入 daily_update.sh

```
数据线 1–5（ex_factors/raw_ohlcv/minute_ohlcv/industry/market_cap）  [已有]
 5b. ★ python data_fetching/fundamentals.py   基本面 PIT 快照 append   [新建]
 6.  refresh_supersets.py        分钟 L2 superset 增量                  [已有]
 7.  run.py 循环 spec（含 cxl，读本地 L1，自动增量）                    [已有, fetch 改读本地后零 API]
 7b. (可选) alpha158_daily_update.py                                    [孪生待接]
 8.  labels 回填                                                        [已有]
```
- 5b 在因子线之前（因子线依赖 L1）；失败 fail-fast（数据线铁律）。
- 周期性 reconcile：因 L1 已冻结，因子线 `--rebuild` 从 L1 重算只是一致性自检（便宜）；L1 本身做"重述审计"（§8）。

---

## 11. cleaned/neu（raw 跑通后再定）

同 alpha158 §11：cleaned（MAD+zscore）/ neu（行业市值中性化）都是逐日纯截面、无状态可增量；
raw 增量落地后对增量段走 `run.py <factor> --evaluate-only` 即可，待 raw 验收后再决定是否单独增量。

---

## 12. 决策记录

| # | 决策 | 取向 |
|---|---|---|
| 1 | **存 L1 基本面基础数据** | ✅ **必须存**（每日 PIT 快照、冻结历史、防漂移/前视、可复现）|
| 2 | **L1 存储布局** | **按字段 WIDE** date×stock（截面免 pivot / 复用 incremental_append / 18 文件 / 1.6GB）|
| 3 | 字段集 | 扫 spec get_factor 并集（当前 18，可扩展；新字段=新文件）|
| 4 | 数据线 | 新建 `data_fetching/fundamentals.py`，append-only 每日快照，零 warmup，绝不回写历史 |
| 5 | 因子线接入 | **fetch 算子 api=get_factor 改读本地**；spec 不改；L3 增量复用 run.py 已落地改造 |
| 6 | warmup | `W2=max(rolling.window)` 按 spec 解析（无状态=1 / reg=252）|
| 7 | bootstrap | 一次性全史 get_factor 播种；真 PIT 从上线日起前向积累（§7 诚实说明）|
| 8 | append-only/幂等/原子写 | 是（复用 `incremental_append`，dedup keep last + tmp/os.replace）|
| 9 | reconcile | **重述审计**：比对本地 vs API 只告警、**绝不覆盖**（与分钟语义相反）|
| 10 | 验收 | 读本地==读API + truncate-replay + 历史冻结指纹（§9）|
| 11 | 编排 | daily_update 加 5b 数据线步（因子线之前）|

---

## 13. 分期实施 & 进度（2026-06-14）

- ✅ **阶段1（L1 存储 + 数据线 + bootstrap）**（commit `fece53e`）：`config.FUNDAMENTALS_DIR` +
  `data_fetching/fundamentals.py`（复用 incremental_append）；全史 bootstrap 跑通（~22min）→
  18 字段面板 `(3991,5552)` 2010-01-04~2026-06-12 共 394MB，skip=0。
- ✅ **阶段2（fetch 改读本地）**（commit `6cb0703`）：`_read_local_fundamentals` 接入；读本地 vs 读 API
  同窗（全 universe×近10日）共同键逐值 `max_rel=0`、NaN 模式一致（API 多出键全为 NaN，pivot 后等价）。
- ✅ **阶段3（增量 + 验收）**（commit `3e947ee`）：`scripts/smoke_cxl_l3_truncate_replay.py`（本地源、
  真实算子 + 生产内核、inf 感知 reconcile）。roe_apoq_mrq 真跑增量 05-27→06-12 append 12 日、历史段
  指纹冻结、新股列自动纳入。**22 因子分类验收**（见 §13.1）。
- ⬜ **阶段4（编排收尾）**：daily_update 加 5b 数据线步 + 重述审计 + cleaned/neu。

### 13.1 增量安全性分类（`core.spec_resolver.incremental_safe`，阶段3 落地）

并非所有 cxl 因子都能"有界尾窗增量"。两类**日历回看无界**的因子改为**全量重算**（从冻结 PIT 源
确定性重算、历史不漂移、截面计算便宜）：
- **rolling 带 `change_on`**（变化日采样）：window 是变化点个数，季频基本面 8 期≈8 季≈504 日、间隔随股异。
- **`filter` 在时序算子之前**：filter 删行 → 其后 rolling/transform 的 window/periods 按过滤后行数计。

| 类别 | 数量 | 因子 |
|---|---|---|
| **有界增量**（truncate-replay 全过 `max_rel=0`） | **17** | roe×5、npf×8（含 sue8/accs8 = row_aggregate 跨列、非时序）、pe×3、cashflow×1 |
| **全量重算**（incremental_safe=False） | **5** | reg_pb_gshe / reg_pe_hist（filter→rolling）、roic_ttm_{all_rnk8,dev_std8,ind_rnk8}（change_on）|

> ⚠️ **W2 解析教训**：`max_warmup_window` 必须覆盖 `transform`(diff/shift/yoy/qoq) 的 `periods`，
> 否则 reg_pe_hist(diff60)/pe_ttm_delta60(diff60)/roic 等会少 warmup 算错（阶段3 发现并修复）。

### 13.2 待确认 / 风险
- **PIT 实测**：bootstrap 前后用 `get_factor` 拉同一老窗口与现有基线对比，量化重述幅度（决定审计频率）。
- **链式时序算子**：当前各 spec 每条依赖链至多一个时序算子 → W2 取 max 正确；若未来同链叠加多个（如
  rolling 后再 diff），需改为按链求和（`max_warmup_window` docstring 已注）。
- **全量重算因子的成本**：5 个全量因子每日 run.py 全史重算（读本地 1.6GB → 截面计算），实测样本秒级；
  全 universe 待编排时确认在可接受耗时内。
