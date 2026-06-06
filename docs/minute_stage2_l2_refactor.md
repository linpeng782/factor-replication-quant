# 阶段2(L2)改造方案 —— minute_intraday_aggregate 接 minute/raw

> 衔接 `minute_incremental_design.md` §6-L2 / §10 阶段2 / §12。
> 范围：**只做 L2**（分钟→日频 superset 算子）。L1 已完成(V1)，L3/编排留阶段3。
> 准则：superset 数学**一字不改**（保证 bit 复刻旧缓存）；只换数据源 + 加增量引擎。

---

## 1. 现状与既有资产（已核实）

| 项 | 状态 |
|---|---|
| L1 loader `core/minute_data.py` | ✅ 完成 + V1 bit 验证（读时复权 raw == 旧 post，~1e-7） |
| 已下数据 `minute/raw/` | 2005-01-04 → 2016-12-26 **连续**（仍在下，err=0），足够全部结构验证 |
| **L2 算子阻塞点** | `minute_intraday_aggregate.py` L37/L192 仍读 `MINUTE_DATA_DIR`（旧 1m_post，**已删**）→ 现在跑不起来 |
| **🎯 黄金参照** | `intermediate-cache/prv_v3__hc09d46528f/`（5505 股，2005→2026-06，36 列）= 旧管线 **v4** superset 冻结真值（hash 与当前代码 v4 完全一致） |

**核心验证逻辑**：V1 证 `raw→loader == 旧post`(bit) ＋ superset 数学(v4)不变 → 新算子(读raw)产出**应当 == 旧缓存**(bit)。这份 v4 缓存覆盖全史，故在 2005–2016 任意 (股,日) 都可直接对账。

---

## 2. 不变量（改造中必须守住）

1. **superset 数学不动**：`_compute_one_stock` 的宽表化 / 同时点σ / 峰岭谷分类 / 36 列计算**逐行保留**——这是 bit 复刻旧缓存的前提。只改"数据从哪来"。
2. **读时复权**：close 必须经 `core.minute_data._post_adjust`（×ffill cum_factor）后再进数学；volume/turnover 原始量不动。
3. **append-only + 幂等 + 原子写**：增量只追加新日；重跑某日得相同值；tmp + `os.replace`。
4. **warmup = W1 = std_window**（=20）：仅**全史起点**前 W1 日为 NaN；分块/增量的 overlap 日是**真实 warmup 数据**，不可误置 NaN。

---

## 3. 改造设计（具体到函数）

### A. 数据源切换（最小改动，先让它能跑）
- `_compute_one_stock(ob, src_path, ...)` → `_compute_one_stock(ob, raw_df, ...)`：
  入参从"读 `MINUTE_DATA_DIR/<ob>.parquet`"改为"接收**已复权的单股长表** raw_df(列: order_book_id,datetime,OHLCV)"。函数体（L218 之后）**完全不变**——它本就只要 `datetime`+OHLCV。
- 复权在进数学前完成：worker 拿到原始单股子表后调用 `core.minute_data._post_adjust(sub, ob)`。

### B. 引擎重构 —— "处理时间窗口"（design §6-L2）
替换 L124 的"逐股各读各文件"为统一窗口引擎：
```
读【日文件窗口】concat → 大表
  groupby(order_book_id) 预切 {股票: 原始子表}  → 挂【模块全局变量】(fork 前建好)
  fork ProcessPool：worker 从全局 dict 按引用取子表(COW,零拷贝,不 pickle)
    → _post_adjust → _compute_one_stock → 返回该股 superset 行
  主进程 append per-stock 缓存 + dedup(keep last) → tmp + os.replace
```
- **全量重算**（建缓存/验证）：内存有界——**逐年分块**迭代，块 k 读 `[年初 - W1 交易日, 年末]`，算完只留 `[年初, 年末]`（前 W1 是 overlap，仅全史第一块的前 W1 因 `min_periods` 自然 NaN）。
- **增量日更**：窗口 = `[T - W1, T]`（~21 日文件），`need=(cache_last, T]`，只算 need 行 → append。
- Mac 调试 `mp.get_context("fork")`；Linux 默认 fork。

### C. 缓存失效策略
旧：per-stock `cache_mtime >= src_mtime`。新源是多日文件 → 改为 **append-only 按日**：
- 增量：读缓存 last_date，只算 `(last_date, T]` 追加。
- 全量重建：删该 `cache_key__h*` 目录后重算（显式，不靠 mtime）。
- 上游回溯(日文件 mtime 变)→ reconcile，**阶段3+ 再做**（design 决策#11 不急）。

### D. params_hash / version
`_resolve_cache_dir` 不变；`_FEATURES_SUPERSET_VERSION` 维持 `v4`（动它就对不上旧缓存）。

---

## 4. 验证方案（全部可在已下载的部分数据上跑，无需全量/无需 API）

| 级别 | 验什么 | 方法 | 验收 |
|---|---|---|---|
| **V2.0** | 能跑通 / 列齐 / 无 NaN 爆炸 | 新算子跑 ~20 股 × 2010–2012 | 36 列、形状对、warmup 后非全 NaN |
| **V2.1 ⭐bit 级** | 新算子 == 旧 v4 缓存（最强，借力 V1+真值） | 取重叠 (股,日)∈2005–2016，逐列比对 `prv_v3__hc09d46528f/<ob>.parquet` | **count/n 类整数列精确相等**；vwap/turnover/moment 等浮点列 rel ≤ 1e-6（close 复权路径差） |
| **V2.2** | 增量 == 全量重算（period-agnostic） | 全量建到 D-1 → 增量 append D；对比全量重算到 D 的 D 行 | 逐值一致 |
| **V2.3 (L3)** | 端到端最简因子 | `peak_minute_count`(std_window=20→rolling mean20) 跑面板 + 增量==全量 | 面板逐值一致 |

> V2.1 是主验收：直接证明"换源+复权"没引入任何偏差。count/turnover/volume 列不受复权影响→应**精确相等**，浮点差只可能出现在 close 派生列(vwap/ridge_return/daily_*)且 ≤1e-6，判别力强。

---

## 5. 执行步骤（顺序）

1. **改 A（数据源）**：`_compute_one_stock` 接 df；worker 内 `_post_adjust`。先用"逐股各读窗口"过渡版跑通 V2.0 + V2.1（不急上 COW，先证数值对）。
2. **改 B（引擎）**：上"日文件窗口 + groupby + fork COW + 分块/增量"。重跑 V2.1 确认仍 bit 一致 + 测内存有界。
3. **增量入口**：加 `[T-W1,T]` 窗口的 append 路径，跑 V2.2。
4. **L3 端到端**：`run.py peak_minute_count`（限 2005–2016 + 限股），跑 V2.3。
5. 文档回填：design §12 进度勾掉阶段2；写本文件"复现结果"段。

---

## 6. 风险 / 决策点

- **R1 复权放主进程还是 worker**：选 **worker**（match §6，复权可并行；主进程只切原始子表，COW 更省内存）。
- **R2 分块 overlap 正确性**：块边界的前 W1 日必须用真实数据 warmup（非 NaN），只有**全史第一日**起的 W1 日是真 NaN。验收 V2.1 在跨年边界取样可发现此类 off-by-W1 错误。
- **R3 浮点容差**：V1 是 ~1e-7（float32 close）。V2.1 浮点列用 rel≤1e-6；若某列超差→说明复权口径/加权方式偏了，需查。
- **R4 旧缓存覆盖到 2026 但 raw 只到 2016**：对账只在重叠区间，天然满足"部分数据先验证"。


---

## 7. 复现结果（2026-06-06 完成）

实现：`core/operators/minute_intraday_aggregate.py`
- `_compute_one_stock(ob, raw_df)`：入参改为已复权单股长表，数学体不动。
- `_refresh_superset_cache`：统一"处理时间窗口"引擎——按交易日分块（`MINUTE_CHUNK_DAYS`，默认 250）
  迭代，块首回看 **`2×std_window`** 交易日真实数据 warmup；`mp.get_context("fork")` 池 + 模块全局
  `_WORKER_SUBTABLES`（COW 共享子表，不 pickle 大表）；per-stock 缓存 `append + dedup(date,keep last)
  + tmp/os.replace`；起点 = 各股缓存 last 次日（幂等增量）。
- 数据可用交易日历直接取自 `minute/raw` 现存日文件（离线、无需 rqdatac）。

验证（`scripts/validate_minute_l2.py` / `_engine.py`，对账 golden `prv_v3__hc09d46528f`，全程写独立临时目录）：

| 验收 | 范围 | 结果 |
|---|---|---|
| V2.1 源+数学 | 6 股 2005–2008 | ✅ 22 exact 列 max_abs=0；close 派生列 ~1e-7；带符号 ridge_return_sum abs≤7.7e-7 |
| V2.1 引擎全链路 | 3 股 2005–2023（~18 块） | ✅ 全 36 列 bit（含 corr_pooled max_abs=0） |
| V2.2 增量==全量 | 1 股 截断2015→追加2025 | ✅ 旧段+追加段全 36 列 == golden bit；追加首日 corr 非 NaN |

### 关键发现：W1 = 2×std_window（非 std_window）
`peakridge_minute_corr_pooled` 嵌套两层 rolling（标签 σ-window → 标签 pooling-window），块首目标日的
pooling 窗口会落进"标签未 warmup"的 overlap 区。**首版按 std_window 取 overlap → 块首 std_window 个
目标日 corr 错（998 行 NaN 错配），bit 对账当场抓出 → 改 2×std_window 后全过。** 这条已回填 design §5/§12。

### 待办（阶段3）
- V2.3 L3 端到端（`run.py peak_minute_count`）：全 universe×全史，建议服务器跑。
- spec 引擎增量 + `daily_update` 编排（design §9）。

---

## 8. V2.3 L3 端到端 + 两处增量 bug（2026-06-06）

数据全量下载完成（5201/5201 交易日，2005-01-04~2026-06-05，0 缺失）后跑通 L3：

- **universe 迁移**（`core/yolo_engine.py`）：`MINUTE_DIR` 旧版扫已删的 per-stock 目录 → 改用
  `all_instruments(CS)`（5551 ⊇ golden 5505 完整超集；无 raw 数据股由算子 groupby 自然跳过）。
- **`run.py peak_minute_count --yolo-only`**：仅增量追加 2026-06-02~06-05（4 天，5208 活跃股），
  → rolling mean20 → 面板 **(3979 日 × 5468 股)**，非空 13.88M。
- **评估**：neu RankIC20d=+0.061 ICIR=0.82、分层单调 +0.957、多空年化 +20.7% Sharpe 3.41
  （研报 RankIC 10.6%/多空 31.6% 同方向，量级合理）。

### 增量 bug #2：前沿必须用 max(cache_last) 不是 min
退市股 golden 缓存 `cache_last` 停在其退市年（实测抽样 ~6% 在 2005~2024）。首版增量起点用
`min(cache_last)` → 被退市股拖到 2005 → **误触发全 universe 全量重建**（5551 股从 2005）。
修：前沿 = **`max(cache_last)`**（append-only 已处理到的最新 raw 日）；退市股在新日无数据、load
不产出，无害。无缓存新股（46 只）增量模式跳过（避免写残缺尾部缓存），需全史另行全量重建。
> 局限（阶段3 reconcile 再处理）：若历史某次增量部分失败致缓存不一致（活跃股 cache_last 落后于
> frontier），max 起点会漏补其缺口；当前 append-only 假设每次运行对全活跃股同步推进。

### 阶段2 收口
L2 全链路（源切换 / 分块 fork-COW 引擎 / append-only 缓存 / 增量前沿 / L3 端到端）验证通过。
