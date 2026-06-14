# 分钟因子增量更新 — 整体设计

> 状态：**L1/L2/L3 增量内核全部落地闭环**（2026-06-14；阶段1-3 完成，剩编排收尾见 §14）。
> 目标：把"一次性烤死版"分钟管线，改造为**工业级 append-only 日更**。
> 设计准则：**简单、清晰、鲁棒**。
> 上手提示：先读 §1 三铁律 + §2 存储决策；§6 是四层统一的增量算法；进度看 §12，落地细节看 §13，待办看 §14。

---

## 1. 三条铁律（先记住这三条）

1. **存原始 + 读时复权**：磁盘只存**不复权**原始数据 + 稀疏复权因子；后复权在**读时**实时算（`价×ffill(ex_cum_factor)`，量/额 `÷cum_factor`）。后复权 append-safe（新除权只影响除权日往后，历史不变）。与日频范式（`raw_ohlcv.py`/`ex_factors.py`）一致。
2. **Append-only + 幂等**：每天只追加 T 日；历史片冻结只读。重跑 T 必须得到完全相同的 T（`dedup(keep last)` + 计算确定性）。历史不动 → IC 可复现。
3. **warm-up 集中声明、自动推导**：增量算 T 必须回看足够窗口，少读一天就算错。窗口从 spec 自动推导（§5）。

---

## 2. 存储布局决策（已定，且经实测验证）

### 结论：分层选布局——分钟按「日」，日频维持「按股」

| 层 | 布局 | 路径 |
|---|---|---|
| **分钟原始**（76GB） | **按日分片，一日一文件装全市场** | `minute/raw/<YYYY-MM-DD>.parquet` |
| **日频原始**（0.71GB） | **维持按股**（不改） | `daily/stock-ohlcv/<stock>.parquet` |

**原则**：按数据体量 + 访问模式选布局，不强行一种布局套全部。两层共享同一套**原则**（存原始/读时复权/append-only/dedup/零手动日期/抽样校验），只是物理粒度因 100× 体量差而不同。

### 实测依据（80 股抽样 → 外推全市场）

**分钟：按日 vs 按年 vs 按股**

| 指标 | 按股单文件 | 按年(全股/年) | **按日(全股/日)← 采用** |
|---|---|---|---|
| 日更写入量 | ~75 GB | ~6 GB | **~34 MB** |
| 日更耗时 | ~553 s | ~47 s | **~0.4 s** |
| 是否重写历史 | 重写全史 | 重写当年 | **零重写,只新建1文件** |
| 读末21天 | ~157 s | ~8.4 s | **~1.7 s / 读0.7GB** |
| 峰值内存 | 全史 | 当年 6GB+ | **≤21天 ~0.7GB** |
| 文件数 | 5505 | 22 | ~5200 |

→ 按日在日更(写 1/177)、读窗口(1/9 I/O)、内存、崩溃安全上全面更优；是"历史冻结"的**物理纯 append-only 实现**；唯一代价"文件多(~5200 扁平、按日期命名)"在现代文件系统上不构成问题。

**日频：维持按股的依据**

- 仅 0.71GB → 按股 append 重写全量也就 ~1–2s，分钟那个"避免 76GB 重写"的理由在日频**不存在**。
- alpha158 **逐股消费**（逐股复权 + 逐股时序 Ref/Mean/Std/Slope）；"读单股全历史" 按股 O(1 文件)、按日 O(5200)。按股**更贴合**。
- 按股管线**已跑通 + alpha158 已 bit 级验证**；改布局要重写 + 重验，为 ~0 收益冒回归风险，不值。

---

## 3. 数据流总览（4 层 + 标签旁路）

```
L0 复权因子  daily/stock-ex-factors/<stock>.parquet      (稀疏, 已有)
L1 原始分钟  minute/raw/<YYYY-MM-DD>.parquet              (全股/日, 不复权, 新)
                │ 读时 × ffill(ex_cum_factor) → 后复权
                ▼
L2 日频superset  intermediate-cache/<key>__h<hash>/<stock>.parquet  (单股长表, 改增量)
                │ spec rolling / compute
                ▼
L3 因子面板  factors/raw/<source>/<group>/<factor>.parquet  (宽表 date×stock, 改增量)

标签(旁路)  market-data/labels/forward_return_Nd.parquet   (近 N+1 天持续 maturing, §7)
日频原始(并行,按股,不改)  daily/stock-ohlcv/<stock>.parquet → alpha158 日频因子
```

---

## 4. 各层数据契约（基于实测结构）

| 层 | 路径 | 结构 | 复权口径 |
|---|---|---|---|
| L0 复权因子 | `daily/stock-ex-factors/<stock>.parquet` | index=ex_date(稀疏~30行)；`ex_cum_factor`/`ex_factor`/`ex_end_date`；ex_cum_factor=上市起累乘绝对值 | — |
| **L1 原始分钟** | `minute/raw/<YYYY-MM-DD>.parquet` | cols=`order_book_id`,`datetime`,open,high,low,close,volume,total_turnover；**价格不复权**，量/额原始 | **不复权**（读时复权） |
| L2 superset | `intermediate-cache/<key>__h<hash>/<stock>.parquet` | 长表：`order_book_id`,`date` + 36 superset 列 | close 已后复权 |
| L3 因子面板 | `factors/raw/<source>/<group>/<factor>.parquet` | 宽表 index=date, columns=stock | — |

> 改动：旧 `minute/stock_data_1m_post/<stock>.parquet`（单股大文件 + 烤死后复权）→ 新 `minute/raw/<YYYY-MM-DD>.parquet`（全股一日一文件 + 不复权 + 读时复权）。

---

## 5. warm-up 窗口（结合 spec 实算，须自动推导）

- **L1→L2（分钟→superset）**：W1 = **`2×std_window`**（=40 日）。⚠️ 修订（2026-06-06 阶段2 V2.1 实测）：
  `peakridge_minute_corr_pooled` 是**嵌套两层 rolling**——某日 corr 依赖前 `std_window` 天的峰/岭标签，
  而每天标签又各需 `std_window` 天同时点 σ warmup → 块首正确须回看 `2×std_window` 天真实数据。
  旧表述"W1=std_window 同时点σ+pooled corr 同窗"**错**（漏了嵌套），分块/增量按 std_window 取 overlap
  会让块首 std_window 个目标日 corr 错（bit 对账可抓出）。
- **L2→L3（superset→因子）**：W2 = `max(该 spec 所有 rolling.window)`（实测 peak_interval 系列=20 日）。
- **稳态日更**：L2 读分钟末 `[T-W1,T]`（~21 个日文件）算 superset[T] → append；L3 读 superset 末 `[T-W2,T]` 算 factor[T] → append。**日成本 O(窗口) 而非 O(全历史)**。
- **实现（按 spec 解析，可定制，禁全局硬编码）**：
  - `W1 = spec 中 aggregate 步骤的 std_window`；`W2 = max(spec 中所有 rolling.window)`。
  - **两数每个 spec 自带**：新研报只要在自己 spec 写 `std_window: 60`/`rolling.window: 120`，引擎解析即自动生效，无需改代码。
  - 批量日更取"本次涉及所有 spec 的 W1/W2 各自 max"作读取窗口（该 max 也是解析得到，非写死）；安全余量 `ceil(W×1.5)`。
  - **当前全库实测（41 个 kysec spec）**：W1 全=20；W2：39个=20、2个=1 → 全库 max W1=20 / W2=20。

---

## 6. 增量算法（四层同构：定起点 → 读 warmup+新日 → 只算新日 → append+dedup+原子写）

### L0 复权因子（已有）
近 30 天回查 → append + dedup(keep last)。

### L1 原始分钟（重写 minute_ohlcv.py，按日分片）
```
end   = get_latest_trading_date()（含 19:05 就绪判断）
起点  = 末个已存在日文件的次一交易日（本地空 → FULL_START=2005-01-01）
对 [起点, end] 每个交易日 D:
  拉 get_price(全股, D, '1m', adjust_type='none')   # 原始不复权
  写 minute/raw/<D>.parquet（新文件；存在则 dedup 覆盖）→ tmp + os.replace
校验 = 抽样对齐（新拉重叠日 vs 磁盘逐分钟比对，口径=none 逐值相等）
```

### L2 日频 superset（改 minute_intraday_aggregate：统一"处理时间窗口"引擎）
**一个引擎，日更/全量都走它，只是窗口不同：**
```
① 主进程读【时间窗口的日文件集】concat → 大表
     日更:窗口=[T-W1, T]≈21个日文件≈0.7GB；全量:按时间分块迭代(逐年[年初-W1,年末]),内存有界
② 主进程 groupby(order_book_id) 预切成 {股票:子表}，挂【模块全局变量】(fork 前建好)
③ fork ProcessPool：每子进程认领一批股票，从全局字典按引用取(写时复制 COW，零拷贝、不 pickle)
     —— 读时复权(×ffill cum_factor)，每只股票仅对 need=(缓存last, T] 算 superset 行
④ append per-stock 缓存 + dedup(keep last) → tmp + os.replace
```
要点：① 大表当参数传给 submit 会被 pickle 复制 → 必须走"模块全局 + fork COW"；
② 生产 Linux 默认 fork 可用；Mac 调试需 `mp.get_context("fork")`。
回溯兜底：某历史日文件 mtime 变 → reconcile 触发该日有界重算。

### L3 因子面板（spec 引擎增量）
```
对每个因子:
  last = 面板最大 date；need = (last, T]
  读 superset 末 [min(need)-W2, T] → 跑 spec rolling/compute → 取 need 因子值
  append 新行到宽表 + dedup → tmp + os.replace
```

---

## 7. 标签（旁路 —— "IC 可复现"必须打的星号）

- `forward_return_Nd[t] = vwap[t+N+1]/vwap[t+1]−1`，依赖未来 N+1 天。
- **最近 ~N+1 天标签未长全，随新交易日逐天回填**（天然非 append-only）。
- 推论：**> N+1 天的 IC 完全可复现**；**最近 ~N+1 天 IC 为 provisional，会变**（forward-return 固有，非 bug）。
- 更新逻辑：每日重算「末尾 N+1 天 + 新日」标签，其余冻结。

---

## 8. 鲁棒性约束

| 约束 | 做法 |
|---|---|
| 原子写 | tmp + `os.replace` |
| 幂等 | 计算确定性 + `dedup(keep last)`；重跑 T 不变 |
| 抽样对齐校验 | L1 增量前：新拉重叠窗口 vs 磁盘逐值比对，不一致中止 |
| reconcile 审计 | 默认冻结历史；周期任务检测上游回溯修正（mtime/重叠值），命中 → 有界重算 + 告警 |
| warm-up 自动推导 | 扫 std_window + max(rolling.window)，禁手写 |
| 零手动日期 | 起点=磁盘 max+1 交易日；终点=get_latest_trading_date() |

---

## 9. 编排：一个日更驱动（按依赖顺序，任一层失败即停 + 告警）

```
daily_update:
  1. ex_factors.py        复权因子增量              [已有]
  2. raw_ohlcv.py         日频 OHLCV 增量(按股,不改) [已有]
  3. minute_ohlcv.py      原始分钟增量(按日文件)     [重写]
  4. minute superset 增量  只算新日 → append 缓存     [改算子]
  5. spec 引擎增量         只算新日 → append 因子面板  [改引擎/加增量入口]
  6. labels 增量          回填末尾 N+1 天 + 新日      [改/加]
  7.(可选) cleaned/neu/ML 刷新
```

---

## 10. 分期实施（每阶段先与"全量重算"逐值对齐再上线）

- **阶段1（L1）**：重写 minute_ohlcv → 原始 + 按日分片 + 增量 + 抽样校验。验收：读时复权 vs 旧 1m_post 逐分钟 bit 级一致。
- **阶段2（L2）**：算子加增量（读日文件窗口、只算新日）。验收：增量缓存 vs 全量重算逐值一致。
- **阶段3（L3+标签+编排）**：spec 增量 + 标签回填 + daily_update 串通。验收：增量面板 vs 全量逐值一致；历史 IC（>N+1 天）复现。
  - ✅ **L3 增量 + 验收已落地（2026-06-14）**：run.py 自动检测增量 + 生产内核 `incremental_append`；
    验收脚本 `scripts/smoke_minute_l3_truncate_replay.py`（truncate-replay 真实数据对账）全过。详见 §13.1 / §14。
  - ⬜ **编排收尾**：daily_update 第7步已自动跑增量（=循环 run.py）；待补周期 reconcile + 文档/注释（§14）。

---

## 11. 决策记录

### ✅ 已定
| # | 决策 | 取向 |
|---|---|---|
| 1 | 存原始不复权、读时复权 | 是 |
| 2 | **分钟存储布局** | **按日分片**（实测：写 1/177、读 1/9、零重写历史） |
| 3 | **日频存储布局** | **维持按股**（0.71GB，改无收益且有回归风险） |
| 4 | L2/L3 真增量 | 是 |
| 5 | append-only + 历史冻结 + 幂等 | 是 |
| 6 | 历史回溯修正 | 默认冻结 + 周期 reconcile 审计 |
| 7 | **a. warm-up 推导** | **按 spec 解析**（W1=std_window, W2=max rolling），可定制、禁硬编码；当前全库 max=20/20 |
| 8 | **c. L2 I/O 模型** | **统一"处理时间窗口"引擎 + fork COW 共享**（日更小窗口、全量分块） |
| 9 | b. 旧 `1m_post` | 阶段1 并行保留作对齐基准，验收 bit 级一致后删除（释放 76GB） |
| 10 | d. labels 增量 | **不急**：暂沿用现 build_labels；后续做"回填末 N+1 天"增量 |
| 11 | e. reconcile | **不急**：先靠抽样对齐校验兜底；阶段3+ 再加自动 reconcile |
| 12 | f. stock-data-fetching 仓同步 | 随阶段1 一起（minute_ohlcv 在该仓），走 git/SSH 流程 |
| 13 | g. 编排器 | 先**单脚本串行 + 失败即停**（简单鲁棒）；以后再上调度 |

### ✅ 已落地（2026-06-14 复盘项，详见 §13 / §14）
| # | 决策 | 取向 |
|---|---|---|
| 14 | **L3 改增量 append** | ✅ **已落地**（读 superset 尾窗只算新日 append+dedup+原子写，与 alpha158 L3 统一；commit `6f84bb6`，验收 `84c22bb`）|
| 15 | **pass2 新股回填** | ✅ **已落地**（持久化首现映射；从真实首现回填、去 504 上限/10日门两魔法数；commit `2f1708d`）|

### ⬜ 待落地（编排收尾，详见 §14）
| # | 决策 | 取向 |
|---|---|---|
| 16 | **L3 周期 reconcile** | 周频 `run.py <factor> --rebuild` 全量对账兜底（抓增量漂移）；先文档+cron 建议，未必改脚本 |
| 17 | **daily_update 收尾** | 第7步加注释（现已自动增量）；接 alpha158 第7b 步（孪生侧 L3 增量，零 API） |

> §1-13（编号 1-13）已全部"先简单、以后增强"定下；14-15（L3 增量 / pass2 简化）**已落地闭环**；16-17 为编排收尾待办。

---

## 12. 当前进度 & 下一步执行清单（⭐ session 恢复后照此继续）

### 已完成（阶段1 代码 + 验证）
- ✅ **`core/minute_data.py`**（factor-repl）：读时复权 loader `load_adjusted_minute_window`。
- ✅ **`stock-data-fetching/minute_ohlcv.py`**（已重写）：原始(none)**按日分片**抓取器，全量/增量同一路径、幂等可续传、双账号配额。
- ✅ **`stock-data-fetching/minute_reshape.py`**：旧post→raw 流式重塑（备选，见下）。
- ✅ **config `MINUTE_RAW_DIR`** = `market-data/minute/raw/`。
- ✅ **V1 全部通过**：反推/真拉 raw + 读时复权 == 旧 1m_post（~1e-7，量Δ=0，含除权期）。

### ⚠️ 全量历史构建：两条路（受磁盘约束，本地二选一）
| 路径 | 命令 | 耗时 | 磁盘 | 适用 |
|---|---|---|---|---|
| **A. 按日重新拉取** | `minute_ohlcv.py --full --workers 16` | **~8–11 小时**(16线程；实测线程仅~2.4x，rqdatac服务端限流) | 安全(一次一文件，删旧后峰值~76GB) | 本地磁盘紧、能等半天 |
| **B. 从旧post重塑** | `minute_reshape.py --chunk 60` | ~30–45min | **需~152GB瞬时**(旧76+新76) → 本地79GB装不下 | **服务器**(空间足) |

> 结论：**本地磁盘只 79GB**，A 慢但安全、B 快但装不下。
> - 想**本地跑熟**：建议先用**小样本**（如 `minute_ohlcv.py --full --limit-days 250` 跑近1年，或挑几十只）把 L1→L2→L3 整条走通，**不必本地全量**。
> - **全量留服务器**（800G）：那里用 B（reshape，~30min，免重新下载）最快。

### 下一步执行顺序（恢复后）
1. **（可选）commit 两个仓**：
   - factor-repl：`core/minute_data.py`、`core/config.py`、`docs/minute_incremental_design.md`
   - stock-data-fetching：`minute_ohlcv.py`、`minute_reshape.py`
2. **腾磁盘 + 建 raw**（本地若要全量/样本，按需）：
   - 验收已过，旧 `minute/stock_data_1m_post/` 可删（释放 76GB）：`rm -rf <DATA>/market-data/minute/stock_data_1m_post`
   - 拉数据：本地样本 `python minute_ohlcv.py --full --limit-days 250 --workers 16`；或服务器全量。
   - 增量日更：`python minute_ohlcv.py --workers 16`（1天~9s，极快）。线程实测 16 即上限（服务端限流，8→16仅+22%）。
3. ✅ **阶段2（L2）完成**（2026-06-06）：`minute_intraday_aggregate.py` 已改读 `minute/raw`（经
   `core.minute_data` 复权）+ 统一"处理时间窗口"引擎（按交易日分块 + fork COW 池 + append-only
   per-stock 缓存 + dedup/原子写）。验证脚本 `scripts/validate_minute_l2{,_engine}.py`。验收全过：
   - V2.1 数学+源 bit 级 == 旧 v4 golden 缓存（6 股，count 类精确，close 派生 ~1e-7）；
   - V2.1-引擎 全链路（分块+fork池+缓存）3 股全 36 列 bit；
   - V2.2 增量 append（截断缓存→追加尾部）== golden bit。
   - **关键发现**：W1=**2×std_window**（见 §5 修订，嵌套两层 rolling）。详见 `minute_stage2_l2_refactor.md`。
4. ✅ **阶段3（L3 增量 + pass2 简化）完成**（2026-06-14，服务器全量数据验收）：
   - **L3 增量**（commit `6f84bb6`）：`run.py` 面板已存在→自动增量（读 superset 尾窗 `[last−(W2+10)交易日, T]`
     只算新日 → `incremental_append` 列并集纳新股 + dedup + 原子写）；`--rebuild` 全量兜底。W2 由
     `core.spec_resolver.max_rolling_window` 按 spec 解析。
   - **验收**（commit `84c22bb`）：`scripts/smoke_minute_l3_truncate_replay.py`（superset 为源、真实 rolling 算子
     + 生产内核、砍尾重放）600 股×20 天 `max_rel=0` + NaN 模式一致 + 真实 IPO 列吻合；全 universe 真跑
     pj_peak_minute_count 仅 append 5 新日、历史段指纹不变（真冻结）、pass2 新股列自动纳入。
   - **pass2 简化 #15**（commit `2f1708d`）：持久化首现映射（并行建 14s，增量 O(1)）；从真实首现回填、
     去 504/10日 两魔法数；真实回填 golden 对账 `max_rel=0`。
   - ⬜ **剩余编排收尾**见 §14（周期 reconcile + daily_update 注释/alpha158 7b）。

### 关键不变量（别忘）
- W1=**2×std_window**（L2 嵌套 rolling，§5 已修订）、W2=max(rolling.window)，**按 spec 解析**（§5），当前 std_window=20→W1=40。
- append-only + 幂等 + 原子写 + 读时复权（§1 三铁律）。
- 标签近 N+1 天 maturing，IC 仅 >N+1 天可复现（§7）。

---

## 13. L3 增量化 + pass2 简化（2026-06-14 复盘）

> 背景：alpha158 日频增量已落地（`docs/alpha158_incremental_design.md`），回头审分钟侧发现 L3 与
> pass2 两处可统一/加固。端到端目标：**L1→L2→L3 全 append-only、每层每日成本有界**。
> 现状：L1✅ L2✅ L3✅（2026-06-14 L3 增量 + pass2 简化均已落地，见下）。

### 13.1 L3 全量重算 → 已改增量 append ✅（2026-06-14 落地）
- **现状**：`run.py <factor>` 经 `minute_intraday_aggregate` 算子，**读全量 per-stock superset**
  （`minute_engine.py:114` 读整文件）→ 重算**整张 (date×stock) panel**。
- **为何当初这样**：实现简单 + 重活在 L2（分钟→superset 已增量缓存），L3 读紧凑 superset(36列)+轻
  rolling(W2=20)，"够便宜"。
- **长期不高效**：每日成本 = O(总天数×股×因子)，**随历史线性增长**（为加 1 天重算 ~4000 天冻结历史，
  compute+IO 双浪费）。与 alpha158 极力避开的"每天重写冻结历史"同病。
- **改法（与 alpha158 L3 统一）**：读 superset **尾窗** `[last_factor − W2 − buffer, T]` → **只算新日 →
  append wide panel + dedup(keep last) + 原子写**。每日成本降到 **O(W2+gap) 恒定**。新股在 L3 由
  `concat 列并集`自动出现（L2 pass2 负责灌进 superset）。L3 无跨日耦合（pooled corr 已在 L2 算完），
  W2=20 尾窗即可，改造干净。**可复用 alpha158 的 truncate-replay 同款验证**。
- **代价/兜底**：增量冻结历史 → 过去的错不自愈 → 配**周期性全量 reconcile**（`--rebuild` 对账，抓漂移，见 §14）。
- ✅ **落地（commit `6f84bb6`）**：注入点 = **run.py 自动检测**（面板存在且非 `--rebuild` → 增量；否则全量覆盖）。
  - 共享内核 `core.yolo_engine.incremental_append(out_path, wide, last, rebuild)`（生产 + 验收共用，防脱节）；
    `YoloEngine.run()` 加 `incremental/rebuild` 参数，写盘处改调内核（含 tmp+os.replace 原子写，全量路径也受益）。
  - W2 = `core.spec_resolver.max_rolling_window(spec)`（解析 `action==rolling` 的 max window，禁硬编码）；
    `fetch_start = last − ceil((W2+10)×1.6) 日历日`（warmup 安全余量，尾窗行 `> last` 切掉丢弃）。
  - ✅ **验收（commit `84c22bb`，`scripts/smoke_minute_l3_truncate_replay.py`）**：以 per-stock superset 为源、
    真实 `rolling` 算子（经最小 Context）+ 生产内核重放。600 股×20 天 `max_abs=0`/`max_rel=0`、NaN 模式逐格一致、
    窗口内真实 IPO 688813.XSHG 列自动回来且吻合。全 universe 真跑 pj：仅 append 5 新日、历史段指纹不变（真冻结）。

### 13.2 pass2 新股逻辑 → 已简化为持久化首现映射 ✅（2026-06-14 落地）
- **旧逻辑**（已替换）：`new_obs ∩ 近10日raw出现` = active → 回扫**最近504日(2年)**建库。
- **脆弱点**：
  1. **504 天硬上限**：上市>2年却无缓存的股（缓存被清/历史 universe 漏）只补 2 年，`[IPO, T-504]` 永久留洞，**不自愈**（文档原标"完整修复要 --rebuild"）。
  2. **10 日检测窗口**：新股上市即停牌>10 日 → 当天被当僵尸跳过，复牌后才补（短暂滞后）。
  3. **本质**：有状态检测（cached 集合 diff + 启发式窗口 + 上限），失败面天生大于 alpha158 的无状态
     `concat 列并集`（零检测）。
- **改法**：把"504 上限"换成"**从该股 raw 首次出现日回填**（无上限）"。真 IPO 成本不变（本就<2年），
  深坑老股**完全自愈**，去掉 504/10日 两个魔法数。逼近无状态鲁棒度。僵尸股仍靠"无任何 raw 数据"
  天然过滤（不必近10日启发式）。
- ✅ **落地（commit `2f1708d`）**：新增**持久化首现映射** `first_appearance_map()`
  （`minute/minute_first_appearance.parquet`，{order_book_id: 首次出现交易日}）：
  - 首建 = 并行全扫 raw 单列 order_book_id（64 进程 ~14s，vs 串行 ~416s）；之后每天只增量扫
    `scanned_through` 之后的新日（O(1)）；哨兵行记录已扫到哪天；纯 raw 派生、可删文件重建、原子写。
  - pass2 重写：**僵尸过滤 = 不在映射**（永无 raw）→ 干掉近 10 日启发式；**ns_start = active 最早真实首现**
    → 干掉 504 上限 → 深坑老股完整回填 `[首现, T]`、完全自愈；退市但有过数据的股也纳入（更完整历史）。
  - ✅ **验收**：并行映射覆盖 5506 股 == 串行基准；幂等 + 哨兵回拨 5 天增量重扫 bit 还原；ns_start 逻辑
    （深坑股从真实首现覆盖、旧版留洞）；**真实回填 golden 对账**（删 301596 缓存→`refresh_cache` 重建
    507 行 `max_rel=0`、NaN 模式一致 == golden 全量）。

### 13.3 为何 L2 保持有状态缓存（不改）
- L2（分钟→superset）的重活值得缓存：分钟 76GB，无状态每天重读太贵。L2 增量本身已正确 append-only。
- 结论：**只动 L3（改增量）+ pass2（简化回填），L2 不动**。这样分钟侧与 alpha158 收敛成同一套
  "尾窗→只算新日→append" 增量范式。

---

## 14. 编排收尾（待落地清单，2026-06-14）

> L1→L2→L3 增量内核已全部落地闭环（§13）。本节是把"剩余编排选项"显式落档，逐项可独立执行。
> **现状**：`pipeline/daily_update.sh` 第 6 步 `refresh_supersets.py`（L2+pass2）、第 7 步循环
> `run.py <spec>`（L3）——因 run.py 已自动检测增量，**第 7 步现在已经在跑增量了，无需改动即生效**。

### 14.1 ⬜ L3 周期性全量 reconcile（决策 #16，兜底）
- **为何**：增量冻结历史 → 万一过去某天算错不自愈（与 L2 同理）。需周期性全量重算对账抓漂移。
- **做法**：周频（如每周日）对每个 spec 跑 `python run.py <spec> --rebuild`（全量覆盖，原子写），
  与 `refresh_supersets.py --rebuild`（L2 兜底）对称。判据：与现有增量面板活区 `max_rel<1e-6`。
- **落地选项**：(a) 文档 + 一行 cron 建议（最简，推荐先做）；(b) 写独立 `pipeline/reconcile_l3.sh` 扫 spec 循环 --rebuild。
- **代价**：全量重算成本（pj 全 universe ~分钟级），周频可接受；放非交易时段。

### 14.2 ⬜ daily_update.sh 收尾（决策 #17）
- **第 7 步加注释**：说明"面板已存在→run.py 自动增量（读 superset 尾窗只算新日 append）；首建/`--rebuild` 才全量"。
  脚本逻辑无需改（END_DATE 已动态取最新 raw 交易日）。
- **接 alpha158 第 7b 步**（孪生侧 L3 增量，零 API，也还没接进编排）：在第 7 步后加
  `python scripts/alpha158_daily_update.py`（失败仅告警不中断，与 7a 一致；见 `docs/alpha158_incremental_design.md` §10）。
- **首现映射热身**：首个增量日 `refresh_supersets.py` 会触发一次性并行全扫建 `minute_first_appearance.parquet`
  （~14s，一次性）；之后每天 O(1)。无需手动预建。

### 14.3 ⬜ 标签增量（决策 #10，仍"不急"）
- `ml/labels.py` 现为全量；§7 标签近 N+1 天 maturing。后续做"回填末 N+1 天 + 新日"增量，IC 仅 >N+1 天可复现不变。

> 排序建议：14.1(a) 文档+cron → 14.2 注释/接 7b → 14.3 标签（独立，可最后）。三项互不阻塞。
