# alpha158 日频因子增量更新 — 整体设计

> 状态：**方案稿**（已与全局架构对齐，待测试验收后转执行）。
> 目标：把 alpha158 从"一次性全量构建"改造为**工业级 append-only 日更**，与分钟侧统一同一套铁律。
> 设计准则：**简单、清晰、鲁棒**——复用已建好的 95%，新增只有一个日更脚本。
> 关联：`minute_incremental_design.md`（分钟侧增量，本文档孪生）/ `daily_update.md`（编排）/ `server_daily_production.md`。

---

## 1. 业务本质（先记住这两条，后面全由它推导）

1. **alpha158 是「无状态、按股独立」的计算**：158 个因子全是单股自身时序操作（Ref/Mean/Std/Slope/Rsquare、自身 close×volume 的 Corr），**raw 阶段零跨股交互**（rank、行业市值中性化都在 cleaned/neu 才发生）。宽表 pivot 只是向量化的实现便利，不是数学必需。
   → **推论**：增量不需要任何缓存状态、不需要分钟侧那套 pass2 新股建库。
2. **最大回看 = `max(WINDOWS) = 60` 个交易日**（所有 rolling/volume 因子都是左锚 window 操作，最长 `Ref(close,60)`/window=60，无更长依赖）。
   → **推论**：增量只要从源头回读尾窗 ~65 个交易日就能 bit 级复现末行（已经 smoke 验证）。

---

## 2. 系统原生契约（决定输出形态，不可违反）

**全系统所有因子（cxl/kysec/founder/...）的落盘形态都是「按因子的 WIDE 面板」**：
`factors/raw/<source>/<group>/<factor>.parquet`，`index=date × columns=stock`。

两个消费端都按 WIDE 读：
- `ml/dataset.py:142` → `read_parquet(...); df.index=日期; df.reindex(index=all_dates, columns=all_stocks)`
- `core/evaluation.py` → `factor_df.index`(日期) / `factor_df.columns`(股票)

→ **alpha158 输出必须是 WIDE 按因子**，与全系统一致。`alpha158_pv` 长表实验**放弃**：长表磁盘大 77%（17.7GB vs 宽表 10GB），且与上述两个消费端读法不兼容（长表 `df.index` 是 RangeIndex，`reindex(columns=...)` 拿到全 NaN）。对 alpha158 而言长表是负优化。

---

## 3. 数据流总览（三层 + 标签旁路）

```
L0 复权因子  daily/stock-ex-factors/<股>.parquet      (稀疏, 已有, ex_factors.py)
L1 日频原始  daily/stock-ohlcv/<股>.parquet            (按股, 不复权, raw_ohlcv.py 已增量)
                │ 读时复权(价×ffill(cum_factor)，量÷cum)
                ▼
[计算]  内存里 glob 全市场 → 各读尾窗 65 交易日 → pivot 临时宽表(date×stock×6字段)
                │ Alpha158Panel.compute_all(windows)
                ▼
L3 因子面板  factors/raw/alpha158/<group>/<factor>.parquet   (WIDE date×stock, 158个)

标签(旁路)  market-data/labels/forward_return_Nd.parquet      (近 N+1 天 maturing)
```

**关键：没有持久化的"统一长表"中间层。** 那个 multiindex 长表只活在计算的内存里（pivot 前的临时态）。持久化统一长表是反模式——parquet 不支持原地 append，每天 append 当日全市场行 = 重写整张大表（分钟侧 §2 实测否决的"按股单文件重写全史"病的翻版）。

---

## 4. 各层数据契约（实测）

| 层 | 路径 | 结构 | 规模 | 增量 | 新股 |
|---|---|---|---|---|---|
| L0 复权因子 | `daily/stock-ex-factors/<股>` | index=ex_date(稀疏)；`ex_cum_factor` | — | ✅ ex_factors.py | — |
| **L1 日频原始** | `daily/stock-ohlcv/<股>` | index=date；OHLCV+turnover+limit；**不复权** | 5508 文件 × 218KB = 688MB | ✅ raw_ohlcv.py 逐股 append+dedup | ✅ 每日 `all_instruments(CS)` 全集，新股自动建文件 |
| **L3 因子面板** | `factors/raw/alpha158/<group>/<因子>` | WIDE index=date, columns=stock | 158 文件 × 中位138MB = 10GB(宽表) | **本方案新建** | 由 §6 concat 列并集自动纳入 |

L3 四组（kline/price/rolling/volume）由 `core/producers/alpha158/groups.py:factor_group()` 归类。

---

## 5. warm-up 窗口（自动推导，禁硬编码）

- `W = max(compute_all 用的 windows)`，当前 `[5,10,20,30,60]` → **W=60**。
- 增量回读窗口取 `ceil(W×1.1)+buffer ≈ 65 个【交易日】`（从交易日历取，**不是 calendar days**）。
- 将来新增 window=120 的变体，W 自动变 120，**不改代码**——这是"自动推导"的意义。
- 与分钟侧统一：W 由计算定义解析得到，禁全局写死数字。

---

## 6. 增量算法（与四层同构：定起点 → 读 warmup+新日 → 只算新日 → append+dedup+原子写）

```
T    = 最新就绪交易日（含 19:05 就绪判断，与 raw_ohlcv 一致）
last = 任一因子面板的 max(date)；need = (last, T]
W    = max(WINDOWS) = 60 → 回读窗口 = 最近 65 个交易日

① glob 全市场按股 OHLCV 文件 → 每只 tail(65 交易日) → 读时复权 → 内存长表
   （新股 glob 到就有；不在 universe 的退市僵尸股自然不出现）
② pivot → 宽表面板 {open,high,low,close,volume,vwap}，65 × ~5508
③ Alpha158Panel.compute_all(windows) → 158 个宽表（尾窗上算，只有 need 的行是新的）
④ 逐因子：
     new_rows = factor_wide.loc[factor_wide.index > last]      # 切出 need=(last,T]
     combined = pd.concat([old_wide, new_rows])                # ★列并集自动纳新股
     result   = combined[~combined.index.duplicated(keep="last")].sort_index()  # 幂等
     result.to_parquet(tmp); os.replace(tmp, path)             # 原子写
```

**三个非平凡 pandas 机制（已在 toy demo 逐格验证，bit 一致）**：
- **新股自动**：`pd.concat([old{A,B,C}, new{A,B,C,D}])` 自动取**列并集** `{A,B,C,D}`，旧行在新列 D 上自动补 NaN。新股上市当天 glob 到 → 就是宽表多出的一列。**无需 pass2**（无状态计算相对分钟侧有状态 superset 的红利）。
- **幂等**：`~index.duplicated(keep="last")` 保留每日期最后一次；计算确定性 → 重跑 T 得到 bit 一致的表。
- **warmup 浪费**：尾窗里历史行（如 d3/d4/d5）被重算后丢弃（只切 `index > last`）——这点重算是 rolling 增量的固有成本，换来日成本 O(W) 而非 O(全史)。

---

## 7. 输出布局：满文件原子重写（不做分区，理由充分）

L3 因子面板每天 `read old → append 新行 → 写回`。因 parquet 无原地 append，等于**每天重写整个面板**（宽表全库 10GB / 158 文件，本机 ~1–2 分钟）。

**为何不学分钟侧做"按时间分片"避免重写**：分钟侧改按日分片是因为 76GB、重写 75GB vs 增量 34MB 的 **2000× 差距非改不可**；alpha158 才 10GB，为省这 1–2 分钟去搞按年分区（`<factor>/year=YYYY/` 4 层路径会破坏 `*/*/*.parquet` 的 glob 契约、消费端全要改）**不划算**。**简单清晰 > 微优化**——满文件重写 + tmp/os.replace 原子写，在 10GB 量级就是最优解。

---

## 8. 鲁棒性约束（与分钟侧统一 checklist）

| 约束 | 做法 |
|---|---|
| 原子写 | tmp + `os.replace`（POSIX 原子，写半截崩溃不损坏旧文件）|
| 幂等 | `dedup(keep last)`；重跑 T 不变（计算在 float-epsilon 相对噪声内确定，见 §9 洞察）|
| warmup 自动推导 | `W=max(windows)`，禁手写 |
| 零手动日期 | last=面板 max+1；T=最新就绪交易日 |
| 新股自动 | concat 列并集（§6）；无 pass2 |
| 抽样/全量对账 | truncate-replay（§9）上线前必过；周期性全量重建 reconcile |
| 标签 provisional | 末 N+1 天 maturing，IC 仅 >N+1 天可复现（旁路，与分钟共用）|

---

## 9. 验收：truncate-replay 真实数据对账（黄金标准）

**思路**：复制真实生产因子面板 → 砍掉尾部 K 天 → 从**未截断的按股 OHLCV 源**重放这 K 天 → 与原始全量面板逐格比。回答终极问题：「若当初是日更跑出来的，会不会和一次性全量构建一模一样？」严格强于旧 `smoke_alpha158_daily_update.py`（用真实全 universe + 真实 IPO + 对生产产出本身）。

**4 个必须设计对的点（否则假阳性通过）**：
1. **截断因子产出、不截源**：日更唯一状态是因子面板；OHLCV 源永远全。测试只砍因子面板副本，源不动。
2. **多天重放**：砍尾 K≈20 天，逐日重放（模拟连续日更），抓"多次 append 后才暴露"的 bug。
3. **显式断言新股**：在截断窗口里主动找一只真 IPO 股（截断面板里整列消失/全 NaN），重放后断言其新列回来且值吻合——不被"总体 max|Δ|=0"淹没。
4. **NaN-aware 比较**：`both=~(inc.isna()|full.isna())` 上算 `max|Δ|`；**另单独断言 NaN 模式逐格一致** `(inc.isna()==full.isna()).all()`，防"少算一片正好是 NaN"。

**通过判据**：重叠 (日期×股票) 上 `max_rel < 1e-6` **且** NaN 模式逐格一致 **且** 点名新股列吻合。

> ⚠️ **非平凡洞察（2026-06-14 实测修订）**：判据**不能用 `max|Δ|==0`（bit 级）**——pandas `rolling.mean/std` 用**滑动累加器**（加新值减旧值），day d 的结果携带从序列起点累积的浮点误差，**全量序列(~5000长) 与尾窗序列(70长) 累积误差不同**。量级小的字段（close~50）噪声 ~1e-16 看不出，量级大的字段（volume~1e8）绝对噪声放大到数千（VMA60 实测 max_abs=8192，但 **max_rel=5.3e-16 = float epsilon**）。故**正确判据是相对误差容差**。实测 6 因子：ROC5=0、MA/VMA~1e-16、STD60~6e-11、CORR5~1.5e-8，全 < 1e-6。
> **生产推论**：日更增量构建的面板与全量重建会差 ~float-epsilon 相对量（对 IC/排序完全无感），这是 pandas rolling 的固有性质、不可消除、可接受。

> ⚠️ **非平凡洞察②：对账只比"活区"，不比上市前死区**。CNT 族（CNTP/CNTN/CNTD）= `Mean((close>Ref(close,1)).astype(float), w)`，**布尔比较把 NaN 吃成 False→0**，加上算子 `Mean` 用 `min_periods=1`（`panel_operators.py:39`），导致 CNTP 在**上市前死区也有伪值 0**（全量构建的 latent 伪值）；增量重放该死区为 NaN，**反而更正确**。两者只在死区不一致，活区（个股上市后）完全吻合，而死区下游必被 `new_stock_mask` 抹掉。故 `reconcile` 按 `date >= 个股首个有效 close` 限定活区比较——这是正确语义，非"掩盖问题"。（MA 族因 close 本身 NaN→死区 NaN，无此问题。）

> **验收结论（2026-06-14，600 股样本 × 20 天重放）**：全 158 因子通过——活区 `max_rel=5.9e-8`（CORD5 最大）、NaN 模式逐格一致、3 只真实 IPO（001393/603435/688635）新股列自动回来且吻合。脚本 `scripts/smoke_alpha158_truncate_replay.py`（`--all` 验 158，默认验 6 个代表因子）。

**红利**：测试内核 = 生产 `alpha158_daily_update.py` 的核心。抽成共享函数 `incremental_append(factor_dir, ohlcv_source, up_to_date)`，生产日更调它、测试砍尾循环调它 → 杜绝"测试通过但生产是另一套"的脱节。

---

## 10. 编排：接入 daily_update.sh

alpha158 **无 spec**，`daily_update.sh` 第 7 步的 `FACTOR_GLOB`（扫 `sources/*/*/specs/*`）会跳过它 → 需独立调用：

```
数据线 1–5（ex_factors/raw_ohlcv/minute_ohlcv/industry/market_cap）  [已有]
 6.  refresh_supersets.py        分钟 L2 superset 增量(+pass2)        [已有]
 7a. paper_27 L3：run.py 循环 23 spec                                 [已有]
 7b. ★ python scripts/alpha158_daily_update.py    alpha158 L3 增量    [新建]
 8.  ml/labels.py                标签回填                              [已有]
```

7b 失败仅告警不中断（与 7a 一致）。end-date 复用脚本动态推导的最新交易日。

---

## 11. cleaned/neu（raw 跑通后再定）

raw 是核心。cleaned（MAD+zscore）与 neu（行业市值中性化）都是**逐日纯截面**操作、同样无状态可增量。最省事：raw 增量落地后，对增量段直接走 `run.py <factor> --evaluate-only`（便宜）。待 raw 验收后再决定是否单独做增量。

---

## 12. 决策记录

| # | 决策 | 取向 |
|---|---|---|
| 1 | 输入存储 | **维持按股**（已有/已验证/新股自动/写放大最小，0.69GB） |
| 2 | 持久化统一长表 | **不建**（parquet append 反模式，重写全表） |
| 3 | 输出形态 | **WIDE 按因子**（系统原生契约/比长表小/已验证）；放弃 `alpha158_pv` 长表 |
| 4 | 输出布局 | **满文件原子重写**（10GB/~2min；不做分区，量级不值得） |
| 5 | 新股 | **concat 列并集自动**，无 pass2（无状态计算红利） |
| 6 | warmup | **W=max(windows)=60**，自动推导，回读 65 交易日 |
| 7 | append-only/幂等/原子写 | 是（dedup keep last + tmp/os.replace） |
| 8 | 验收 | **truncate-replay 真实数据对账**（§9，4 判据） |
| 9 | 编排 | daily_update.sh 加 7b 独立步 |
| 10 | cleaned/neu | raw 跑通后再定（暂走 evaluate-only） |

---

## 13. 分期实施 & 进度（2026-06-14）

- ✅ **阶段1+2**：`scripts/smoke_alpha158_truncate_replay.py`（截断重放对账）全 158 因子通过（活区 rel<1e-6 + 3 真实 IPO 吻合）；`scripts/alpha158_daily_update.py` 已写（按股并行读 + compute_all + 逐因子 WIDE append + dedup + 原子写）。
- ✅ **首次基线 = 增量 catch-up（非全量重建）**：旧 WIDE 基线到 05-29（后复权 append-safe → ≤05-29 值不受 ex_factors 更新影响，可直接复用），用 `alpha158_daily_update.py` 一次性 append (05-29, 06-12] 10 个交易日补齐到最新。独立用 Alpha158Panel 重算 06-12 行对账：全 158 因子 max_rel=5.9e-8 ✅。**省去全量重建**。
  - 多天 catch-up 关键：回读窗口 = `[last-(W+buffer)交易日, T]`，随 gap 自适应，保证最早新日的 warmup（单日日更同一路径）。
- ⬜ **阶段3**：接入 `daily_update.sh` 7b（`python scripts/alpha158_daily_update.py`，零 API，失败仅告警）；首次稳态日更后跑一次 reconcile 确认 IC 复现。
- 📌 **偶尔全量重建**：`build_alpha158.py --full`（WIDE，按股分块内存安全）作为周期性对账/重置基线手段，非日常路径。
