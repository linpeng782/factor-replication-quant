# 高频因子工厂 — 总体设计

> 状态：**设计定稿**（经 12 篇研报压力测试归纳：开源 4 篇[聪明钱2.0/APM进阶/APM原始/峰岭谷] + 方正 8 篇[适度冒险/潮汐/勇攀高峰/球队硬币/模糊性/飞蛾扑火/草木皆兵/水中行舟]）。
> 关联：`minute_incremental_design.md`(L1+L2a 日更) / `minute_stage2_l2_refactor.md`(L2a 实现与 bit 验证)。
> 准则：**简单、清晰、鲁棒；不变的重活只写一次，可变的归约每篇只写一段。**

---

## 1. 一句话定位

把"研报 → 高频选股因子"工业化：**分钟级原始数据 → 可缓存的日频中间产物 → spec 图组合成因子 → 清洗/中性化**。
核心洞察：**最贵的"分钟→日频归约"算一次、按家族共享；因子只是其上的轻量时序/截面组合。**

---

## 2. 设计依据：12 篇实证出的【三种计算模式】

| 模式 | 计算形态 | 代表研报 | 落在 |
|---|---|---|---|
| **① 逐股 minute→日频** | 只看单股、单时间维度内的归约 | 峰岭谷、潮汐、勇攀高峰、模糊性、跳跃度、聪明钱、分时收益 | **L2a Reducer** |
| **② 市场/参照序列** | 沿股票维度、每个时间点聚合，得一条共享序列 | APM 指数收益、草木皆兵 中证全指、适度冒险/球队硬币 截面均值、水中行舟 分钟分化度 | **L2b 截面 Reducer** |
| **③ 跨股耦合 / 面板** | 某股依赖**全体**股票的序列（N² 两两相关） | 水中行舟（随波逐流/孤雁出群） | **L3 面板算子** |

**纪律（被 12 篇校准出来的边界）**：
- 跨**日**滚动 → L3（别塞进 reducer）；跨**股**聚合/相关 → L2b 或 L3 面板（逐股 reducer 一次只看一只股，做不了）；
- 纯**日频**因子（球队硬币、草木皆兵主体）**不进分钟引擎**，直接走日频+截面层；
- **reducer 只管"单股、单时间维度内"的归约**——这条守住，抽象就稳。

---

## 3. 总架构（分层）

```
L1  原始数据
    minute/raw/<日>.parquet (个股分钟, 不复权, 已建)
    daily/stock-ohlcv + stock-ex-factors (日频+复权因子, 已建)
    index/minute|daily (指数, 新增基建 —— ②需要)
      │ 读时复权 (价×ffill cum_factor; 量/额原样)
      ▼
L2a 逐股 minute→日频 Reducer  ── MinuteAggregateEngine + 可插拔 Reducer (已建✅, bit验证)
      产物: intermediate-cache/<cache_key>__h<hash>/<股>.parquet  (逐股×逐日 × N列)
L2b 截面/市场参照 Reducer (新) ── 同引擎的"跨股聚合"模式 或 指数序列
      产物: 一条全市场共享序列 (date[×minute]); 例: 截面均值/分钟分化度/指数分时收益
      ▼
L3  spec 图算子
      逐股: rolling / compute / rank / transform        (已有)
      截面: cross_section_regress                       (已有)
      面板: 两两相关 N² (BLAS 相关矩阵→行均值)          (新, ③需要)
      ▼
    factors/raw → cleaned(MAD+zscore+mask) → neu(行业市值中性化)
```

---

## 4. L2a：逐股 minute→日频 Engine + Reducer（已建，本会话 bit 验证）

**把"不变的引擎"与"可变的 reducer"切开**——这是泛化性的关键。

### 4.1 Engine（写一次，所有研报共用；本会话已实现并 bit 验证）
职责（全是易错的重活，已踩坑调对）：窗口读取日文件 → 读时复权 → pivot 宽表 → **按交易日分块**（内存有界）→ **fork COW 进程池**（子表挂模块全局，不 pickle 大表）→ **append-only per-stock 缓存 + dedup + 原子写** → **增量前沿 / warmup-overlap**。
- 输入给 reducer：**单股、一个窗口的（已复权）分钟数据**。
- 引擎不关心归约数学。

### 4.2 Reducer 接口（每篇研报**只写这一段**）
```python
class Reducer:
    superset_columns: list[str]      # 它产出哪些日频特征列
    warmup: int                      # 它的跨日回看(交易日); 嵌套rolling要算够(见§7)
    granularity: "half_day|hourly|minute"   # 它要多细的日内数据 → 引擎按需读
    params: dict                     # 变体参数 → 进 cache_key 的 hash
    def reduce(self, one_stock_window_minutes) -> daily_rows_df: ...
```
- `minute_intraday_aggregate`(峰岭谷)、`minute_pricejump_aggregate`(跳跃) 从"两份引擎拷贝"降级为**两个 Reducer**。
- 新研报边际成本 ≈ 写 `reduce()` + 声明 4 个属性。

### 4.3 缓存契约
- 路径 `intermediate-cache/<cache_key>__h<sha1(params+version)>/<股>.parquet`，逐股长表（order_book_id,date + superset列）。
- **一份 superset 服务一族因子**（cache_key 即"研报家族"）：胖（峰岭谷 36 列共享 23 因子）或瘦（聪明钱 2 列）皆可。
- 变体（β/cutoff/std_window…）= params → 不同 hash → 各自缓存；append-only，历史冻结。

---

## 5. L2b：截面/市场参照 Reducer（新）

模式② 的需求：一条**全市场共享序列**供逐股因子比较。两种来源，统一为"沿股票维度、按时间点聚合"：
- **指数派生**：把 §4 的 Reducer（如分时收益）作用在**指数 instrument** 上 → 指数分时收益（APM 的 R_am/R_pm；草木皆兵中证全指日收益）。
- **横截面聚合**：对 minute×stock 宽矩阵**按时间点跨股聚合** → 分钟市场分化度(std)、截面均值（水中行舟、适度冒险、球队硬币）。

接口：与 §4 同引擎，增加"cross-sectional 模式"（reduce 沿 stock 轴、对每个时间点聚合，输出一条 date[×minute] 序列）。产物小、共享、append-only。

---

## 6. L3：spec 图算子层

| 类 | 算子 | 状态 |
|---|---|---|
| 逐股时序 | rolling(mean/std/sum) / compute / rank / transform / row_polyfit / row_correlate | 已有 |
| 截面回归 | cross_section_regress（stat~Ret20 取残差；截面去均值/翻转） | 已有 |
| **面板相关** | **pairwise N² 相关 → 行均值**（水中行舟随波逐流/孤雁出群） | **新增** |
| 滚动回归 | 20日窗口 OLS 残差→t-stat（APM stat）；可由 rolling 矩组合或新算子 | 待加/组合 |

**reduce/L3 分界 = 可分解性**：能拆成"逐日量 + 跨日线性滚动"（σ、矩求和、均值）→ 留 L3；**不可分解的跨日池化**（聪明钱的 10 日累计成交量分位选择）→ 必须进 reducer（引擎用 warmup 支撑）。

---

## 7. 关键不变量（铁律——本会话验证 / 踩坑得出，违反即错）

1. **存原始、读时复权**：磁盘存不复权分钟 + 稀疏复权因子；价×ffill(cum_factor)、量/额原样。append-safe。
2. **append-only + 幂等 + 原子写**：tmp+os.replace；dedup(date,keep last)；重跑某日得相同值。
3. **warmup 由 Reducer 声明**：⚠️ 峰岭谷实测 **W1=2×std_window**（嵌套两层 rolling：标签σ窗 → 标签pooling窗）。每个 reducer 自报其嵌套深度，引擎据此取分块 overlap。**别全局硬编码。**
4. **增量前沿 = max(cache_last)**：⚠️ 退市股 cache_last 停在退市年（实测~6%在2005~2024），用 min 会被拖到 2005 误触发全量。退市股在新日无数据、load 不产出，无害。
5. **新股**：无缓存的新上市股，增量模式跳过（避免残缺尾部）；需按 listed_date 单独建（短）历史 —— **待实现**。
6. **universe（分钟）= all_instruments(CS)**（含退市，5551⊇缓存5505）；无 raw 数据股算子自动跳过。
7. **数据粒度 Reducer 声明**：原始APM半日频(4价/天)、模糊性分钟——引擎按需读，不强制 240 bar。

---

## 8. 增量日更编排（独立 L2 步骤，解耦"刷新"与"算因子"）

```
daily_update:
 1 ex_factors        复权因子增量          [已有]
 2 raw_ohlcv         日频增量              [已有]
 3 minute_ohlcv      分钟 raw 按日追加      [已建]
 4 ★ L2a/L2b 刷新     遍历所有注册 Reducer/cache_key 各增量一次(append新日) [待抽成独立步骤]
 5 L3 因子           spec 引擎(此时缓存已最新→cache-hit)              [增量化待做]
 6 labels            回填末 N+1 天          [待做]
```
当前 L2 刷新是"懒触发"（塞在因子算子第一步）；阶段3 抽成第 4 步独立入口 `refresh_all_supersets()`。

---

## 9. 验证方法论（每层上线前必做）

- **bit 对账**：新实现 vs 旧系统冻结产物（golden 缓存 `prv_v3__hc09d46528f`）逐列比对——计数/原始量列**精确相等**，close 派生列 ~1e-7(float32复权)，带符号量用绝对容差。
- **增量==全量**：截断缓存→增量追加 == 从头全量重算，逐值一致（period-agnostic，部分数据即可验）。
- 已验证：L2a 全链路 + 4 个复杂因子 raw/neu 全 bit 对齐（本会话）。

---

## 10. 附录：12 篇 → 模式 / 数据 / 新件 映射

| 研报 | 模式 | 数据 | reducer | 新件 |
|---|---|---|---|---|
| 峰岭谷(开源27) | ① | 分钟 | 峰岭谷(已建) | — |
| 跳跃度/飞蛾扑火(方正6) | ①+日频+截面 | 分钟+日高低 | 跳跃(待迁) | 截面翻转 |
| 潮汐(方正2) | ① 纯 | 分钟 | 新 | — |
| 勇攀高峰(方正3) | ① 纯 | 分钟OHLC | 新 | — |
| 模糊性/云开雾散(方正5) | ① 纯 | 分钟 | 新(嵌套vol) | — |
| 适度冒险(方正1) | ①+截面 | 分钟 | 新 | 截面均值(L2b) |
| 聪明钱2.0(开源3) | ①(池化) | 分钟 | 新(10日池化) | — |
| APM 原始/进阶(开源5+方正凤鸣朝阳) | ②+L3回归+截面 | 半日频/小时+**指数** | 分时收益(个股+指数) | L2b指数, 滚动回归 |
| 草木皆兵(方正8) | 日频+② | 日频+**中证全指** | (主体不用分钟) | 指数日收益 |
| 球队硬币(方正4) | 日频+截面 | **纯日频** | 不用分钟 | — |
| 水中行舟(方正9) | ③ 跨股 | 分钟+截面 | 高低额差(逐股) | **面板N²相关 + 分钟分化度(L2b)** |

→ 8/12 用 L2a 逐股 reducer（其中 4 篇纯①）；②市场参照被 4 篇需要；③跨股面板仅 1 篇但必须支持。

---

## 11. 落地路线（按依赖 + 风险递增）

1. **抽引擎**：把 L2a 引擎从 `minute_intraday_aggregate` 抽成 `MinuteAggregateEngine`，现有逻辑变 `PeakRidgeValleyReducer`。回归：bit 复刻 golden。
2. **迁 pricejump**：`minute_pricejump_aggregate` → `JumpReducer`（不再拷引擎）。bit 复刻其旧产物。
3. **新 Reducer**（纯①，零新基建）：潮汐 / 勇攀高峰 / 聪明钱 —— 验证"新研报=只写 reduce"。
4. **L2b**：指数分钟基建 + 截面 reducer 模式 → 跑 APM（验证②+滚动回归+cross_section_regress）。
5. **面板算子**：N² 相关 → 跑水中行舟（验证③）。
6. **编排**：独立 L2 刷新步骤 + daily_update 串通；L3 spec 增量。

---

## 12. 待决 / 局限

- **新股全史构建**（§7.5）：增量跳过，需 listed_date 短建分支。
- **缓存不一致 reconcile**（部分失败致活跃股落后 frontier）：当前 append-only 假设每次对全活跃股同步推进；reconcile 留阶段3+。
- **N² 面板相关性能**：水中行舟全市场两两相关，BLAS 相关矩阵 O(N²×T)，需评估内存/耗时（可分块）。
- **指数 instrument 复权口径**：指数本身不复权；分时收益用原始指数价即可。
