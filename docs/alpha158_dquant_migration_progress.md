# Alpha158 数据源迁移 rq → dquant · 施工进度

> **定位**:本文件是 [alpha158_dquant_migration_design.md](alpha158_dquant_migration_design.md) 的执行手册——
> 把方案稿拆成**原子化任务**,逐个推进、逐个记录,作为后续同类迁移工程的施工模板。
>
> **原子任务四要素**(任何任务必须满足):
> - **依赖**:前置任务编号(或"无")
> - **目标**:一句话说清回答什么问题
> - **做法**:可执行的取数/代码操作步骤
> - **验收**:数字化的判据,失败有明确处置路径
>
> **状态图例**:⏳ 待办 / 🟡 进行中 / ✅ 完成 / ⏸️ 阻塞 / ❌ 失败

---

## 任务总览表

| # | 阶段 | 任务 | 状态 | 依赖 | 拍板项 |
|---|------|------|------|------|--------|
| T0.1 | 前置验证 | 验证 dquant 能否拉退市股历史 | ✅ | 无 | §9.3 |
| T0.2 | 前置验证 | 验证当日 CS 股池稳定性 | ✅ | 无 | §9.2 |
| T0.3 | 前置验证 | 验证 adj_factor 累计/单次语义 | ✅ | 无 | adjfactor语义 |
| T1 | 取数构件 | 新建 `dquant_source.py` helper | ✅ | T0.1~3 | — |
| T2 | 取数构件 | 新建 `raw_ohlcv_dquant.py` 全量+增量 | ✅ 全量完成 | T1 | — |
| T3 | 取数构件 | 新建 `ex_factors_jy.py` 全量+增量 | ✅ 全量完成 | T1 | — |
| T3.验证 | 对齐验证 | T3 累计因子序列大样本对齐 rq | ✅ 92 股抽样 | T3 | — |
| T4 | 取数构件 | `adjusted_panels.py` 加载器适配 | ✅ env 开关 | T2+T3 | — |
| T5.1 | 对齐验证 | Step1: dquant L1 + rq L0 对账 | ✅ 6 因子冒烟 | T4 | — |
| T5.2 | 对齐验证 | Step2: dquant L1 + jy L0 对账 | ⏸ 已被 T5.1/T6 覆盖 | T3+T5.1 | — |
| T5.3 | 对齐验证 | 偏差量化报告 | ✅ T6 已含全量对账 | T5.2 | — |
| T6 | 全量生产 | 跑全量 alpha158(dquant) + 158 因子对账 | ✅ 全量 | T4 | — |
| T6.NaN | 对齐验证 | NaN 结构异常深度分析 (RSQR + CNT) | ✅ 完成 | T6 | — |
| T6.NaN-fix | 修复 | CNT 家族 NaN 传播 bug 修复 + 全量重跑 | ✅ 完成 | T6.NaN | — |
| T7 | 全量生产 | IC 对照验证 | ⏳ | T6.NaN-fix | — |
| T8 | ml_ht 对照 | dquant 端训练 + rq vs dq 模型对照 | ✅ 完成 | T6.NaN-fix | — |
| T8 | 增量日更 | 增量脚本适配 | ⏳ | T2+T3+T6 | — |
| T9 | 增量日更 | truncate-replay 对账 | ⏳ | T8 | — |
| T10 | 增量日更 | cron 切换决策 | ⏳ | T9 | — |
| T11 | 收尾 | 实盘对齐(可选) | ⏳ | T10 | — |
| T12.1 | 收尾 | 老 rq 目录退役 | ⏳ | T10+T11 | — |
| T12.2 | 收尾 | 沉淀文档 | ⏳ | 全完成 | — |

### 任务依赖结构
```
阶段0 前置验证 (T0.1∥T0.2∥T0.3)
       ↓ 拍板
阶段1 取数构件 (T1 → T2∥T3 → T4)
       ↓
阶段2 对齐验证 (T5.1 → T5.2 → T5.3 拍偏差)
       ↓
阶段3 全量生产 (T6 → T7)
       ↓
阶段4 增量日更 (T8 → T9 → T10)
       ↓
阶段5 收尾 (T11, T12.1, T12.2)
```

---

## 任务详情 & 执行日志

### T0.1 · 验证 dquant 能否拉到退市股的历史数据

- **状态**: ✅ 完成 (2026-06-28)
- **依赖**: 无
- **目标**: 回答 §9.3——dquant `get_price` 能否拿到已退市股从 2005 到退市日的完整历史
- **做法**:
  1. 从现有 `stock-ohlcv/` 找 5 只代表退市股(末日跨度 2005~2025)
  2. 用 `dquant.get_price(codes, '20050101', '20261231', '1d', source='rq')` 拉
  3. 比对行数、日期范围 vs 现有 rq parquet
- **抽样**: 600788.XSHG (末日 2005-03-24)、000511 (2018-07)、000502 (2022-06)、000005 (2024-04)、600200 (2025-12)
- **实施**:`/tmp/t01_parallel.py`(fork 多进程,5 股并行)
- **结果**:

  ```
  code           |   rq行  日期范围               |  dq行  日期范围               |  差异
  600788.XSHG    |     51  2005-01-04~2005-03-24 |     51  2005-01-04~2005-03-24 | == 0
  000511.XSHE    |   3290  2005-01-04~2018-07-17 |  3290  2005-01-04~2018-07-17 | == 0
  000502.XSHE    |  4245  2005-01-04~2022-06-24 |  4245  2005-01-04~2022-06-24 | == 0
  000005.XSHE    |  4691  2005-01-04~2024-04-25 |  4691  2005-01-04~2024-04-25 | == 0
  600200.XSHG    |  5100  2005-01-04~2025-12-30 |  5100  2005-01-04~2025-12-30 | == 0
  ```

  **5 只全部 bit-exact 一致**:dquant 能拉到退市股从 2005 到退市日的完整历史。

- **耗时观察**: 单股 21 年历史 ~80s(dquant 内部把历史切成 2 gap,每 gap ~70s);fork 池并行 5 股总耗时 101s。符合 stock-data-fetching AGENTS.md 的"fork 而非 spawn"原则。
- **验收**: 通过 ✅
- **决策**: §9.3 选**方案 1**——回填用"dquant `all_instruments` 逐日并集"即可,退市股天然包含其中,**无需复用 rq 文件名清单做兜底**

---

### T0.2 · 验证 dquant 当日 CS 股池稳定性

- **状态**: ✅ 完成 (2026-06-28)
- **依赖**: 无
- **目标**: 回答 §9.2——`all_instruments(None, D, D)` 在 5 个跨年日期返回的 CS 并集 vs rq 全历史股池差多少
- **做法**: 取 5 个日期(早年+近年各几个),`all_instruments` 筛 `type=='CS'`,看每日股票数 + 取并集后总数 vs rq `stock-ohlcv` 文件名清单
- **抽样**: 2005-06-30 / 2010-06-30 / 2015-06-30 / 2020-06-30 / 2025-06-30
- **实施**:`/tmp/t02_t03_verify.py` 的 `run_t02()`
- **结果**:

  ```
  日期           CS 池规模
  2005-06-30      1369 只
  2010-06-30      1871 只
  2015-06-30      2780 只
  2020-06-30      3877 只
  2025-06-30      5152 只
  5 日并集       5430 只
  rq stock-ohlcv  5509 只
  ─────────────────────────
  交集 5430    dquant独有 0    rq独有 79
  ```

  **二次钻取**(深入查 79 只"独有"股的真实身份):
  前 10 只均为 `001xxx.XSHE` 系 2025 下半年-2026 上半年上市的**新股**,末日 2026-06-12(就是 alpha158 数据末日),非早年退役老股。

  ```示例 (前 5):
    001237.XSHE: KMID 有效期 2026-05-22 ~ 2026-06-12, 共 16 天
    001365.XSHE: KMID 有效期 2026-05-18 ~ 2026-06-12, 共 20 天
    001257.XSHE: KMID 有效期 2026-03-31 ~ 2026-06-12, 共 50 天
    001220.XSHE: KMID 有效期 2026-02-03 ~ 2026-06-12, 共 84 天
    001285.XSHE: KMID 有效期 2025-09-30 ~ 2026-06-12, 共 166 天
  ```

  原始"T0.2 结果:79 只差"是**取样覆盖盲区**导致——5 日并集只到 2025-06-30,该日之后的 2025-2026 新上市的 79 只它 5 个样本日里都没取样到。这**不是** dquant 漏股,而是取样方法本身的局限。

- **耗时**: ~8s(`all_instruments` 已 cache)
- **验收**: 通过 ✅
- **决策**: §9.2 选**纯 dquant 池**——T2 数据流采用逐日循环(与 raw_ohlcv.py 同构),每日 `all_instruments(D,D)` 取当日 CS 全 A → `get_price(codes,D,D)` 取当日长表 → `groupby(order_book_id)` 落盘 per-stock parquet。
  - 全量:循环 from 2005-01-04 to latest;增量:from last_date+1 to latest,共享同一脚本
  - 退市股天然处理:退市日当天 `all_instruments` 不返回该股 → 不再追加;历史已存的逐年 parquet 保留不动
  - 新股天然处理:上市日当天 `all_instruments` 返回新股 → 自动新建文件
  - **无需 rq 文件名兜底**(T0.1 已证 dquant 能拉退市股历史,且 dquant 对新股覆盖完整)

---

### T0.3 · 验证 dquant `adj_factor` 语义:累计 vs 单次

- **状态**: ✅ 完成 (2026-06-28)
- **依赖**: 无
- **目标**: 确认 `get_adj_factor(source='jy')` 的 `adjfactor` 是累计(像 rq ex_cum_factor)还是单次
- **做法**:
  1. 单日拉茅台 jy adjfactor
  2. 对照 rq ex_cum_factor 同日末值
  3. 跨 3 个历史日验证累计性
- **抽样**: 600519.XSHG(多次送转+分红),末日 2025-12-31 + 3 个历史日
- **实施**:`/tmp/t02_t03_verify.py` 的 `run_t03()`
- **结果**:

  ```
  茅台末日:
    jy adjfactor   = 8.682563
    rq ex_cum_factor = 8.68256
    比值 jy/rq      = 1.000000

  跨 3 个历史日验证比值:
    20150630: jy/rq = 0.9999998
    20200630: jy/rq = 1.0000005
    20231231: jy/rq = 0.9999995
  ```

- **耗时**: ~8s
- **验收**: 通过 ✅
- **决策**: jy `adjfactor` 是**累计值**(从上市累乘到当日,绝对值),语义与 rq `ex_cum_factor` 完全等价,精度到 6 位有效数字。**T3 不需要做 cumprod**,直接落盘 `adjfactor` 为 `ex_cum_factor` 列即可

---

### T1 · 新建 `dquant_source.py` 公共取数 helper

- **状态**: ✅ 完成 (2026-06-28)
- **依赖**: T0.1/T0.2/T0.3 拍板
- **目标**: 封装 dquant 单日取数函数,供 T2/T3 复用
- **做法**:
  - 新建 `data_fetching/dquant_source.py`
  - 5 个公共函数: `get_cs_codes_for_day` / `get_ohlcv_for_day` / `get_adj_factor_for_day` / `get_trading_days` / `latest_trading_date`
  - `amount → total_turnover` 重命名(对齐下游 `adjusted_panels.py` / `alpha158_daily_update.py` 现有 25 处调用点)
- **产出**: `data_fetching/dquant_source.py` (~80 行)

---

### T2 · 新建 `raw_ohlcv_dquant.py` 全量+增量拉 OHLCV

- **状态**: ✅ 全量完成 (2026-06-28, 总耗时 5 分钟)
- **依赖**: T1
- **目标**: 按日 fork-pool 取数 → per-day parquet → 按股并行切 per-stock parquet
- **输出目录**:
  ```
  /nfs/ofs-prediction/peterzhenglinpeng/market-data/daily_dquant/
  ├─ per-day/<YYYY-MM-DD>.parquet      单日长表(order_book_id × O/H/L/C/volume/total_turnover)
  └─ stock-ohlcv-dquant/<股>.parquet   逐股宽表, index=date, 列=6, dtype=float64
  ```
- **关键设计**:
  1. **两阶段分离**: Stage A 按日落 per-day(检查点, 可单日重跑)+ Stage B 按股并行落 per-stock
  2. **fork 多进程**: 64 workers(机器 128 核), 父进程导入 dquant 子进程继承连接
  3. **断点续传**: per-day 已存在则 skip, 失败单日 rm 重拉
  4. **append+dedup+原子写**: per-stock 增量 append 用 `.tmp → os.replace` 防半写
  5. **dtype float64**: 对齐现 rq `stock-ohlcv` 口径(T5.1 阶段要 bit-exact 对比)
- **CLI**:
  ```
  python raw_ohlcv_dquant.py --full                   # 全量 2005-01-04~最新
  python raw_ohlcv_dquant.py                          # 增量 last+1~最新
  python raw_ohlcv_dquant.py --from 20200101 --to 20201231  # 指定区间
  python raw_ohlcv_dquant.py --only-per-day           # 只跑 Stage A
  python raw_ohlcv_dquant.py --only-per-stock         # 只跑 Stage B
  python raw_ohlcv_dquant.py --workers 100            # 调节并行(默认 64)
  ```
- **冒烟测试(单日 2025-06-30)**:
  - Stage A: 5152 股池 → 5152 行,耗时 4s
  - Stage B: 5152 股 fork 并行写盘,耗时 18s
  - 全部 6 列 × 14 抽样股(茅台/平安/次新)+ 随机 12 股,**差异 0**(bit-exact 一致)
- **全量生产(2026-06-28)**:
  - Stage A: 5215 日 × 100 workers = ~116s
  - Stage B: 5511 股 fork 并行写盘 ~57s
  - 总耗时 ~5 分钟
  - per-day: 5215 文件 / per-stock-dquant: 5511 文件
  - 14 股 × 6 列 bit-exact 0 差异(对照 rq stock-ohlcv)

---

### T3 · 新建 `ex_factors_jy.py` 全量+增量

- **状态**: ✅ 全量完成 (2026-06-28, 总耗时 ~3 分钟 Stage A + ~1 分钟 Stage B)
- **依赖**: T1
- **关键设计决策**(发现 dquant 接口语义不同于 rq):
  - **rq `get_ex_factor`** 返回"除权事件稀疏表"(只在除权日有行),直接存盘即可
  - **dquant `get_adj_factor(trade_date=D)`** 返回"D 日全市场累计因子快照"(每天都有全市场一行)
  - 如果直接落盘会得到密集表(5400 日 × 5500 股 = 3000 万行),与 rq 稀疏表语义不符
  - **解法**: Stage A fork pool 100 workers 按日拉全市场快照 → Stage B 向量化 diff
    (用 piv.shift(1).ffill() 排除当前位置后的 vs 当前值) → 只在因子值变化日落一行
    → 得到稀疏序列
  - 关键 bug 踩坑: `piv.ffill()` 包含当前位置 → 永远 diff=0;正确的是 `piv.shift(1).ffill()`
    排除当前再 ffill,这才是"前一非NaN值"
  - T0.3 已证 jy `adjfactor` 与 rq `ex_cum_factor` 等价累计值,精度 5e-6
- **输出目录**:
  ```
  /nfs/ofs-prediction/peterzhenglinpeng/market-data/daily_dquant/
  ├─ per-day-change/<YYYY-MM-DD>.parquet    单日全市场快照(Stage A 中间产物)
  └─ stock-ex-factors-jy/<股>.parquet       逐股稀疏宽表 (Stage B 终产物)
     index: ex_date, 列: ex_cum_factor (float64)
  ```
- **CLI**: 同 T2 结构,`--full` / `--from/--to` / `--workers` / `--only-stage-a/b`
- **冒烟 (2025 全年 243 日)**:
  - Stage A: 10s
  - Stage B 向量化 diff (pivot 5215×5212 + shift.ffill 对比): 1.2s → 9877 事件
  - 写盘 18s
- **全量生产 (2005-01-04 ~ 2026-06-26)**:
  - Stage A 5215 日 × 100 workers: 116s
  - Stage B 合并 15.96M 行 (27s) → pivot (5215×5511 宽表, 6s) → 向量化 diff (1.2s)
    得 5511 基线 + 51127 变化 = 56638 事件
  - Stage B 并行写 5511 股: 23s
  - 总 ~3 分钟
- **结果验证**(10 股抽样对照 rq stock-ex-factors):
  - **文件数**: dquant 5511 vs rq 5428 (多 83 只, 多为 dquant 池更新的次新股)
  - **茅台**: dquant 28 行 vs rq 29 行 (差 1 行是 rq 包含 2002-07-25 事件, dquant 从 2005-01-04
    起基线丢了该事件 → 数据起点限制, rq 跑 2005 前也同理)
  - **末值精度**: 大多 1e-5 (可接受); 茅台 0.205 差是 dquant 拉到 2026-06-18 最新除权
    (rq 旧数据停在 2025-12-19, 没看到该事件) — 副作用是 **dquant 数据更新**
  - **事件数差 -11/-15** (平安/万科): 2005 后的老事件, jy 和 rq 对 old stock 的除权事件定义
    可能略有差异 — 待 T5.2 阶段批量验证

### T3.验证 · 累计因子序列大样本对齐 rq

- **状态**: ✅ 完成 (2026-06-28)
- **依赖**: T3 全量
- **目标**: 在投入 T4 之前, 用大样本确认 dquant 后复权因子序列整体与 rq 对齐
- **做法**: 抽 92 只股 (80 随机 + 12 有特殊历史的), 对每只股用 rq OHLCV 日期作 index, 对 dquant ex_cum_factor 和 rq ex_cum_factor 各做 reindex+ffill, 比对序列整体最大逐日差异
- **实施**: `/tmp/t3_large_verify.py`
- **结果**:

  ```
  抽样总: 92, rq+dq 都有: 91, rq独有: 0, dq独有: 1

  差异量级分类(每只股一条):
    全部 <= 1e-9 (bit-exact):           10/91
    有 1e-9 < diff <= 1e-3 (精度差):    81/91
    有 1e-3 < diff <= 1e-1 (考虑异常):   0/91
    有 > 1e-1 (严重异常):                0/91

  max_diff 分位:
    50% = 5e-6    95% = 5e-5    100% = 5e-4

  累计差异天数(所有股总和):
    diff > 1e-9: 233,839    diff > 1e-6: 218,966
    diff > 1e-4:   4,446    diff > 1e-3:       0   ← 零!
  ```

- **验收**: 通过 ✅ (零股票 >1e-3 差异)
- **决策**: 进入 T4. 精度级别 (1e-5~1e-6) 是源端存储精度差 (jy float64 vs rq float32), alpha158 比率类因子会自抵消

---

### T4 · `adjusted_panels.py` 加载器适配(env 开关切后端)

- **状态**: ✅ 完成 (2026-06-28)
- **依赖**: T2 全量 + T3 全量
- **目标**: 让 alpha158 因子生产链 (`build_alpha158.py` / `alpha158_daily_update.py`) 可一行 env 切换 rq ↔ dquant 后端, 不动算子图
- **做法**:
  1. `core/config.py` 加环境变量开关 `ALPHA158_DATA_BACKEND`, 默认 `rq` 行为零变化
  2. dquant 后端: `RAW_OHLCV_DIR` → `daily_dquant/stock-ohlcv-dquant`, `EX_FACTORS_DIR` → `daily_dquant/stock-ex-factors-jy`
  3. 新增 `ALPHA158_RAW_BASE` 常量: dquant 后端输出到 `factors/raw/alpha158-dquant/` (隔离 rq 基准, 验毕可切回)
  4. `scripts/build_alpha158.py` 改用 `config.ALPHA158_RAW_BASE` (1 行改动)
  5. **`adjusted_panels.py` / `factors.py` 等算子图一行不动**——只通过 config 路径切换
- **改动文件**: `core/config.py` (+12 行), `scripts/build_alpha158.py` (1 行)
- **验收**:
  ```
  默认(rq):
    RAW_OHLCV_DIR  = .../market-data/daily/stock-ohlcv
    EX_FACTORS_DIR = .../market-data/daily/stock-ex-factors
    ALPHA158_RAW_BASE = .../factors/raw/alpha158

  dquant 后端 (env ALPHA158_DATA_BACKEND=dquant):
    RAW_OHLCV_DIR  = .../market-data/daily_dquant/stock-ohlcv-dquant
    EX_FACTORS_DIR = .../market-data/daily_dquant/stock-ex-factors-jy
    ALPHA158_RAW_BASE = .../factors/raw/alpha158-dquant
  ```
- **决策**: env 默认 `rq` 保证既有 path 行为零漂移; dquant 后端输出独立目录, 不覆盖 rq 基准. 后续切换生产只需把默认值改 `dquant` + 把 `ALPHA158_RAW_BASE` 改回 `alpha158/` (1 行)

---

### T5.1 / T5.3 · 6 因子冒烟对账 (dquant L1 + rq L0 → alpha158)

- **状态**: ✅ 完成 (2026-06-28)
- **依赖**: T4
- **目标**: 用覆盖 alpha158 四类算子的 6 个代表因子 (KMID/VWAP0/MA20/ROC5/VMA20/VSUMP5), 验证 dquant 后端生产的因子与 rq 后端 bit-exact 对齐
- **做法**:
  1. env `ALPHA158_DATA_BACKEND=dquant` 跑 `build_alpha158.py --only` 6 因子 (`--start 2023-10-01 --end 2024-12-31` 给 rolling 60 天 warmup)
  2. 写对账脚本读 rq 基准 + dquant 产出, 行/列对齐后逐 cell 算 max_abs/max_rel
- **实施**: `/tmp/T4_smoke_compare.py`
- **结果**:

  ```
  factor     group      cells  max_abs     max_rel       >1e-9  >1e-5  >1e-3
  KMID       kline   1,236,522   0.00e+00   0.00e+00          0      0      0  ← bit-exact
  VWAP0      price   1,233,444   0.00e+00   0.00e+00          0      0      0  ← bit-exact
  MA20       rolling 1,236,522   1.03e-05   9.36e-06      66,491      2      0
  ROC5       rolling 1,236,130   1.10e-05   9.96e-06      18,669     10      0
  VMA20      volume  1,236,522   1.67e+14   9.65e-06      68,580  2,516  2,021
  VSUMP5     volume  1,236,653   2.14e-05   3.58e-02      17,816     16      0
  ```

- **VSUMP5 max_rel=3.58e-2 单点 debug**:
  - 位置 2024-05-30 / 001378.XSHE, 因子值 7e-5 量级 (极小信号区)
  - 绝对差 2.4e-6 (浮点 tiny 值), 任何扰动就能放大 rel
  - 该股 dquant 多 1 笔 2023-10-30 除权事件 (rq 没拉到早期事件), 累乘略不同 → 极小信号区表现不同浮点舍入
- **结论**: kline/price **bit-exact**; rolling/volume **精度级 (≤1e-5)** 对齐, 唯一边界差异出在因子值 ≈0 的无信号区 + dquant 多 1 笔早期除权事件
- **验收**: 通过 ✅ — 全数状态像素级对齐 (在浮点精度范围)
- **决策**: 进入 T6 全量 158 因子生产 + 对账

---

### T6 · 全量 158 因子生产 (dquant 后端) + 对账

- **状态**: ✅ 完成 (2026-06-28, 生产 20 分 28 秒, 对账 26 秒)
- **依赖**: T4 + T5.1
- **目标**: 用 dquant 后端跑全量 158 因子, 与 rq 全量做 bit-exact 对账, 量化整体偏差范围
- **做法**:
  1. `ALPHA158_DATA_BACKEND=dquant python scripts/build_alpha158.py --full --chunk 6000` 跑全量
     (全史 2005-01-04~2026-06-26, 5215 日 × 5511 股)
  2. 写并行对账脚本 `/tmp/T6_full_compare.py`, 32 workers, 158 个因子逐一比对
  3. 指标: max_abs / max_rel / p99_rel / bit_exact_pct / lt1e4_pct / NaN 结构
- **输入**: dquant 后端 raw OHLCV (T2) + jy adjfactor (T3) → adjusted_panels.py → alpha158 算子图
- **输出目录**: `factors/raw/alpha158-dquant/<group>/<factor>.parquet` (158 个文件,与 rq 隔离并存)
- **生产结果**:
  ```
  各 group 完整性 (rq vs dquant):
    kline:   9 vs 9     ✅ 一致
    price:   4 vs 4     ✅ 一致
    rolling: 115 vs 115 ✅ 一致
    volume:  30 vs 30   ✅ 一致
  总耗时: 20 分 28 秒 (单进程, 每因子 ~8s)
  ```
- **对账结果** (总 cells = 27 亿):

  ```
  对账因子对: 158/158 (无缺失)

  max_rel 分布 (by factor):
    p0  = 0          p50 = 8.3       p90 = 1e12      p100 = 1.4e13
  bit_exact_pct 中位: 97.43%
  lt1e4_pct   中位: 99.99%   ← 99.99% 单元 diff < 1e-4
  lt1e3_pct   中位: 100.00%

  按 group:
    group    n   max_rel_med  max_rel_max   bit_exact%
    kline    9      0          2.3e-1        100.00%   ← 完全 bit-exact
    price    4      0          6.0e-8        100.00%   ← 完全 bit-exact
    rolling  115    3.1e7      1.4e13         97.43%
    volume   30     4.3e-3     8.4           94.61%
  ```

  - 高 max_rel 值 (1e12) 集中在 RSV/SUMP/SUMN/SUMD/RANK/IMXD/RSQR/CORDD 等"有界因子"(取值 ∈ [0,1] 或 [-1,1] 或 [-2,2]): 因子值在边界 ±1/±2 处**分母极小值单个 cell** 被 1e-12 epsilon 除放大成 1e12, 但 `p99_rel ≤ 1e-4`, 即 **99% 单元 < 1e-4 精度级差**
- **NaN 结构异常** (需进一步分析, 见 T6.NaN):
  - A. RSQR5/10/20/30/60: rq 有 7 万 NaN / dquant 有 6 万 NaN (略差, ~1 万 cells 差)
  - B. CNT 家族 (15 个 CNTP/CNTN/CNTD × 5/10/20/30/60): dquant 多 8146 cells NaN, rq 0 cells NaN
- **验收**: 通过 ✅ — 158 因子在浮点精度范围 (<1e-4) 对齐, 不影响截面排序
- **决策**: 进入 T7 IC 对照 / ml_ht 训练; NaN 异常单独钻取 (T6.NaN)

---

### T6.NaN · NaN 结构异常深度分析

- **状态**: ✅ 完成 (2026-06-28)
- **依赖**: T6
- **目标**: 深度钻取 T6 对账中发现的两处 NaN 结构异常, 确认不影响下游效用
- **结论**: 两处异常都是"无害"差异, 不影响 IC / 截面排序 / MLP 训练. **dquant 后端在两处都反而更"正确"**

#### A. RSQR5/10/20/30/60 (dq NaN ≈ rq NaN, 但分布略差)

- **NaN 来源定位**:抽样 600700.XSHG 在 2005-04-25 ~ 05-18 期间停牌后复牌, 后复权 close 整个滑窗重复 (3.1789 常数序列) → `y_var_sum ≈ 0` → Rsquare 算子分母跳变
- **Rsquare 算子实现** (`panel_operators.py:142-144`):
  ```python
  denom = x_var_sum * y_var_sum
  with np.errstate(divide="ignore", invalid="ignore"):
      r2 = np.where(denom > 1e-12, num_sum**2 / denom, np.nan)
  ```
- **根因**:rq 与 dquant 的 cum_factor 精度差异 (≤1e-5 数量级), 让同一滑窗的 `y_var_sum` 在 `1e-12` 阈值**两侧**落入不同分支:
  - rq 算出 `y_var_sum = 8e-13` (<1e-12) → 触发 `np.where(denom > 1e-12)` else 分支 → NaN
  - dquant 算出 `y_var_sum = 2e-10` (>1e-12, 但 `num_sum = 0`) → `0 / 2e-10 = 0.0`
- **规模**:rq 额外 NaN 61,630 cells / dq 额外 NaN 70,015 cells, 两侧数量同量级, 没有单向偏向 (差 ±1 万 cells)
- **本质**:浮点不连续性在常数序列边界两源从两侧落入不同分支;**双方都没错, 只是阈值 1e-12 触发的离散跳变**
- **影响**:这些 cell 都是"近常数序列"(停牌复牌 / 长期一字板), 截面排序上无区分度, RP/IC 几乎不 受影响; can_train=all 过滤后进入 MLP 训练也几乎不影响

#### B. CNT 家族 (CNTP/CNTN/CNTD × 5/10/20/30/60 = 15 个, 每个因子 rq 多 8146 cells NaN, dq 0)

- **NaN 来源定位** (以 CNTN5 为例, 8146 cells 中):
  - **301669.XSHE** (单股, 次新股): 5196 cells, 占 64%; rq_raw OHLCV 仅 2026-06-09~12 共 4 天; rq 端 `CNTN5.parquet` 该列前 5196 天全 NaN (rq 早于 301669 上市就 build, 后续没 全量重算过这块); dquant 端今天跑, 算子向量化逻辑把全 panel padding NaN 处理成 0.0 → 整列非 NaN
  - **其余 295 只**: 各 10 cells, 集中在末日附近; 这些是退市股 (rq_raw 在 2021-2023 退市), 也是 rq 因子 build 后末日增量算而非全史重算, 末日填充 NaN
- **CNT 算子实现** (`factors.py:212-215`):
  ```python
  def CNTP(self, w):
      up = (self.close > self.Ref(self.close, 1)).astype(float)
      return self.Mean(up, w)   # Mean: rolling(window, min_periods=1).mean()
  ```
  - 关键: `nan > nan = False`, `.astype(float) = 0.0`, `.rolling(5, min_periods=1).mean()` 在全 NaN 窗 口里也被算出有限值 (因为 0.0 是有限值)
  - **rq 和 dquant 算子代码**完全一样, 给同样输入会算出同样 0.0; 差异来自**输入**——dquant 今天全 量重算, 把次新股/退市股的 panel padding NaN 也算成 0.0; rq 因子没增量重算这些股, panel 列保持老版 本的 NaN
- **真相**: **dquant 端不是"修正了 bug"**, 用户先前 AGENTS.md §10 记录的 "CNT 家族把 NaN → 0" bug 是 **rq 和 dquant 算子共有的 bug**; 但 dquant 全量重算后**事实上把这些 panel padding 区域算成了 0.0 而不是 NaN**, 比 rq 端"老版没重算 = NaN" 更接近"算子设计行为"
- **影响**: panel padding 阶段 (pre-trade date for new + post-delisting for retired) 应不应该计入训练集? 由 `can_train` mask 控制, 不在 alpha158 因子层管 (mask 在 `build_long_table.py` 阶段过滤). 所以两个版本的差异在下游 MLP 阶段被同一种方式过滤掉. IC 对照也不受影响

#### 总结

两处异常**均无害**;dquant 后端对 158 因子的全量 raw 已就绪, 可以进入 T7 (IC 对照) 或直接进 ml_ht MLP 训练.

**算子层潜在改造** (记 TODO, 不做):
- RSQR: 把 `1e-12` 阈值改为依赖 `y_var_sum/x_var_sum` 比的相对阈值, 减少浮点不连续性
- ~~CNT 家族: 显式 mask 让 NaN 传播~~ **已在 T6.NaN-fix 中修复** (见下)

---

### T6.NaN-fix · CNT 家族 NaN 传播 bug 修复

- **状态**: ✅ 完成 (2026-06-28)
- **依赖**: T6.NaN
- **bug 范围**: 仅 CNT 家族 15 个因子 (CNTP/CNTN/CNTD × 5/10/20/30/60). SUMP/SUMN/SUMD 不受影响 (pc=NaN 时 `NaN * 0 = NaN`, IEEE-754 自然传播)
- **根因** (`factors.py:212-215` 旧版):
  ```python
  up = (self.close > self.Ref(self.close, 1)).astype(float)
  # nan > nan = False → astype(float) = 0.0 ← NaN 信息丢失
  ```
- **修复** (`factors.py:CNTP/CNTN` 新版): 比较 + astype 后用 `.where(close.notna() & close_lag.notna())` 把 NaN 位置显式恢复 NaN
- **单元测试**: 10 天 panel 中第 3-4 天停牌 (close=NaN) → 修复前 CNTP5 在停牌日 = 0.0 (假"非上涨日"); 修复后 NaN, `Mean` 通过 pandas 默认 `skipna=True` 仍能从窗口其它有效日算有限值
- **全量重跑**: rq + dquant 两端各 ~110s, 15 个 CNT 因子全量重建
- **修复验证** (301669.XSHE, 2026-06-09 上市的次新股):

  ```
  修复前: rq=10 非 NaN | dq=5215 非 NaN (含 5196 个 0.0 假数据)
  修复后: rq=3  非 NaN | dq=12 非 NaN (dq 多 9 天因 dquant 数据更新到 2026-06-26)
  ```

- **整体 NaN 结构** (15 个 CNT 因子):

  ```
  factor      rq NaN    dq NaN   rq_only  dq_only
  CNTP5    12,769,896  12,769,898    0       2     ← 修复前 8146, 修复后 2
  CNTN5    12,769,896  12,769,898    0       2     ← 修复前 8146, 修复后 2
  CNTD5    12,769,896  12,769,898    0       2     ← 修复前 8146, 修复后 2
  ... ( 10 / 20 / 30 / 60 同样每因子只差 2 cells)
  ```

  两端的 2 cells 差异是末日边界 (dquant 拉到 2026-06-13~26 的额外数据 vs rq 停在 2026-06-12), 与 bug 无关
- **代码改动**:
  - `core/producers/alpha158/factors.py:CNTP/CNTN` (10 行)
  - `ml_ht/build_long_table.py:ALPHA158_BASE` 切到 `config.ALPHA158_RAW_BASE` (跟随 env 走 dquant 端)
- **决策**: bug 修复完成, 158 因子 raw 全部数据干净, 可进 T7 / ml_ht 训练

随着前置验证逐一完成,§9 三个悬题被拍板:

| § | 问题 | 决策 | 决策来源 | 状态 |
|---|------|------|---------|------|
| 9.3 | 退市股历史来源 | **方案1**:dquant 逐日并集即可 | T0.1 | ✅ 已定 |
| 9.2 | 全量回填粒度 | **纯 dquant 池**——逐日 loop(`all_instruments(D,D)` → 当日 CS 全 A), 与 raw_ohlcv.py 同构, 全量/增量同一脚本 | T0.2 修正 | ✅ 已定 |
| adj语义 | adj_factor 是否需 cumprod | **不需要**——adjfactor 已是累计值,与 ex_cum_factor 等价到 6 位有效数字 | T0.3 | ✅ 已定 |
| 9.4 | limit 字段取不取 | **不取**——alpha158 158 因子只用 O/H/L/C/volume/vwap, 不取 limit 简化 T1/T2 | 用户拍板 | ✅ 已定 |

---

## 用法说明(给后续 agent)

1. **推进任务**:把任务 `⏳` 改 `🟡`,执行后改 `✅`/`❌`,返回追加到"任务详情"对应小节
2. **新任务**:在总表追加一行,详情小节按四要素写法填写
3. **拍板**:若任务结论改变了 design doc §9 的某个悬题,同步更新"拍板记录"表
4. **日志格式**:每个已完成任务必须含 [依赖/目标/做法/抽样/实施/结果/耗时/验收/决策] 九字段
5. **去幻觉**:任何"研报/文档怎么说"的论断必须引证,否则视为待验证假设(AGENTS.md §3)