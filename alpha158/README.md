# alpha158 —— 因子生产线总览（看这一份就够）

> alpha158 = 158 个量价技术因子（9 K线 + 4 价格 + 23×5 rolling + 6×5 量）。
> 本目录收敛 alpha158 的**全部操作脚本**（业务全量线 + 增量日更线 + 对账 + 出图）。
> **算子引擎**在 `alpha158/engine/`（本目录子目录，自包含）。

---

## 1. 目录里有什么（本目录 = 操作入口）

| 文件 | 作用 | 频率 |
|---|---|---|
| `build.py` | **全量生产**：读后复权面板→算 158 因子→落盘（WIDE，按股分块内存安全）。周期性对账/重置基线用 | 罕见 |
| `daily_update.py` | **增量日更**：读旧面板末日→只算 (last, T] 新交易日→逐因子 append+dedup+原子写。零 API | 每交易日 |
| `smoke_replay.py` | **增量正确性对账**：truncate-replay（砍尾重放 vs 全量），内核与 `daily_update.py` 共享 | 改动后 |
| `verify_repro.py` | **复现校验**：重算 vs 参考面板逐格比（max_rel / 相关 / 容差匹配率） | 改动后 |
| `plot.py` | **评估出图**：直读 cleaned/neu → IC/分层/单调 + 2×2 报告 PNG | 按需 |

引擎：`alpha158/engine/`
- `factors.py` 158 因子定义 · `groups.py` 分组(kline/price/rolling/volume) · `panel_operators.py` 宽表算子。
- 后复权加载：`core/data/adjusted_panels.py`（通用基础设施，非 alpha158 专属，被 `core/operators/industry_co_momentum.py` 共享）。

---

## 2. 一条线的全貌（数据 → 因子 → 消费）

```
数据线（data_fetching/，拉原始 OHLCV + 复权因子）
   dquant: raw_ohlcv_dquant.py + ex_factors_jy.py → market-data/daily_dquant/{stock-ohlcv-dquant, stock-ex-factors-jy}
   rq    : raw_ohlcv.py       + ex_factors.py     → market-data/daily/{stock-ohlcv, stock-ex-factors}
        │  读时后复权（价×ffill(cum)，量÷cum）：core/data/adjusted_panels.py
        ▼
引擎 alpha158/engine/  →  build.py（全量）/ daily_update.py（增量）
        ▼
因子面板 config.ALPHA158_RAW_BASE/<group>/<factor>.parquet
   dquant → factors/raw/alpha158-dquant/     rq → factors/raw/alpha158/
        ▼
消费：ml_core / ml_ht / ml 通过 discover_features(sources=['alpha158-dquant']) 读取
```

---

## 3. ⚠️ 后端开关（今天踩坑点，务必理解）

alpha158 的**源目录 + 产出目录**由 `ALPHA158_DATA_BACKEND` 决定（缺省继承主开关 `DATA_BACKEND=dquant`）：

| env | 读源 `RAW_OHLCV_DIR` / `EX_FACTORS_DIR` | 写产出 `ALPHA158_RAW_BASE` |
|---|---|---|
| `dquant`（默认） | daily_dquant/{stock-ohlcv-dquant, stock-ex-factors-jy} | **factors/raw/alpha158-dquant/** |
| `rq` | daily/{stock-ohlcv, stock-ex-factors} | factors/raw/alpha158/ |

- **生产模型（lgbm_a158_p27_top64 等）读的是 `alpha158-dquant`** → 用默认 dquant env 即可。
- `build.py` / `daily_update.py` 都用 `config.ALPHA158_RAW_BASE`（后端感知，两端各写各目录，互不覆盖）。
  历史 bug：`daily_update.py` 曾写死 `RAW_FACTOR_BASE/"alpha158"`（rq 目录），导致读 dquant 源却写 rq 目录 —— 已修为 `ALPHA158_RAW_BASE`。
- 与分钟因子的 `MINUTE_DATA_BACKEND` **是两条独立轴**：alpha158 只认 `ALPHA158_DATA_BACKEND`。

---

## 4. 日常怎么用（命令，仓库根执行，先 `source venv`）

```bash
# 默认 dquant 后端（生产）；上游数据线须先更新到目标日（见 data_fetching/DAILY_UPDATE_GUIDE.md A 段）
PYTHONPATH=. python alpha158/daily_update.py        # 增量日更：自动补 (基线末日, 源末日]
PYTHONPATH=. python alpha158/build.py --full        # 全量重建（罕见：重置基线/周期对账）
PYTHONPATH=. python alpha158/build.py --only KMID MA20   # 只算指定因子
PYTHONPATH=. python alpha158/smoke_replay.py --all  # 增量正确性对账（改动增量内核后跑）
PYTHONPATH=. python alpha158/verify_repro.py        # 复现校验
PYTHONPATH=. python alpha158/plot.py --workers 4    # 出评估图

# rq 后端（回退/对比基线）：
ALPHA158_DATA_BACKEND=rq PYTHONPATH=. python alpha158/daily_update.py
```

> `daily_update.py` 在日更流程中由 agent 按 `data_fetching/DAILY_UPDATE_GUIDE.md` B 段调用（零 API，失败即停）。

---

## 5. 相关设计文档（在 docs/）

- `docs/alpha158_incremental_design.md` —— 增量算法 + truncate-replay 对账设计
- `docs/alpha158_dquant_migration_design.md` / `_progress.md` —— rq→dquant 后端迁移设计与进度
- `docs/server_daily_production.md` —— 服务器日更全链路运行手册（因子→掩码→信号→回测）
