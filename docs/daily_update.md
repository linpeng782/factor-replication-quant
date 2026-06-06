# 日更流程 daily_update —— 运行手册

> 定位：**本地 Mac = 最小可执行参考实例**（全史数据已在本地）；**服务器 = scp 数据后跑同一套命令**。
> 一套命令两处通用。本文档先把流程理顺；标 `[待建]` 的是尚未实现、需补的件。
> 关联：`minute_incremental_design.md`(L1+L2 增量) / `hf_factor_factory_design.md`(工厂三层) §8。

---

## 0. 两处环境的关系

| | 本地 Mac | 服务器 |
|---|---|---|
| 数据根 | `/Users/didi/DATA`（`export FACTOR_REPL_DATA_ROOT=/Users/didi/DATA`） | scp 后的路径（设同名环境变量） |
| venv | `source /Users/didi/kdj/peterdidi/bin/activate` | 服务器对应 venv |
| 内存 | 受限 → `MINUTE_CHUNK_DAYS=30` 防 OEM | 大 → 可 `MINUTE_CHUNK_DAYS=250`（快） |
| 角色 | 跑通、验证、最小执行 | 全量 + 日常日更 |

**命令完全相同，只差环境变量/并发参数。**

---

## 1. 数据布局（输入 → 中间 → 产出）

```
market-data/
  daily/stock-ohlcv/<股>.parquet         日频原始(不复权)   [raw_ohlcv.py]
  daily/stock-ex-factors/<股>.parquet     稀疏复权因子        [ex_factors.py]
  minute/raw/<YYYY-MM-DD>.parquet         分钟原始(按日,不复权)[minute_ohlcv.py]  ← 大(76GB),需 scp
  industry/ market_cap/ masks/ labels/ prices/   中性化+评估输入 [industry/market_cap/labels...]
intermediate-cache/<cache_key>__h<hash>/<股>.parquet   L2 superset 缓存(可再生) [本仓引擎]
factors/{raw,cleaned,neu}/<source>/<group>/<因子>.parquet   L3 因子三阶段       [run.py]
```

**scp 到服务器**：必传 `market-data/minute/raw`（大）；其余 `market-data/*` 可 scp 或在服务器用数据线 `--full` 重建（小）；`intermediate-cache`（superset）可 scp 省去首次全量重建（与 minute 数据一致即有效），否则服务器首跑自动重建。

---

## 2. 完整流程（7 步，依赖序，任一失败即停）

> 数据线在 **stock-data-fetching 仓**；因子线在 **factor-repilcation-quant 仓**。全部默认**增量**。

### 数据线（cd stock-data-fetching；需 rqdatac 账号）
```bash
python ex_factors.py        # 1. 复权因子增量(查近30天)
python raw_ohlcv.py         # 2. 日频OHLCV增量(disk_max+1→今)
python minute_ohlcv.py      # 3. 分钟raw按日追加
python minute_ohlcv.py --full   # 3b.(幂等)补任何缺日 —— 见坑③
python industry.py          # 4. 中信行业面板增量(中性化用)
python market_cap.py        # 5. 总市值面板增量(中性化用)
```

### 因子线（cd factor-repilcation-quant）
```bash
python scripts/refresh_supersets.py            # 6. 刷新所有 superset 缓存   [待建,见§4①]
for f in <所有 minute factor spec>; do python run.py "$f"; done   # 7. L3 因子重算+评估
python ml/labels.py                            # 8. labels 回填(末N+1天)    [增量回填待确认]
```

> **隐式 fallback**：即使跳过第6步，第7步的 factor 算子第一件事就是 `engine.refresh_cache`（懒触发刷 superset）。
> 故"数据线 + 循环跑因子"已能自洽日更；独立第6步是为了**可控、可只刷数据不算因子**。

---

## 3. 当前【可跑】vs【待建】

| | 状态 |
|---|---|
| 数据线 1–5 | ✅ 全是增量 CLI，可直接跑 |
| L2 superset 增量（懒触发） | ✅ 引擎已实现 + bit 验证（append-only / 前沿=max / warmup overlap） |
| L3 因子 | ✅ `run.py <因子>` 可跑（cache-hit superset；**全量重算非增量**，但便宜） |
| **① 独立刷 superset 入口** | ❌ `scripts/refresh_supersets.py` 待建（扫 spec→去重 (action,cache_key,params)→各 refresh） |
| **② 编排器** | ❌ `scripts/daily_update.sh`（fail-fast 串 1–8）待建 |
| **③ 新股建库分支** | ❌ 见坑② —— **日更正确性必需** |
| labels 增量回填 | ⚠️ `ml/labels.py` 有构建函数，"末 N+1 天回填"待确认/补 |

---

## 4. 待建的 3 个件（规格）

**① `scripts/refresh_supersets.py`**：扫 `sources/*/*/specs/*/spec.yaml` 找 action∈REDUCER_BY_ACTION 的步 →
去重 unique `(action, cache_key, params)` → 每个 `REDUCER_BY_ACTION[action].from_step(step)` +
`MinuteAggregateEngine(reducer).refresh_cache(all_instruments(CS))`。**spec = 在用 superset 的唯一真相源。**

**② `scripts/daily_update.sh`**：`set -e` 串起 §2 全部步骤（跨两仓，cd 切换），逐步打印 + 失败即停。

**③ 新股建库分支**（坑②）：`refresh_cache` 当前对无缓存新股**跳过**；需检测"有 raw 数据但无缓存"的新股，
按其 `listed_date` 起单独全史（短）build。临时缓解：周期性删某 superset 目录后全量重建（覆盖新股）。

---

## 5. 三个坑（上服务器前必读）

1. **首次全量 superset 构建**：每个 reducer 首建要把全史分钟读一遍（服务器快；本地 prv 36列那次数十分钟）。
   日更稳态后才是秒级增量。可 scp `intermediate-cache` 跳过。
2. **⚠️ 新股不进 superset**：增量跳过无缓存新股 → 新上市股**持续漏出**因子覆盖。**必须补件③**，否则覆盖度递减。
3. **minute 缺日**：某天部分失败留缺口；增量后跑一次 `minute_ohlcv.py --full`（幂等补缺）兜底。

---

## 6. 验证一次日更是否成功

```bash
# a. raw 末日 = 最新交易日
ls market-data/minute/raw/[0-9]*.parquet | tail -1
# b. superset cache_last 已追到 raw 末日（抽1股）
python - <<'PY'
import pandas as pd, glob
d=sorted(glob.glob("<DATA>/intermediate-cache/prv_v3__*/000001.XSHE.parquet"))[0]
print(pd.read_parquet(d, columns=["date"])["date"].max())
PY
# c. 因子面板末日推进；spot IC 方向不变
```

---

## 7. 本地最小可执行 ↔ 服务器全量

- **本地**：全史数据已在（2005-01-04~2026-06-05），既可全量也可日更；内存约束 → `MINUTE_CHUNK_DAYS=30`。
  当前已验证：minute raw 完整、prv/tide/sm 三个 superset 可建、3 个因子复现。
- **服务器**：scp minute/raw（+ 可选 intermediate-cache）→ 设环境变量 → 跑同一套命令；
  `MINUTE_CHUNK_DAYS=250`、`MINUTE_WORKERS` 调大 → 全量首建快，之后日更。

> 落地顺序建议：先建 ①②（让本地一键跑通整条），再建 ③（新股，正确性必需），最后 labels 增量。
