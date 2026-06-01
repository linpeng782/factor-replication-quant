# 分钟级因子复现指南

> 高频研报（开源_微观_27、方正适度冒险等）的分钟数据 → 日频因子的实现细节。
> 从 AGENTS.md §7 拆出，AGENTS.md 仅留指针。

---

## 1. 数据 & 股票池

**分钟数据**：`MINUTE_DATA_DIR`（本机 `<DATA_ROOT>/market-data/minute/stock_data_1m_post/`）
- per-stock parquet（`<order_book_id>.parquet`），列 `[datetime, open, high, low, close, volume, total_turnover]`
- **后复权 1m**：价格 ×`ex_cum_factor`、**量/额不复权**；**2005-01-04 至今**，5455+ 只股票
- 拉取（`stock-data-fetching/minute_ohlcv.py`）：`get_price(adjust_type='none')` → 价格 ×本地 `stock-ex-factors.ex_cum_factor`(ffill) → 量/额原样 → float32。已对 5 只股逐分钟验证与 SSH 参考一致（价 maxRel~4e-6=float32 舍入，量额 maxΔ=0）
- 路径常量：`core.config.MINUTE_DATA_DIR`

**股票池声明**：spec 写 `universe.primary_index: MINUTE_DIR`，`build_universe()` 自动扫目录返回 `[0-9]*.XSH[EG]` parquet stem。

---

## 2. 核心算子 `minute_intraday_aggregate`

文件：`core/operators/minute_intraday_aggregate.py`

**流程**：一只股票一只股票流式：load → 同时点 σ → 喷发标签 → 峰/岭/谷分类 → 日频 reduce。

**输出**：日频 long 表（`(order_book_id, date, *features)`），与下游 `rolling` / `compute` / `rank` 完全兼容。

**spec 字段**：
- `cache_key`（必填，bump 时自动失效，见 §3）
- `features`（必填，必须是 `_SUPERSET_COLUMNS` 子集）
- `std_window`（默认 20）
- `std_threshold`（默认 1.0）

**superset 列**（spec.features 必须从中挑）：见算子文件 `_SUPERSET_COLUMNS`。包含三类 count / vol_sum / turnover_sum / vwap、峰/岭间隔 5 阶矩、喷发后下一分钟成交额、日频价/量。

**warmup 日**（前 `std_window` 日）所有 feature 列输出 NaN（**不是 0**）。

---

## 3. 缓存机制（per-stock parquet，应对日更）

**路径**：`INTERMEDIATE_CACHE_DIR / <cache_key>__h<params_hash> / <order_book_id>.parquet`

**hash 来源**：`params_hash = sha1(std_window, std_threshold, _FEATURES_SUPERSET_VERSION)[:10]` —— 任何参数变化自动 cache miss。

**命中规则**：source mtime ≤ cache mtime → load；miss → 重算 + atomic rename 写盘。用户日更某只 source parquet → 该股 cache 单独失效，其他不动。

**bump cache_key 规则**：
- 算子 superset 列定义变更时，bump `_FEATURES_SUPERSET_VERSION`（自动失效所有 cache）
- 只是想强制重算某个 spec 时，改 spec 的 `cache_key` 字段（如 `prv_v1` → `prv_v2`）

---

## 4. 并发

环境变量 `MINUTE_WORKERS`（默认 8）控制 **ProcessPoolExecutor**（真多进程，绕开 GIL）并发度。

**128 核机器请用 `MINUTE_WORKERS=100`**：
- 实测 64 worker 已能 55 stocks/sec、全市场 ~62 秒
- 100 worker 进一步压到 ~40–60 秒区间，NFS bandwidth 接近瓶颈

**线程池版（旧）实测仅 ~3 核效率，不要用**。

---

## 5. `compute` 算子的 NaN 惯用法

`_BUILTINS` 已包含 `"nan"` 字面量；但 `pd.eval` **不支持 `where()` 函数**（即使在 `_BUILTINS` 里），实现条件 NaN 用"divide-by-mask"惯用法：

```yaml
formula: kurt_raw * (gate / gate)   # gate=0 → 0/0=NaN；gate=1 → 1/1=1
```

---

## 6. 典型 spec 模板

- 最简：`sources/kysec/paper_27_microstructure/specs/peak_minute_count/`
- 高阶矩 + 守门 NaN：`sources/kysec/paper_27_microstructure/specs/peak_interval_kurt/`

---

## 7. 实战经验：ridge 类因子高缺失率（已缓解 2026-05-25）

**症状**：`peak_ridge_price_ratio` / `valley_ridge_price_ratio` / `ridge_relative_vwap` 原版 miss_listed ~78%。

**根因**：rolling 默认 `min_periods = window = 20` + ridge 稀疏 → 20 日窗口内任一天 0-ridge 即整窗 NaN。

**缓解**：通过 `__mp10` 变体（`min_periods: 10`）降至 ~10%；ICIR 仅损失 ~7%。详见 `sources/kysec/paper_27_microstructure/README.md` "min_periods 衰减实验"。

**坑**：**不要 fillna `ridge_vwap`**——会污染因子语义（无 ridge 的天本应是 NaN，硬填会引入虚假 ratio）。
