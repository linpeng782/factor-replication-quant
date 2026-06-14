# ML 训练流水线操作手册

> 本手册覆盖：训练 → 推理 → 信号导出 → 回测对比的完整串行流程。

---

## 一、完整串行流程

### Step 1: 训练（Training）

```bash
cd /nfs/volume-1593-1/peterzhenglinpeng/factor-replication-quant-new

# GBDT 版
python -m ml.run \
  --sources alpha158 cxl kysec/paper_27_microstructure \
  --run-id <run_id> \
  --top-k 64

# SHAP 版
python -m ml.run \
  --sources alpha158 cxl kysec/paper_27_microstructure \
  --run-id <run_id>_shap \
  --select-method shap \
  --top-k 64
```

**产物**：
- `ml/models/<run_id>/model.txt`
- `ml/models/<run_id>/selected_features.json`
- `ml/models/<run_id>/scaler_x.parquet`
- `ml/predictions/<run_id>/pred_panel.parquet`
- `ml/predictions/<run_id>/ic_series.parquet`
- `ml/logs/<run_id>_<时间戳>.log`

---

### Step 2: 实盘推理（Live Prediction）

```bash
python -m ml.predict_live --run-id <run_id> --start 2022-01-01
```

**产物**：`ml/predictions/<run_id>/pred_panel_live.parquet`

---

### Step 3: 信号导出（Signal Export）

```bash
python -m ml.export_signal \
  --run-id <run_id> \
  --source live \
  --layout daily \
  --top-n 500
```

**产物**：`ml/signals/<run_id>/YYYY-MM-DD.txt`

---

### Step 4: 回测对比（Backtest）

```bash
cd /nfs/volume-1593-1/peterzhenglinpeng/daily-realtime-backtest-pipeline
source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate

# 单信号回测
python compare_signals.py --signals <run_id>

# 多信号对比
python compare_signals.py --signals <run_id_a> <run_id_b>
```

**产物**：`results/compare/compare_annual_returns_<时间戳>.png`

---

## 二、关键注意事项

### 1. new_stock_mask 必须保持最新

`new_stock_mask_long.parquet` 目前**不是** combo_mask 日更的一部分。发现新股漏网时，需要手动重新生成：

```bash
cd /nfs/volume-1593-1/peterzhenglinpeng/stock-data-fetching
source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate
python build/build_new_stock_mask_rq_only.py

# 然后复制到两个目录
cp backtest_engine/cache_dir/new_stock_mask_rq_YYYYMMDD.parquet \
   backtest_engine/cache_dir/new_stock_mask_long.parquet
cp backtest_engine/cache_dir/new_stock_mask_rq_YYYYMMDD.parquet \
   /nfs/ofs-prediction/peterzhenglinpeng/market-data/masks/new_stock_mask_long.parquet
```

### 2. pre_mask 口径已统一

`ml/dataset.py` 的 `load_pre_mask()` 已改用 `alpha_shared.cleaning.mask_loader`。

当前逻辑：
- `pre_mask = NOT(is_st[t+1] OR is_suspended[t+1] OR is_new_stock[t+1])`（shift(-1) 语义）
- 涨停股**不过滤**（留给回测系统处理）

训练/推理口径一致，无需额外处理。

### 3. 数据源选择

- combo_mask 读 `backtest_engine/cache_dir/combo_mask_long.parquet`（日更到最新）
- new_stock_mask 读 `market-data/masks/new_stock_mask_long.parquet`（需手动更新）

---

## 三、快速参考：常用 run-id 命名规范

| 类型 | 示例 |
|------|------|
| GBDT | `cxl_a158_p27_raw` → `cxl_a158_p27_raw_v2` |
| SHAP | `cxl_a158_p27_raw_shap` → `cxl_a158_p27_raw_shap_v2` |

---

## 四、环境依赖

```bash
# 激活虚拟环境
source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate

# 确认 rqdatac 可用（生成 new_stock_mask 时需要）
python -c "import rqdatac; print('ok')"
```
