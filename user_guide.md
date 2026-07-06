# ML 选股管线 User Guide

> 从「启动训练」到「导出回测信号」的运行手册。覆盖两条生产线（LGBM / MLP）+ 一个共享内核（ml_core）。

---

## 一、管线总览

仓库里有**两条独立的 ML 选股生产线**，外加一个被两条线共用的**管线内核**：

| 包 | 模型 | 角色 | 入口 |
|----|------|------|------|
| `ml_core/` | 模型无关内核（LGBM / MLP 一套配置切换） | **推荐主线**：配置驱动训练 + 推理 | 训练 `python -m ml_core.run`、推理 `python -m ml_core.predict` |
| `ml/` | LightGBM（两阶段：筛选 → 合成） | 旧生产线，CLI 完整（参数走命令行） | `python -m ml.run` |
| `ml_ht/` | PyTorch FCNN（华泰人工智能系列复现） | 旧生产线，CLI 完整 | `python ml_ht/run.py` |

> `ml_core` 现已具备**正式的训练 / 推理 CLI 入口**（不再只是库 + 验证脚本）：训练改 `ml_core/train_config.yaml`、推理改 `ml_core/predict_config.yaml`，源码零改动。两条线（LGBM / MLP）只靠配置 `model:` 一行切换。下文 §三 即此主线手册。

### ml_core 的分层（两条线为什么能共用）

`ml_core` 把两条线**逐字节等价**的环节上提，分两层：

- **模型无关内核**：`universe`（底座）/ `features`（因子组装）/ `splits`（时间切分）/ `metrics`（IC）/ `signals`（信号导出）/ `pipeline`（编排）
- **可插拔策略**（模型相关，但用策略模式收敛，正交注入）：
  - 标签 `labels`：`ExcessReturn`(LGBM 回归) ↔ `BinaryMedian`(MLP 二分类)
  - 标准化 `scaling`：`WholeSetRobustZ`(LGBM 全集稳健Z，有状态持久化) ↔ `DailyCrossSectionMAD`(MLP 逐日截面MAD，无状态)
  - 模型 `model`：`LGBMAdapter` ↔ `MLPAdapter`

核心不变量：**预测路径（`predict_live`）训练/实盘共用同一套 `build_universe` + `build_feature_matrix`**，杜绝 train/serve skew。目前 `ml/predict_live` 与 `ml_ht` 的预测都已重接到 `ml_core` 并通过零漂移验证。

---

## 二、环境准备

```bash
source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate
```

**数据后端主开关 `DATA_BACKEND`**（默认 `dquant`，一处切换消费侧数据源）：

- `DATA_BACKEND=dquant`（默认）→ 读 dquant 系数据，ml_ht 产物落 `ml/ht_dquant/`
- `DATA_BACKEND=rq` → 回退旧 rq 基准，ml_ht 产物落 `ml/ht/`
- 细粒度仍可单独覆盖：`ALPHA158_DATA_BACKEND` / `ML_HT_BACKEND`
- ⚠️ `MINUTE_DATA_BACKEND` 故意**不随**主开关（它是分钟因子的「生产隔离轴」，会重定向整个 `RAW_FACTOR_BASE`）。**默认已是 `dquant`**，无需 export；仅复现旧 rq 基线时才 `export MINUTE_DATA_BACKEND=rq`。

> 下面所有命令默认 `DATA_BACKEND=dquant`（不写即默认）。所有命令均为单行，可直接复制。

### dquant 因子目录速查（ml_core 训练读这里）

ml_core 通过 `discover_features(sources=[...])` 从 `factors/raw/<source>/` 发现因子宽表。当前默认 sources 及目录结构：

```
<nfs/ofs-prediction/peterzhenglinpeng>/factors/raw/
├── alpha158-dquant/                    ← ml_core 默认 source（158 因子）
│   ├── kline/      (9 个: KMID/KLEN/KLOW/KUP/KSFT...)
│   ├── price/      (4 个: OPEN0/HIGH0/LOW0/VWAP0)
│   ├── rolling/    (115 个: BETA*/CNT*/MA*/RSI*/VSTD...)
│   └── volume/     (30 个: VMA*/VSTD*)
├── kysec-dquant/
│   └── paper_27_microstructure/        ← ml_core 默认 source（23 个分钟微结构因子）
└── style-dquant/                       ← 风格/日历因子（week_of_year 等，规划中）
```

每个因子是一个 parquet 宽表：`index=交易日(DatetimeIndex, 2005-01-04~最新)` × `columns=股票代码(000001.XSHE 格式)` × `values=float32`。`discover_features` 按 `<source>/<group>/<factor>.parquet` 两级路径发现，source 名带 `-dquant` 后缀（与 rq 版本 `alpha158/`、`kysec/` 平级共存，互不覆盖）。

> ⚠️ **两套 dquant 目录别混淆**：
> - **消费轴** `factors/raw/<source>-dquant/`（上面这些）—— ml_core 训练/推理读这里，由 `DATA_BACKEND=dquant` 驱动。
> - **生产隔离轴** `factors/raw-dquant/<source>/`（分钟因子生产落这里）—— 由 `MINUTE_DATA_BACKEND` 驱动（**默认 dquant**，无需 export），与消费轴独立，ml_core 默认不读。

### 日更：把因子更新到最新（推理前置）

推理只能推到「因子面板末日」。日更增量把模型读的两套 dquant 因子（`alpha158-dquant` + `kysec-dquant/paper_27_microstructure`）推到最新交易日，再跑 `latest_n` 增量推理即可。全部 append-only、历史冻结、零 API（数据线除外）。

**① 数据线（dquant 专用取数，硬编码写 dquant 目录；增量补缺日）**

```bash
PYTHONPATH=. python data_fetching/raw_ohlcv_dquant.py
```
```bash
PYTHONPATH=. python data_fetching/ex_factors_jy.py
```
```bash
PYTHONPATH=. python data_fetching/minute_ohlcv_dquant.py --workers 64
```

**② alpha158-dquant 因子（读 daily_dquant，零 API；后端感知写 `alpha158-dquant`）**

```bash
PYTHONPATH=. python alpha158/daily_update.py
```

**③ p27 微结构因子（分钟线，默认 `MINUTE_DATA_BACKEND=dquant`，无需 export；先刷 superset 再批量 L3）**

```bash
PYTHONPATH=. python pipeline/refresh_supersets.py --cache-key prv_v3
```
```bash
PYTHONPATH=. python pipeline/refresh_factors_batch.py --factor-glob 'sources/kysec/paper_27_microstructure/specs/*'
```

**④ 验证末日（两套都应 = 最新交易日）**

```bash
PYTHONPATH=. python -c "import pandas as pd; from core import config as c; f=lambda p: pd.to_datetime(pd.read_parquet(p, columns=[]).index).max().date(); print('a158', f(c.ALPHA158_RAW_BASE/'rolling/MAX60.parquet')); print('p27', f(c.RAW_FACTOR_BASE/'kysec-dquant/paper_27_microstructure/peak_minute_count.parquet'))"
```

> ⚠️ 后端要点：alpha158 认 `ALPHA158_DATA_BACKEND`（默认 dquant）→ 写 `alpha158-dquant`；p27 分钟因子认 `MINUTE_DATA_BACKEND`，**默认已是 dquant**，无需显式 export，产物落 `factors/*-dquant/` + `output-dquant/`，模型可直接读到。仅复现旧 rq 基线时才 `export MINUTE_DATA_BACKEND=rq`。两者是独立轴。
> 细节与全链路（掩码/信号/回测）见 `docs/server_daily_production.md`；alpha158 线总览见 `alpha158/README.md`。
> `pipeline/daily_update.sh` 是日更编排器（不设 `MINUTE_DATA_BACKEND`，跟随默认）；**默认翻转后它现在也产 dquant 分钟/p27**。要用它跑旧 rq 基线，需先 `export MINUTE_DATA_BACKEND=rq` 再执行。

---

## 三、ml_core 主线：配置驱动训练 + 推理（推荐）

> 全流程只改两个 yaml、源码零改动：训练 `ml_core/train_config.yaml` → `python -m ml_core.run`；
> 推理 `ml_core/predict_config.yaml` → `python -m ml_core.predict`。
> 下面以最常见诉求示范：**LGBM 选 64 因子 → 训练 → 出回测信号**。

### 1) 训练：改 `ml_core/train_config.yaml` 哪几行

跑 LGBM + SHAP 选 64 因子，**只需关注下面这几个键**（其余保持默认即可）：

| 键 | 设成 | 说明 |
|----|------|------|
| `run_id` | 起个名字，如 `lgbm_a158_p27_top64` | **强烈建议显式命名**（推理时 `predict_config.yaml` 的 `model_run_id` 要照抄它）。留 `null` 会用启动时间戳，那样还得去 logs 里翻名字 |
| `model` | `lgbm` | 换 MLP 就改这一行为 `mlp`，超参看 `mlp:` 段 |
| `sources` | 你的因子源列表 | 默认已是 `alpha158-dquant` + `kysec-dquant/paper_27_microstructure` |
| `neu_sources` | `null` | 需要中性化因子再填 |
| `select_method` | `shap` | 两阶段筛选；`gbdt`=增益排序；`null`=不选、单阶段全特征 |
| `top_k` | `64` | 入选因子数 |
| `split` | 按需 | 训练/验证/测试时间段；默认 train≤2017-11、valid 2018~2019-11、test 2020~ |
| `lgbm:` 段 | 一般不动 | `seed/num_threads/learning_rate/num_leaves`，缺省键走代码默认 |

```bash
python -m ml_core.run
```

> 入选的 64 个因子**会按重要性降序逐行打印到日志**（`1. xxx importance=0.0834 …`），便于复看排序——这条排序也同时落在 `selected_features.json` 的 `scores` 字段里。

### 2) 推理 / 出信号：改 `ml_core/predict_config.yaml` 哪几行

模型口径（model / sources / 策略 / 因子顺序）由训练落盘的 `run_meta.json` 自带，推理直读 → **零 train/serve 漂移**。本文件只决定「用哪个模型 + 哪段区间 + 怎么导信号」：

| 键 | 设成 | 说明 |
|----|------|------|
| `model_run_id` | = 训练时的 `run_id` | 指向 `ml/models/<run_id>/` |
| `latest_n` | 全量 `null` / 日更填整数（如 `1`） | 全量重生成走 `start/end`；日更只算最近 N 个交易日（append-only 自动补缺、冻结历史） |
| `start` / `end` | 全量区间 | `end: null` = 自动到入选因子「共同覆盖」的最末交易日 |
| `rebuild` | 首次全量出信号 `true`；日更 `false` | `false`=append-only 不覆盖历史；`true`=全段重写（重训后覆盖旧信号） |
| `top_n` | `500` | 每日取预测分 top-N |
| `signal_dir` | `null` | `null` → `ml/predictions/<run_id>/signals` |
| `coverage_*` | 默认即可 | 面板陈旧守门：近 N 日非空骤降则告警；`strict_coverage: true` 则中止 |

```bash
python -m ml_core.predict
```

- **新模型首次全量**：`latest_n: null` + `start/end` + `rebuild: true`
- **老模型日更增量**：`latest_n: 1` + `rebuild: false`（cron 语义，只补当天、冻结历史）

### 3) 会生成哪些产物、落在哪

`<DATA_ROOT>` 默认 `/nfs/ofs-prediction/peterzhenglinpeng`（受 `FACTOR_REPL_DATA_ROOT` 覆盖）。

**训练（`python -m ml_core.run`）产物：**

| 产物 | 路径 | 内容 |
|------|------|------|
| 模型 | `ml/models/<run_id>/model.txt` | LGBM booster（MLP 则为 `model.pt`） |
| 尺子 | `ml/models/<run_id>/scaler_x.parquet` | train 段拟合的标准化器（LGBM 有状态；MLP 无状态则为空操作） |
| 特征列序 | `ml/models/<run_id>/feature_names.json` | 喂模型的因子顺序（serving 必须同序） |
| 推理元信息 | `ml/models/<run_id>/run_meta.json` | model/sources/策略/horizon/feature_order —— 推理自包含的唯一真相源 |
| 入选明细 | `ml/models/<run_id>/selected_features.json` | 两阶段才有：64 因子 + 重要性 `scores`（即日志里那份排序） |
| 评估面板 | `ml/predictions/<run_id>/pred_panel.parquet` | test 段 ŷ 面板（date×stock） |
| IC 序列 | `ml/predictions/<run_id>/ic_series.parquet` | 逐日样本外模型 IC |
| 训练日志 | `ml_core/logs/<run_id>_<时间戳>.log` | 全程：选因子曲线 + **入选 64 因子排序** + 样本外 IC |

**推理（`python -m ml_core.predict`）产物：**

| 产物 | 路径 | 内容 |
|------|------|------|
| 回测信号 | `ml/predictions/<run_id>/signals/YYYY-MM-DD.txt` | 每日一份，每行 `YYYY-MM-DD_股票代码`，行序=预测分降序选股优先级；默认 append-only |
| 推理日志 | `ml_core/logs/predict_<run_id>_<时间戳>.log` | 区间解析 + 覆盖守门 + 导出统计 |

> 注：`ml_core.predict` 当前只导信号 txt，不另存 ŷ 面板 parquet（信号即终产物，回测只用排序）。

---

## 四、旧 CLI 生产线：LGBM（`ml/`，参数走命令行）

> 与 §三 的 `ml_core` 等价（同数值口径），区别只在参数从命令行传而非 yaml。新工作建议直接用 §三。

### 1) 训练：全因子 → SHAP/GBDT 选 top-k → 重训 → 样本外 IC

```bash
python -m ml.run --sources alpha158-dquant --select-method shap --top-k 64 --run-id my_run
```

常用参数：

- `--sources`：raw 因子源（路径分量，如 `alpha158-dquant`、`kysec-dquant/paper_27_microstructure`），可多个
- `--neu-sources`：从 `factors/neu` 读的中性化因子源
- `--select-method gbdt|shap`：筛选方法（gbdt=自带增益，shap=TreeExplainer）
- `--top-k 64`：选多少因子
- `--num-threads`：并行多实验时降线程（经验 `128 // N`）
- `--seed`：随机种子（默认 42）
- `--date-sample 5` / `--max-features 60`：冒烟加速

**产物**（`<DATA_ROOT>/ml/`）：

- `models/<run_id>/model.txt`、`selected_features.json`（含 features + sources）、`scaler_x.parquet`（train 段拟合的尺子）
- `predictions/<run_id>/pred_panel.parquet`（评估面板）、`ic_series.parquet`
- `ml/logs/<run_id>_<时间戳>.log`（训练全记录 + 入选因子重要性）

### 2) 实盘推理：补全到最新因子日的 ŷ 面板（只过 pre_mask，不过 label）

```bash
python -m ml.predict_live --run-id my_run --start 2022-01-01
```

- `--end` 留空 → 自动取入选因子「共同覆盖」的最末交易日
- `--strict-coverage`：任一入选因子近 10 日覆盖骤降则中止（防陈旧面板污染信号）
- **产物**：`predictions/<run_id>/pred_panel_live.parquet`

### 3) 导出回测可读信号：每日排序选股名单 txt（append-only）

```bash
python -m ml.export_signal --run-id my_run --source live --top-n 500
```

- `--source live|eval`：用实盘面板 / 评估面板
- `--layout daily|merged`：每日一份 `YYYY-MM-DD.txt` / 单个 `signal.txt`
- 默认 **append-only**（只补新增交易日，冻结历史）；`--rebuild` 才全段重写
- **产物**：`signals/<run_id>/YYYY-MM-DD.txt`（每行 `YYYY-MM-DD_股票代码`，行序=选股优先级）

---

## 五、旧 CLI 生产线：MLP（`ml_ht/`，华泰 FCNN）

### 训练（存 `model.pt` + 逐年 test 报告）

```bash
python ml_ht/run.py --train
```

### 预测 + 导出信号（因子组装走 ml_core 现算，已脱离预物化长表）

```bash
python ml_ht/run.py --predict --start-date 2026-04-01 --end-date 2026-06-26
```

- `--latest-n 1`：日频增量，只跑最后 N 个交易日（cron 语义）
- `--train --predict`：训完立即预测
- `--out-dir`：自定义信号目录（默认 `ml/ht_dquant/signals/`）

**产物**（`<DATA_ROOT>/ml/ht_dquant/`，rq 端为 `ht/`）：

- `models/stock_mlp.pt`（最新指针）、`models/feature_names.json`（固化训练列序，serving 必须同序）
- `runs/<时间戳>/model.pt` + `test_report.json`
- `signals/YYYY-MM-DD.txt`

---

## 六、产物目录速查

| 内容 | 路径（`<DATA_ROOT>` 默认 `/nfs/ofs-prediction/peterzhenglinpeng`） |
|------|------|
| **ml_core 模型/尺子/元信息** | `ml/models/<run_id>/`（`model.txt`/`scaler_x.parquet`/`feature_names.json`/`run_meta.json`/`selected_features.json`） |
| **ml_core 预测面板/IC** | `ml/predictions/<run_id>/`（`pred_panel.parquet`/`ic_series.parquet`） |
| **ml_core 回测信号** | `ml/predictions/<run_id>/signals/YYYY-MM-DD.txt` |
| **ml_core 训练/推理日志** | `<repo>/ml_core/logs/<run_id>_<时间戳>.log`、`predict_<run_id>_<时间戳>.log` |
| LGBM(`ml/`) 模型/尺子 | `ml/models/<run_id>/` |
| LGBM(`ml/`) 预测面板/IC | `ml/predictions/<run_id>/` |
| LGBM(`ml/`)/MLP 信号 | `ml/signals/<run_id>/`、`ml/ht_dquant/signals/` |
| LGBM(`ml/`) 训练日志 | `<repo>/ml/logs/<run_id>_<时间戳>.log` |
| MLP(`ml_ht/`) 模型/run | `ml/ht_dquant/{models,runs}/` |

---

## 七、复现示例：`a158_p27_shap_dquant_nocxl`

该模型 = alpha158-dquant + 微结构因子（paper_27），SHAP 选 top-64，dquant 后端。原始训练日志见 `ml/logs/a158_p27_shap_dquant_nocxl_20260629_191634.log`（结果：64 因子，样本外 IC=+0.1184 ICIR=+1.182）。

复现训练（LGBM 设了 `deterministic=True` + `seed=42` + `num_threads=64`，同数据应可复现）：

```bash
python -m ml.run --sources alpha158-dquant kysec-dquant/paper_27_microstructure --select-method shap --top-k 64 --run-id a158_p27_shap_dquant_nocxl_repro
```

> 用了 `_repro` 后缀，避免覆盖原产物，便于和原模型逐项对照。出信号接 §四 的 `predict_live` → `export_signal`（`--run-id a158_p27_shap_dquant_nocxl_repro`），或用 §三 `ml_core.predict`（`model_run_id` 填该 run_id）。

---

## 八、零漂移验证脚本（`ml_core/verify_*`）

重构等价性自检（不碰生产产物，走隔离命名空间）：

```bash
python -m ml_core.verify_e2e_lgbm
```
```bash
python -m ml_core.verify_train_smoke
```

- `verify_e2e_lgbm`：同进程同数据下，`ml_core.predict_live` 与 `ml.predict_live` 逐元素比对 ŷ（应 max_abs=0）+ 比信号名单
- `verify_train_smoke`：`run_train` 跑通 LGBM(回归) + MLP(二分类) 两条路径（date_sample 加速，只验方向）
- 其余：`verify_features` / `verify_universe` / `verify_strategies`
