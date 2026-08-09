# 因子复现 Agent —— CLI 必读

> 研报/文本 → 自动复现为可执行量化因子（基于米筐 RQData）

---

## 1. 环境与路径

```bash
# 远端 SSH 机器（高频因子产线，65 GB 分钟数据所在）
source /nfs/ofs-prediction/peterzhenglinpeng-code/peterdidi/bin/activate   # Python 3.11

# ⚠️ /tmp 是 overlay 只有 20G，经常被占满（实测 83%+）。所有临时文件、缓存、
# 中间产物一律放 /nfs/ofs-prediction/ 下（90T，12T 可用），不要写 /tmp。
# 例如：tmpdir 用 /nfs/ofs-prediction/peterzhenglinpeng/tmp/ 而非 /tmp/

# 本机 macOS（基本面因子复现；高频因子留远端，详见 LOCAL_SETUP.md）
source /Users/didi/kdj/peterdidi/bin/activate                        # Python 3.11.9 venv（本机唯一 venv，勿再找）
export FACTOR_REPL_DATA_ROOT=/Users/didi/DATA                      # 数据根重定向

# 本机路径速记（避免反复探测）：
#   venv         /Users/didi/kdj/peterdidi
#   本仓         /Users/didi/kdj/factor-repilcation-quant
#   alpha-shared /Users/didi/kdj/alpha-shared
#   数据根       /Users/didi/DATA  （RAW_FACTOR_DIR 下本机有 ~22 个基本面 raw 因子）
```

数据根 `<DATA_ROOT>` 下**两个角色桶**：`factors/`（因子产出）+ `market-data/`（评估输入）。

> **路径常量速查（config/ 子模块 → 实际路径；DATA_BACKEND=dquant 默认口径）**
> 搞不清数据在哪时看这里，不用每次探测。所有常量由 `config/` 解析，`import config` 即可取。

### market-data/ 评估输入（`config/market_data.py`）

| 常量 | 实际路径（dquant 默认） | 内容 |
|------|------|------|
| `COMBO_MASK_PATH` | `<DATA_ROOT>/backtest_engine/cache_dir_dquant/combo_mask_long.parquet` | 交易状态长表（ST/停牌/涨停/新股），评估零 API |
| `NEW_STOCK_MASK_PATH` | `<DATA_ROOT>/backtest_engine/cache_dir_dquant/new_stock_mask_long.parquet` | 新股 mask 长表 |
| `VWAP_PANEL_PATH` | `<DATA_ROOT>/market-data/labels/vwap_panel.parquet` | PIT vwap 宽表（T×N），labels 缺 horizon 时 fallback 现算 |
| `LABELS_DIR` | `<DATA_ROOT>/market-data/labels/` | `forward_return_{1,2,5,10,20}d.parquet` 预算远期收益 |
| `INDUSTRY_PANEL_ZX_PATH` | `<DATA_ROOT>/market-data/industry-dquant/industry_panel_zx_dquant.parquet` | 中信一级行业日频宽表（中性化/画像用） |
| `MARKET_CAP_PANEL_PATH` | `<DATA_ROOT>/market-data/market_cap/market_cap_panel.parquet` | 总市值日频宽表（中性化/大小盘画像用） |
| `TURNOVER_RATE_PANEL_PATH` | `<DATA_ROOT>/market-data/turnover-dquant/turnover_rate_panel.parquet` | 日换手率宽表（T×N，%，dquant `get_turnover_rate.today`；改进动量因子权重源） |
| `NORMAL_DAY_PANEL_PATH` | `<DATA_ROOT>/market-data/limit-dquant/normal_day_panel.parquet` | 正常交易日 mask（T×N，1=非停牌且未触涨跌停；长端动量剔除无效日用） |
| `INDUSTRY_INDEX_RETURN_PATH` | `<DATA_ROOT>/market-data/industry/industry_index_return.parquet` | 中信一级行业指数日收益（T×33，联合动量因子用） |
| `INDEX_SEGMENTS_PATH` | `<DATA_ROOT>/market-data/index/000985_segments.parquet` | 000985 中证全指日频四段收益（APM 回归市场参照） |

### factors/ 因子三阶段产物（`config/factors_output.py`，随 MINUTE_BACKEND 生产隔离轴）

| 常量 | 实际路径（dquant 默认） | 内容 |
|------|------|------|
| `RAW_FACTOR_BASE` | `<DATA_ROOT>/factors/raw-dquant/<source>/<group>/` | 因子 raw 阶段；namespace=`<source>/<group>` |
| `CLEANED_FACTOR_BASE` | `<DATA_ROOT>/factors/cleaned-dquant/<source>/<group>/` | 因子 cleaned 阶段（MAD+zscore+mask） |
| `NEU_FACTOR_BASE` | `<DATA_ROOT>/factors/neu-dquant/<source>/<group>/` | 因子 neu 阶段（强制行业市值中性化） |
| `ALPHA158_RAW_BASE` | `<DATA_ROOT>/factors/raw-dquant/alpha158-dquant/` | alpha158 raw 产物（消费轴随 ALPHA158_BACKEND） |
| `RET20_PANEL_PATH` | `<DATA_ROOT>/factors/helpers/ret20_panel.parquet` | 20 日后复权收益面板（APM 截面回归去动量用） |
| `RET1_PANEL_PATH` | `<DATA_ROOT>/factors/helpers/ret1_panel.parquet` | 日频后复权收益面板（改进动量因子收益源） |
| `AMP_PANEL_PATH` / `AMP_HL_PANEL_PATH` | `<DATA_ROOT>/factors/helpers/{amp,amp_hl}_panel.parquet` | 日振幅面板：(H−L)/前收（长端动量 2.0）/ H÷L−1（1.0） |
| `OUTPUT_DIR` | `<repo>/output-dquant/<source>/<group>/<factor>/` | 评估产物（两张 PNG，项目相对路径） |

### 因子产线原料（`config/factors_input.py`）

| 常量 | 实际路径（dquant 默认） | 内容 |
|------|------|------|
| `RAW_OHLCV_DIR` | `<DATA_ROOT>/market-data/daily-dquant/stock-ohlcv-dquant/` | 逐股原始日频 OHLCV（不复权） |
| `EX_FACTORS_DIR` | `<DATA_ROOT>/market-data/daily-dquant/stock-ex-factors-jy/` | 逐股稀疏复权因子（jy adjfactor） |
| `FUNDAMENTALS_DIR` | `<DATA_ROOT>/market-data/fundamentals-dquant/` | 基本面 PIT 字段快照（cxl 生产原料） |
| `INDUSTRY_PANEL_ZX_DQUANT_PATH` | `<DATA_ROOT>/market-data/industry-dquant/industry_panel_zx_dquant.parquet` | dquant 中信一级行业面板（因子生产 fetch custom 用） |
| `MINUTE_RAW_DIR` | `<DATA_ROOT>/market-data/minute-dquant/raw/` | 分钟原始（不复权）按日分片 `<YYYY-MM-DD>.parquet` |
| `MINUTE_EX_FACTORS_DIR` | `<DATA_ROOT>/market-data/daily-dquant/stock-ex-factors-jy/` | 分钟复权因子（与 alpha158-dquant 同源） |
| `INTERMEDIATE_CACHE_DIR` | `<DATA_ROOT>/intermediate-cache-dquant/` | 分钟→日频特征中间缓存（可再生） |

### ML 训练/预测产物（`config/ml.py`）

| 常量 | 实际路径 | 内容 |
|------|------|------|
| `ML_MODELS_DIR` | `<DATA_ROOT>/ml/models/<run_id>/` | LGBM 模型 + scaler_x + run_meta.json + selected_features.json |
| `ML_PREDICTIONS_DIR` | `<DATA_ROOT>/ml/predictions/<run_id>/` | `pred_panel_live.parquet`（T×N 分数面板）+ `signals/` 每日 txt |
| `ML_DATASETS_DIR` | `<DATA_ROOT>/ml/datasets/` | (可选) train/valid/test 矩阵 |
| `ML_LOGS_DIR` | `<repo>/ml_core/logs/` | 训练日志（repo 内，.gitignore 排除） |

### 分析产物（`config/analysis.py`）

| 常量 | 实际路径 | 内容 |
|------|------|------|
| `INVENTORY_ROOT` | `<DATA_ROOT>/factor-inventory/` | 因子总账/IC序列/相关性/对比等分析产物 |

### 横切参数（`config/params.py`）

| 常量 | 值 | 说明 |
|------|------|------|
| `DEFAULT_START_DATE` | `20100101` | fetch 区间起点（给 8 期 PIT 滚动因子留 warm-up） |
| `DEFAULT_END_DATE` | `20260527` | fetch 区间终点 |
| `DEFAULT_EVAL_START_DATE` | `20160101` | 评估区间起点（IC/ICIR 对齐历史） |
| `DEFAULT_EVAL_END_DATE` | `20251231` | 评估区间终点 |

### 后端开关（`config/base.py`）

| 环境变量 | 默认 | 可选值 | 影响 |
|------|------|------|------|
| `FACTOR_REPL_DATA_ROOT` | `/nfs/ofs-prediction/peterzhenglinpeng` | 本机路径 | 数据根重定向 |
| `DATA_BACKEND` | `dquant` | `dquant`/`rq` | 主开关：alpha158/基本面/mask 整体后端 |
| `MINUTE_DATA_BACKEND` | `dquant` | `dquant`/`rq` | 生产隔离轴：分钟+三阶段产物整体重定向到 `-dquant` 并行目录 |

> ⚠️ `MINUTE_DATA_BACKEND` 故意不随主开关——dquant 把整个 `factors/{raw,cleaned,neu}` + `output` 重定向到并行 `-dquant` 目录，与消费轴语义不同，强行合并会静默破坏因子发现。

`<DATA_ROOT>` 默认 `/nfs/ofs-prediction/peterzhenglinpeng`；预计算数据已更新到 2026-05-15。

**本机化运行**：`export FACTOR_REPL_DATA_ROOT=/Users/didi/DATA` 一键把所有数据路径
重定向到本机；远端不设环境变量行为零变化。`OUTPUT_DIR` 是项目相对路径不受影响。

**数据起始统一 2005-01-01**：日频（raw OHLCV / ex-factors）与分钟频（1m）数据**一律从 2005-01-01 下载**（rqdatac 分钟最早即 2005-01-04）。复权口径：**价格后复权（× `ex_cum_factor`，ffill 到当日），成交量 / 成交额不复权**；复权因子用本地 `<DATA_ROOT>/market-data/daily/stock-ex-factors/`（可日更，与米筐 `adjust_type='post'` 口径一致，已逐分钟 bit 级验证）。分钟拉取配方：`get_price(adjust_type='none')` 取原始 → 价格 ×`ex_cum_factor` → 量/额原样 → float32 落 `MINUTE_DATA_DIR`。

**本机数据现状（`/Users/didi/DATA`，已验证就绪，勿再探测）**：

| 路径 | 内容 | 形状/数量 |
|------|------|-----------|
| `factors/{raw,cleaned,neu}/cxl/<group>/*.parquet` | cxl 系基本面因子 三阶段（6 个 group 系列） | 各 22 个 |
| `market-data/labels/forward_return_{1,2,5,10,20}d.parquet` | PIT 远期收益（评估直读，bit-exact） | (5178, 5501) |
| `market-data/prices/vwap_panel.parquet` | PIT vwap 宽表（labels 缺 horizon 时 fallback 现算） | (5178, 5501) |
| `market-data/masks/{combo,new_stock}_mask_long.parquet` | 交易状态长表（ST/停牌/涨停/新股） | ~17M 行 |
| `market-data/{market_cap,industry}/*.parquet` | 总市值 + 中信行业面板（中性化用） | (5178/5196, ~5500) |

旁注：`<DATA_ROOT>` 下还有 `alpha158/` `alpha191/` `cxl-work/` 等旧顶层目录，**尚未纳入 `factors/<source>/<group>/`**（待 alpha158 接入时统一）。

本机 22 因子可直接 `python run.py <factor> --evaluate-only`（零 API 调用）。
跑脚本前置：`source /Users/didi/kdj/peterdidi/bin/activate` 且在仓库根执行（`core` 包在 cwd）。

**alpha_shared 工具包**（已内嵌，`alpha_shared/` 在仓库根）：IC / 分层回测 / 清洗 / mask 加载等
数值算法。`core/evaluation.py` 直接 `from alpha_shared.{evaluation,cleaning}...` 取算法，
mask 路径在调用点注入 config。改算法直接改 `alpha_shared/`，改完务必跑
`scripts/regression_baseline_replication.py` + `scripts/regression_compare.py` 验数值零漂移。
评估编排：`core/evaluation.py`；可视化：`core/eval_plots.py`。

---

## 2. 核心工作流

```
研报 sources/<pub>/<group>/{paper.md | inputs/<factor>.md}
    ↓ python -m core.spec_generator <pub>/<group>/<factor>      （LLM + spec_schema 校验闭环）
sources/<pub>/<group>/specs/<factor>/spec.yaml + .llm_session.json
    ↓ python run.py <factor>                                    （fetch + 算子图 + 评估）
raw → cleaned（MAD+zscore+mask）→ neu（强制行业市值中性化）→ output/<source>/<group>/<factor>/
      落 factors/{raw,cleaned,neu}/<source>/<group>/ 三阶段 + 两张 PNG（__cleaned/__neu）
    ↓ 沉淀
sources/<pub>/<group>/docs/<factor>.md                          （因子原理 + 工程经验）
```

**因子源归类（sources/ 顶层）**：
- `kysec/` / `founder/` — 券商研报（每篇研报一个 `paper_<NN>_<slug>/`，paper.md 共享）
- `fundamental/` — 经典基本面，无券商来源（按系列分：`npf_series/` `roe_series/` `roic_series/` `pe_series/` `cross_section_regress/` `cashflow_series/`，**inputs/<factor>.md 复数**，每因子一份）
- `internal/` — 同事/内部因子（按贡献人或主题分）

---

## 3. 常用命令

```bash
# spec 生成（rare，慢）—— 新建必须用限定路径
python -m core.spec_generator fundamental/npf_series/npf_mrq_sue8

# 全流程：YOLO + 评估（裸名 OK；重名时用 <pub>/<group>/<factor> 消歧）
python run.py roe_mrq_new
python run.py roe_mrq_new --yolo-only        # 只 fetch + 落 raw
python run.py roe_mrq_new --evaluate-only    # 只评估（已有 raw）
python run.py roe_mrq_new --start-date 20200101 --end-date 20251231 --workers 16

# 批量：按 sources 子目录循环
for f in $(ls sources/kysec/paper_27_microstructure/specs); do python run.py $f --evaluate-only; done

# Factor Inventory：所有 panel 因子总账（~2 分钟，100 workers, 197 因子）
python scripts/build_factor_inventory.py

# 因子相关性：BLAS GEMM 一次算 N² + 层次聚类，输出矩阵/热力图/summary
python scripts/factor_correlation.py --pattern 'pj_*' --name paper_33   # ~10s for 18 因子

# 错过赢家 SHAP 月度诊断（模型风格洞监控，参数在脚本顶部改，~3 分钟）
# 自动找期间涨幅 top20 → 判定抓住/错过 → 对错过者逐日 SHAP 解剖（拖累因子定位）
# 产出: 控制台报告 + ml/diagnostics/missed_winners_shap/ 落盘 audit trail
# 建议每月跑一次（改 PERIOD_START/END），监控"错过赢家占比"与拖累因子族是否漂移
PYTHONPATH=. python scripts/diagnose_missed_winners_shap.py
```

**CLI 设计原则**：
- **生成与执行分离**：`spec_generator`（rare）vs `run.py`（daily）
- **裸名优先**：自动扫 `sources/*/*/specs/<factor>/`；重名强制 `<pub>/<group>/<factor>`；新建 spec 必须用限定路径
- **论文论断必须引证 paper.md**：任何关于"研报怎么定义"的论断，先 grep `paper.md` 找行号给引用，否则视为待验证假设（曾踩幻觉坑）

---

## 4. Spec 规范（spec.yaml 是唯一源真相）

每个 `sources/<pub>/<group>/specs/<factor>/` 下**必须有** `spec.yaml`，可选 `.llm_session.json`。
完整 schema + few-shot 示例见 `prompts/research_to_yaml.md`，校验由 `core/spec_schema.py` 强制。

核心契约（违反 → 静态校验 raise）：
1. `factor.column` 必填，且必须等于某个 step 的 `output_column`
2. 每个 transform/compute/rank/rolling/row_polyfit/row_correlate 必须显式声明 `source_column[s]` + `output_column`
3. 同一 DataFrame 内 `output_column` 不允许重复
4. 多字段 fetch 用 `output_columns: {field: col}`；多 DataFrame 用 `output_dataframe` + 显式 `merge` step

**因子变体（A/B 实验）`<base>__<tag>`**：参数实验时新建并列目录，原 spec/parquet 一字不动。tag 自描述（`mp10`/`agg20d`，不要 `v2`）；`factor.name` + `factor.column` 同步成 `<base>__<tag>`；`description` 首段写"Variant of <base>; 改了什么; 动机"。淘汰档 `rm -rf` 即可。

---

## 5. 关键算子约定

1. **米筐认证**：YOLO 前 `rqdatac.init()` 成功
2. **PIT 模式**：财务用 `get_pit_financials_ex`，公告日字段 `info_date`；`statements='all'` 才能拿全版本
3. **`_mrq_n` 是单季度值**：`net_profit_mrq_0` 等已预计算单季度，**禁止再 diff**
4. **市值用 `market_cap_3`**（米筐有 `_2`/`_3`，约定 `_3`）
5. **TTM 财务字段命名（以官方 get_factor 文档为准）**：
   - **三大报表基础会计科目**（net_profit / revenue 等）：蛇形 + 数字尾缀 `_ttm_0`，即 `net_profit_ttm_0` / `revenue_ttm_0`
   - **衍生比率**（pe/pb 等）：蛇形无数字 `pe_ratio_ttm` / `pb_ratio_lf`
   - ⚠️ **不要用驼峰**：`net_profitTTM` 是未文档化遗留字段，实测**≈ 但 ≠** `net_profit_ttm_0`（约 0.5% 单元/860 只股票有差异，覆盖更少，对科创板等新股行为异常）；`revenueTTM` 更**直接返回 None 取不到数**，照驼峰写会静默拿全 NaN。一律用 `_ttm_0`。
   - 实证见 `sources/cxl/cross_section_regress/docs/reg_pe_hist.md`（2026-05-29 字段修订记录）。
6. **季度数据 yoy=4，qoq=1**；资产负债表（净资产/总资产）是时点值，直接用不要 diff
7. **清洗因子保留完整时间范围**：评估按 `--start-date/--end-date` 动态截取

---

## 6. 复现新因子的 5 步标准动作

1. **写研报输入** `sources/<pub>/<group>/{paper.md | inputs/<factor>.md}`
2. **LLM 生成 spec** `python -m core.spec_generator <pub>/<group>/<factor>`，spec_schema 闭环校验
3. **跑全市场** `python run.py <factor>`，得 IC / ICIR / Sharpe / 单调性 / 覆盖率
4. **冒烟验证（推荐）**：单股 / 3 股端到端手算对照（纯 numpy + rqdatac）
5. **写沉淀文档** `sources/<pub>/<group>/docs/<factor>.md`（**不可省略**），含：因子定义（数学+经济直觉）/ 研报歧义+工程选择 / 实现高层逻辑 / 复现结果 / **非平凡洞察**（最值钱）/ 与基准差距 / 改进路径

参考模板：`sources/cxl/npf_series/docs/npf_mrq_sue8.md`（数学技巧）；`.../docs/npf_mrq_accs8.md`（关键洞察+冒烟方法+多版本对照）。

> 知识沉淀是 5000+ 因子尺度的**核心价值**——spec.yaml 让因子可执行，docs/<factor>.md 让因子的研究决策可追溯、可复用、可教学。**任何"非平凡发现"必须立刻写进 docs**，否则 6 个月后没人记得。

**沉淀文档约束**：每份 docs/<factor>.md **≤ 200 行**，宁可少写一句，不可多费一行。

---

## 7. 分钟级因子复现

详见 `docs/minute_factor_replication.md`。最常用三件事：
- 分钟数据 = `MINUTE_DATA_DIR`；spec 用 `universe.primary_index: MINUTE_DIR` 自动扫
- 全市场跑：`MINUTE_WORKERS=100 python run.py <factor>`（~50s，**不要用线程池版**）
- 改算子 superset 列定义时 bump `_FEATURES_SUPERSET_VERSION`；强制重算改 spec 的 `cache_key`

模板：`sources/kysec/paper_27_microstructure/specs/peak_minute_count/`（最简）；`.../peak_interval_kurt/`（高阶矩 + NaN 守门）。

---

## 8. Git 工作流

**Commit message ≤ 10 词**（subject 一行说清楚就够，不写 body）：
```bash
git commit -m "feat: add row_polyfit and row_correlate operators"
```

**避免 HEREDOC 嵌套** —— `git commit -m "$(cat <<'EOF' ... EOF)"` 内含 `cat` 子命令会触发二次审批 + 中断留下 `.git/index.lock` 残留；长 message 用 `git commit -F /tmp/msg.txt`。

**提交前**：`git status` 确认意图、`git diff --cached` 看改动；不要 `git add -A`（容易拉杂物）。

---

## 9. 目录速查

```
sources/             业务资产按"来源"分组
  kysec/ founder/    券商研报（paper.md 共享，specs/<factor>/, docs/<factor>.md）
  fundamental/       经典基本面（按系列分；inputs/<factor>.md 复数；specs/, docs/）
  internal/          同事/内部因子
prompts/             LLM prompt（research_to_yaml.md = spec 生成 schema）
core/
  config.py          路径常量 + DEFAULT_START_DATE / DEFAULT_END_DATE
  spec_schema.py     spec.yaml 静态校验
  spec_generator.py  LLM 单段式生成 + 重试闭环
  spec_resolver.py   因子标识符 → 路径解析
  operators/         fetch / compute / filter / rank / rolling / transform / merge /
                     row_aggregate / row_polyfit / row_correlate /
                     rolling_weighted_mean / rolling_sorted_subset / load_panel /
                     minute_intraday_aggregate / cross_section_regress
  evaluation.py      单因子评估编排（清洗→强制行业市值中性化→cleaned/neu 各评一版）
  eval_plots.py      评估可视化（2×2 报告 PNG，本地审美）
  yolo_engine.py     spec yaml → 算子图执行
output/<source>/<group>/<factor>/  评估产物，gitignore；两张图 evaluation_<range>__{cleaned,neu}.png
run.py               日常 CLI（默认 yolo + 评估，裸名/限定路径都接受）
scripts/             一次性迁移脚本 + factor_inventory + factor_correlation 等
docs/                项目级架构文档
ml_ht/               华泰人工智能系列复现（独立，不碰 ml/ 和 core/；详见 §10）
```

`.gitignore` 已忽略 `output/`、`*.bak/`、`*.parquet`、`__pycache__/`、`.env`。

---

## 10. ml_core — ML 选股管线内核

> 统一的 ML 选股管线内核，配置驱动，LGBM / MLP 一套切换。
> 训练 `python -m ml_core.run`（改 `ml_core/train_config.yaml`）、推理 `python -m ml_core.predict`（改 `ml_core/predict_config.yaml`）。
> 详见 `user_guide.md`。

### can_train 过滤器设计

`can_train(T, X)` 回答"股票 X 在 T 日能否进入训练集"，三个独立条件的交集：

| 条件 | 含义 | 实现 | 时间语义 |
|------|------|------|---------|
| `has_factor` | 全部 158 因子非 NaN | `~isnan(factor_stack).any(axis=0)` | T 当天 |
| `can_buy` | T+1 非 ST/停牌/新股 | 原始 mask 查 T+1（无 shift） | 看 T+1 |
| `has_label` | 远期收益可计算 | `forward_return_20d.notna()` | T+1~T+21 |

**设计原则**：
- **无 shift 魔术**：`can_buy` 直接查 `mask[T+1]`，不预移整个矩阵
- **涨停不过滤**：涨停股票的因子值和标签均可观测，执行约束交给外部回测系统
- **has_factor 用 all**：MLP 需要 158 个输入全部有限，任一 NaN 即排除

### 已知问题

alpha158 因子存在 NaN 不一致性（111/158 个因子的 NaN 位置与 KMID 不同）：
- **CNT 家族 bug** (已修复 2026-06-28): `(close > Ref(close,1)).astype(float)` 在 NaN 处返回 0.0 而非 NaN, 让停牌/未上市/退市区被误算成"非上涨日 0.0". 修复: `factors.py:CNTP/CNTN` 加 `.where(close.notna() & close_lag.notna())` 显式恢复 NaN. SUMP/SUMN/SUMD 不受影响 (pc=NaN 时 `NaN * 0 = NaN` IEEE-754 自然传播)
- **滚动窗口预热**：BETA/RSQR/CORR 等使用 `sliding_window_view` 的因子，新股上市前 w 天有额外 NaN
- **VWAP0**：volume=0 时除零产生 NaN
- 使用 `has_factor = all` 过滤后，这些不一致性不影响 ML 训练（NaN 行被整体排除）

**⚠️ NaN 填充优化（TODO，后续解决）**：
当前 `has_factor=all` 过于严格——被过滤的 573,938 行中：
- 60.6% 只有 1~2 个 NaN（158 个因子中仅缺 1~2 个）
- 中位数 NaN 因子数 = 4，75 分位 = 10
- 提议：NaN ≤ 5 个的行用截面中位数填充（救回 ~70% 行），> 5 个的丢弃
- 当前先用严格过滤推进，后续迭代时再加填充逻辑
