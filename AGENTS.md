# 因子复现 Agent —— CLI 必读

> 研报/文本 → 自动复现为可执行量化因子（基于米筐 RQData）

---

## 1. 环境与路径

```bash
# 远端 SSH 机器（高频因子产线，65 GB 分钟数据所在）
source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate   # Python 3.11
pip install -e /nfs/volume-1593-1/peterzhenglinpeng/alpha-shared     # 共享原语库（首次配置）

# 本机 macOS（基本面因子复现；高频因子留远端，详见 LOCAL_SETUP.md）
source /Users/didi/kdj/peterdidi/bin/activate                        # Python 3.11.9 venv（本机唯一 venv，勿再找）
pip install -e /Users/didi/kdj/alpha-shared                          # 共享原语库（editable，首次配置）
export FACTOR_REPL_DATA_ROOT=/Users/didi/DATA                      # 数据根重定向

# 本机路径速记（避免反复探测）：
#   venv         /Users/didi/kdj/peterdidi
#   本仓         /Users/didi/kdj/factor-repilcation-quant
#   alpha-shared /Users/didi/kdj/alpha-shared
#   数据根       /Users/didi/DATA  （RAW_FACTOR_DIR 下本机有 ~22 个基本面 raw 因子）
```

数据根 `<DATA_ROOT>` 下**两个角色桶**：`factors/`（因子产出）+ `market-data/`（评估输入）。

| 变量 | 路径 | 说明 |
|------|------|------|
| `RAW_FACTOR_BASE` / `CLEANED_FACTOR_BASE` / `NEU_FACTOR_BASE` | `<DATA_ROOT>/factors/{raw,cleaned,neu}/<source>/<group>/` | 因子三阶段；namespace=`<source>/<group>` 由 spec 路径推导（cxl/cross_section_regress/...） |
| `COMBO_MASK_PATH` / `NEW_STOCK_MASK_PATH` | `<DATA_ROOT>/market-data/masks/` | 交易状态 mask，评估**零 API 调用** |
| `VWAP_PANEL_PATH` / `VWAP_POST_PATH` / `LABELS_DIR` | `<DATA_ROOT>/market-data/{prices,labels}/` | PIT vwap + 远期收益 labels |
| `INDUSTRY_PANEL_ZX_PATH` / `MARKET_CAP_PANEL_PATH` | `<DATA_ROOT>/market-data/{industry,market_cap}/` | 中信行业 + 总市值面板（stock-data-fetching 产出，中性化用） |
| `OUTPUT_DIR` | `factor-repilcation-quant/output/<source>/<group>/<factor>/` | 评估产物（两张 PNG） |

`<DATA_ROOT>` 默认 `/nfs/ofs-prediction/peterzhenglinpeng`；预计算数据已更新到 2026-05-15。

**本机化运行**：`export FACTOR_REPL_DATA_ROOT=/Users/didi/DATA` 一键把所有数据路径
重定向到本机；远端不设环境变量行为零变化。`OUTPUT_DIR` 是项目相对路径不受影响。

**数据起始统一 2005-01-01**：日频（raw OHLCV / ex-factors）与分钟频（1m）数据**一律从 2005-01-01 下载**（rqdatac 分钟最早即 2005-01-04）。复权口径：**价格后复权（× `ex_cum_factor`，ffill 到当日），成交量 / 成交额不复权**；复权因子用本地 `<DATA_ROOT>/market-data/my-alpha-engine-meta/stock-ex-factors/`（可日更，与米筐 `adjust_type='post'` 口径一致，已逐分钟 bit 级验证）。分钟拉取配方：`get_price(adjust_type='none')` 取原始 → 价格 ×`ex_cum_factor` → 量/额原样 → float32 落 `MINUTE_DATA_DIR`。

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

**alpha-shared 共享库**：IC / 分层回测 / 清洗 / mask 加载等**数值算法**全部在
`/nfs/volume-1593-1/peterzhenglinpeng/alpha-shared/`（本机 `/Users/didi/kdj/alpha-shared`，
独立 git 仓）。本仓**不再镜像其目录结构、无 thin wrapper**——`core/evaluation.py`
直接 `from alpha_shared.{evaluation,cleaning}...` 取算法，mask 路径在调用点注入 config。
改算法去 alpha-shared 改一份；改完务必跑 `scripts/regression_baseline_replication.py`
+ `regression_compare.py` 验两边数值零漂移。本仓评估代码只剩两个本地文件：
`core/evaluation.py`（编排 evaluate_single_factor）+ `core/eval_plots.py`（可视化，本地审美保留）。

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
                     minute_intraday_aggregate / cross_section_regress
  evaluation.py      单因子评估编排（清洗→强制行业市值中性化→cleaned/neu 各评一版）
  eval_plots.py      评估可视化（2×2 报告 PNG，本地审美）
  yolo_engine.py     spec yaml → 算子图执行
output/<source>/<group>/<factor>/  评估产物，gitignore；两张图 evaluation_<range>__{cleaned,neu}.png
run.py               日常 CLI（默认 yolo + 评估，裸名/限定路径都接受）
scripts/             一次性迁移脚本 + factor_inventory + factor_correlation 等
docs/                项目级架构文档
```

`.gitignore` 已忽略 `output/`、`*.bak/`、`*.parquet`、`__pycache__/`、`.env`。
