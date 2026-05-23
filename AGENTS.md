# 因子复现 Agent —— CLI 必读

> 研报/文本 → 自动复现为可执行量化因子（基于米筐 RQData）

---

## 1. 环境与路径

```bash
source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate  # Python 3.11
pip install -e /nfs/volume-1593-1/peterzhenglinpeng/alpha-shared    # 共享原语库（首次配置）
```

| 变量 | 路径 | 说明 |
|------|------|------|
| `RAW_FACTOR_DIR` | `/nfs/ofs-prediction/peterzhenglinpeng/my-alpha-engine/factor-panel/spec/` | 原始因子（spec engine 产物，统一 panel 命名空间） |
| `CLEANED_FACTOR_DIR` | `/nfs/ofs-prediction/peterzhenglinpeng/my-alpha-engine/cleaned-factor-panel/spec/` | 清洗后因子（MAD + zscore + mask） |
| `OUTPUT_DIR` | `factor-repilcation-quant/output/` | 评估产物（png + report） |
| `COMBO_MASK_PATH` / `NEW_STOCK_MASK_PATH` / `VWAP_POST_PATH` | `.../backtest_engine/cache_dir/` | 预计算 mask + vwap，评估时**零 API 调用** |

预计算数据已更新到 2026-05-15。

**alpha-shared 共享库**：因子评估的 IC / 分层回测 / 清洗 / mask 加载等 primitive
统一抽到 `/nfs/volume-1593-1/peterzhenglinpeng/alpha-shared/`（独立 git 仓），
本仓 `core/evaluation/{ic,layered,returns}.py` 与 `core/cleaning/{preprocess,
mask_loader}.py` 都是 thin wrapper 转发到 `alpha_shared.*`。改算法去 alpha-shared
改一份；改完务必跑 `scripts/regression_baseline_replication.py` + `regression_compare.py`
确认两边数值零漂移。

---

## 2. 核心工作流

```
研报 sources/<pub>/<group>/{input.md | inputs/<factor>.md}
    ↓ python -m core.spec_generator <pub>/<group>/<factor>      （LLM + spec_schema 校验闭环）
sources/<pub>/<group>/specs/<factor>/spec.yaml + .llm_session.json
    ↓ python run.py <factor>                                    （fetch + 算子图 + 评估）
raw_factor/<factor>.parquet
    ↓ 清洗 + 评估
cleaned_factor/<factor>.parquet + output/<factor>/{evaluation_*.png, report.md}
    ↓ 沉淀
sources/<pub>/<group>/docs/<factor>.md                          （因子原理 + 工程经验）
```

**因子源归类（sources/ 顶层）**：
- `kysec/` — 开源证券（每篇研报一个 `paper_<NN>_<slug>/` 子目录，input.md 共享）
- `founder/` — 方正证券（同上）
- `fundamental/` — 经典基本面，无券商研报来源（按系列分：`npf_series/`、`roe_series/`、`roic_series/`、`pe_series/`、`cross_section_regress/`，**inputs/<factor>.md 为复数**，每因子一份）
- `internal/` — 同事/内部因子（按贡献人或主题分）

---

## 3. 常用命令

```bash
# 1) 从研报生成 spec.yaml（LLM + 静态校验闭环）—— 新建必须用限定路径
python -m core.spec_generator kysec/paper_27_microstructure/peak_minute_count
python -m core.spec_generator fundamental/npf_series/npf_mrq_sue8

# 2) 默认全流程：YOLO + 评估（裸名 OK，自动扫 sources/ 定位）
python run.py roe_mrq_new
python run.py kysec/paper_27_microstructure/peak_minute_count   # 限定路径，重名时消歧

# 3) 仅 YOLO / 仅评估
python run.py roe_mrq_new --yolo-only
python run.py roe_mrq_new --evaluate-only

# 4) 自定义区间 + 并发
python run.py roe_mrq_new --start-date 20200101 --end-date 20251231 --workers 16

# 5) 批量回归用 shell 循环（按 sources 子目录）
for f in $(ls sources/kysec/paper_27_microstructure/specs); do python run.py $f --evaluate-only; done
```

**CLI 设计原则**：
- **spec 生成与执行分离**：`python -m core.spec_generator`（rare，慢）vs `python run.py`（daily，快）
- **裸名 vs 限定路径**：执行时裸名优先（自动扫 `sources/*/*/specs/<factor>/`），重名时强制用 `<pub>/<group>/<factor>` 消歧；新建 spec 必须用限定路径
- **约定优于配置**：研报输入路径由 group 目录推断（`group/inputs/<factor>.md` 或 `group/input.md`），CLI 不重复传

---

## 4. Spec 规范（spec.yaml 是唯一源真相）

每个因子目录 `sources/<pub>/<group>/specs/<factor>/` 下**必须有** `spec.yaml`，可选 `.llm_session.json`（LLM 完整对话日志）。

**spec.yaml 是机器可执行的因子定义**，结构、字段、契约由 `core/spec_schema.py` 强制校验。完整 schema + few-shot 示例见 `prompts/research_to_yaml.md`。

核心契约（违反 → 静态校验 raise）：
1. `factor.column` 必填，且必须等于某个 step 的 `output_column`
2. 每个 transform/compute/rank/rolling/row_polyfit/row_correlate 必须显式声明 `source_column[s]` 和 `output_column`
3. 同一 DataFrame 内 `output_column` 不允许重复（防覆盖）
4. 多字段 fetch 用 `output_columns: {field: col}` 映射；多 DataFrame 用 `output_dataframe` + 显式 `merge` step

**spec.md 不是规范**——曾经是双文件强制，现在已淘汰。人类阅读用 `sources/<pub>/<group>/docs/<factor>.md`（见 §6）。

---

## 5. 关键算子约定

1. **米筐认证**：YOLO 前 `rqdatac.init()` 成功
2. **PIT 模式**：财务数据用 `get_pit_financials_ex`，公告日字段是 `info_date`，注意 `statements='all'` 才能拿到所有版本（含原始 + 重述）
3. **`_mrq_n` 字段是单季度值**：米筐 `net_profit_mrq_0` 等后缀字段**已经预计算为单季度值**，**禁止再 diff**
4. **市值字段统一用 `market_cap_3`**：米筐有 `market_cap` / `_2` / `_3`，项目约定 `_3`
5. **米筐 TTM 财务字段是驼峰**：`net_profitTTM` / `revenueTTM` / `operating_revenueTTM`，不是 `net_profit_ttm`；但比率类仍是蛇形：`pe_ratio_ttm` / `pb_ratio_lf`
6. **季度数据 yoy=4，qoq=1**
7. **资产负债表（净资产、总资产）是时点值**：直接用，不要 diff
8. **清洗因子保留完整时间范围**：评估时根据 `--start-date/--end-date` 动态截取

---

## 6. 复现新因子的标准动作（含知识沉淀）

**复现一个新因子的完整流程必须走完以下 5 步**：

1. **写研报输入**：`sources/<pub>/<group>/{input.md | inputs/<factor>.md}`（券商 paper 共享 input.md；fundamental 系列每因子一份 inputs/<factor>.md）
2. **LLM 生成 spec**：`python -m core.spec_generator <pub>/<group>/<factor>`（新建必须用限定路径），让 spec_schema 闭环校验通过
3. **跑全市场**：`python run.py <factor>`（裸名 OK），得到 IC / ICIR / Sharpe / 单调性 / 覆盖率
4. **冒烟验证（推荐）**：单股 / 3 股端到端手算对照（纯 numpy + rqdatac），理解因子真实行为，10 分钟换避免误读全市场结果
5. **写沉淀文档**：`sources/<pub>/<group>/docs/<factor>.md`（**不可省略**），至少包含：
    - **因子定义**（数学 + 经济直觉）
    - **研报描述歧义** + 我们的工程选择 + 理由
    - **实现 N 步 spec** 高层逻辑
    - **复现结果**（IC / ICIR / Sharpe / 单调性 / 覆盖率）
    - **任何"非平凡的洞察"**（这一栏最值钱，例：npf_mrq_accs8 发现"前后 4 期相关性过滤其实是季节性过滤"）
    - **与基准的差距分析**（如有研报基准）
    - **未来改进路径**（v2 / v3 候选）

**参考模板**：
- `sources/fundamental/npf_series/docs/npf_mrq_sue8.md` —— 含数学技巧（望远镜求和 + 方差恒等式）
- `sources/fundamental/npf_series/docs/npf_mrq_accs8.md` —— 含关键洞察（季节性过滤）+ 单股冒烟测试方法 + 多版本对照

> 知识沉淀是这个系统在 5000+ 因子尺度上的**核心价值**——spec.yaml 让因子可执行，docs/<factor>.md 让因子的研究决策可追溯、可复用、可教学。**任何"非平凡的发现"必须立刻写进 docs**，否则 6 个月后没人记得。

**沉淀文档约束**：每份 docs/<factor>.md **行数 ≤ 200**，找最精炼、最重点的表达，避免冗余啰嗦。宁可少写一句，不可多费一行。

---

## 7. 分钟级因子复现（开源_微观_27 等高频研报）

**数据**：`/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir/stock_data_1m_post/`
- per-stock parquet（`<order_book_id>.parquet`），列 `[datetime, open, high, low, close, volume, total_turnover]`
- 用户日更，5455+ 只股票，2010-至今
- 路径常量：`core.config.MINUTE_DATA_DIR`

**股票池声明**：spec 写 `universe.primary_index: MINUTE_DIR`，`build_universe()` 自动扫目录返回 `[0-9]*.XSH[EG]` parquet stem。

**核心算子 `minute_intraday_aggregate`**（`core/operators/minute_intraday_aggregate.py`）：
- 一只股票一只股票流式：load → 同时点 σ → 喷发标签 → 峰/岭/谷分类 → 日频 reduce
- 输出**日频** long 表（`(order_book_id, date, *features)`），与下游 `rolling`/`compute`/`rank` 完全兼容
- spec 字段：`cache_key`（必填）、`features`（必填，superset 子集）、`std_window` (默认 20)、`std_threshold` (默认 1.0)
- **superset 列**（spec.features 必须从中挑）：见算子文件 `_SUPERSET_COLUMNS`，含三类 count/vol_sum/turnover_sum/vwap、峰/岭间隔 5 阶矩、喷发后下一分钟成交额、日频价/量
- warmup 日（前 std_window 日）所有 feature 列输出 NaN（不是 0）

**缓存机制（per-stock parquet，应对日更）**：
- 路径：`INTERMEDIATE_CACHE_DIR / <cache_key>__h<params_hash> / <order_book_id>.parquet`
- `params_hash = sha1(std_window, std_threshold, _FEATURES_SUPERSET_VERSION)[:10]` —— 任何参数变化自动 cache miss
- 命中：source mtime ≤ cache mtime → load；miss → 重算 + atomic rename 写盘
- 用户日更某只 source parquet → 该股 cache 单独失效，其他不动
- **bump cache_key 规则**：算子 superset 列定义变更时，bump `_FEATURES_SUPERSET_VERSION`（自动失效）；只是想强制重算时改 spec 的 `cache_key` 字段（如 `prv_v1` → `prv_v2`）

**并发**：环境变量 `MINUTE_WORKERS`（默认 8）控制 **ProcessPoolExecutor**（真多进程，绕开 GIL）并发度。128 核机器请用 `MINUTE_WORKERS=100`（实测 64 worker 已能 55 stocks/sec、一遍全市场 ~62 秒；100 worker 进一步压到 ~40-60 秒区间，NFS bandwidth 接近瓶颈）。**线程池版（旧）实测仅 ~3 核效率，不要用**。

**`compute` 算子的 nan 用法**：`_BUILTINS` 已包含 `"nan"` 字面量；但 `pd.eval` **不支持 `where()` 函数**（即使在 `_BUILTINS` 里），实现条件 NaN 用"divide-by-mask"惯用法：
```yaml
formula: kurt_raw * (gate / gate)   # gate=0 → 0/0=NaN；gate=1 → 1/1=1
```

**典型 spec 模板**：见 `sources/kysec/paper_27_microstructure/specs/peak_minute_count/`（最简）和 `.../specs/peak_interval_kurt/`（高阶矩 + 守门 NaN）。

---

## 8. Git 工作流

**Commit message 约定 ≤ 10 词**（subject 一行说清楚就够，不写 body）：

```bash
git commit -m "feat: add row_polyfit and row_correlate operators"   # ✅
git commit -m "fix(prompt): drop misleading anti-row_aggregate guidance"  # ✅
```

**避免 HEREDOC 嵌套** —— `git commit -m "$(cat <<'EOF' ... EOF)"` 内含 `cat` 子命令会触发 Claude Code 二次审批 + 中断留下 `.git/index.lock` 残留；长 message 用 `git commit -F /tmp/msg.txt`。

**提交前检查**：
- `git status` 确认只提交意图中的文件
- `git diff --cached` 看改动内容
- 不要 `git add -A`（容易把 untracked 杂物拉进来）

---

## 9. 目录速查

```
sources/                          ★ 业务资产按"来源"分组
  kysec/                          开源证券
    paper_27_microstructure/
      input.md                    研报原文（paper-level 共享）
      README.md                   论文元信息（作者/日期/共享方法论）
      specs/<factor>/             spec.yaml + .llm_session.json
      docs/<factor>.md            因子沉淀
  founder/                        方正证券（同上结构）
  fundamental/                    经典基本面，无券商研报来源
    npf_series/  roe_series/  roic_series/  pe_series/  cross_section_regress/
      inputs/<factor>.md          ★ 复数：每因子一份小 input
      specs/<factor>/  docs/<factor>.md
  internal/                       同事/内部因子（按贡献人或主题分）
prompts/                          LLM prompt（research_to_yaml.md = spec 生成 schema）
core/
  config.py                       路径 + DEFAULT_START_DATE / DEFAULT_END_DATE
  spec_schema.py                  spec.yaml 静态校验
  spec_generator.py               LLM 单段式生成 + 重试闭环
  spec_resolver.py                因子标识符 → 路径解析（裸名 / 限定路径）
  operators/                      算子库（fetch / compute / filter / rank / rolling /
                                  transform / merge / row_aggregate / row_polyfit /
                                  row_correlate / minute_intraday_aggregate /
                                  cross_section_regress）
  cleaning/                       MAD + zscore + mask
  evaluation/                     IC + 分层 + 绘图
  yolo_engine.py                  spec yaml → 算子图执行
output/<factor>/                  评估产物 (png + report.md)，gitignore；按 factor 名 flat
run.py                            日常执行 CLI（默认 yolo + 评估，裸名/限定路径都接受）
scripts/migrate_to_sources.py     一次性迁移脚本（保留为审计痕迹）
docs/                             项目级架构文档（如 multi_factor_paper_architecture.md）
```

`.gitignore` 已忽略 `output/`、`*.bak/`、`*.parquet`、`__pycache__/`、`.env`。
