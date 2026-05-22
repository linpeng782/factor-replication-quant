# 因子复现 Agent —— CLI 必读

> 研报/文本 → 自动复现为可执行量化因子（基于米筐 RQData）

---

## 1. 环境与路径

```bash
source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate  # Python 3.11
```

| 变量 | 路径 | 说明 |
|------|------|------|
| `RAW_FACTOR_DIR` | `/nfs/ofs-prediction/peterzhenglinpeng/factor-replication/raw_factor/` | 原始因子（YOLO 输出） |
| `CLEANED_FACTOR_DIR` | `/nfs/ofs-prediction/peterzhenglinpeng/factor-replication/cleaned_factor/` | 清洗后因子（MAD + zscore + mask） |
| `OUTPUT_DIR` | `factor-repilcation-quant/output/` | 评估产物（png + report） |
| `COMBO_MASK_PATH` / `NEW_STOCK_MASK_PATH` / `VWAP_POST_PATH` | `.../backtest_engine/cache_dir/` | 预计算 mask + vwap，评估时**零 API 调用** |

预计算数据已更新到 2026-05-15。

---

## 2. 核心工作流

```
研报 inputs/<factor>.md
    ↓ python -m core.spec_generator <factor>           （LLM + spec_schema 校验闭环）
specs/<factor>/spec.yaml + .llm_session.json
    ↓ python run.py <factor>                           （fetch + 算子图 + 评估）
raw_factor/<factor>.parquet
    ↓ 清洗 + 评估
cleaned_factor/<factor>.parquet + output/<factor>/{evaluation_*.png, report.md}
    ↓ 沉淀
docs/<factor>.md                                       （因子原理 + 工程经验）
```

---

## 3. 常用命令

```bash
# 1) 从研报生成 spec.yaml（LLM + 静态校验闭环）
python -m core.spec_generator npf_mrq_sue8         # 读 inputs/npf_mrq_sue8.md

# 2) 默认全流程：YOLO + 评估
python run.py roe_mrq_new

# 3) 仅 YOLO / 仅评估
python run.py roe_mrq_new --yolo-only
python run.py roe_mrq_new --evaluate-only

# 4) 自定义区间 + 并发
python run.py roe_mrq_new --start-date 20200101 --end-date 20251231 --workers 16

# 5) 批量回归用 shell 循环（CLI 不内置 --all）
for f in $(ls specs); do python run.py $f --evaluate-only; done
```

**CLI 设计原则**：
- **spec 生成与执行分离**：`python -m core.spec_generator`（rare，慢）vs `python run.py`（daily，快）
- **约定优于配置**：`inputs/<FACTOR>.md`、`specs/<FACTOR>/spec.yaml`，路径不在 CLI 里反复传

---

## 4. Spec 规范（spec.yaml 是唯一源真相）

每个因子目录 `specs/<factor>/` 下**必须有** `spec.yaml`，可选 `.llm_session.json`（LLM 完整对话日志）。

**spec.yaml 是机器可执行的因子定义**，结构、字段、契约由 `core/spec_schema.py` 强制校验。完整 schema + few-shot 示例见 `prompts/research_to_yaml.md`。

核心契约（违反 → 静态校验 raise）：
1. `factor.column` 必填，且必须等于某个 step 的 `output_column`
2. 每个 transform/compute/rank/rolling/row_polyfit/row_correlate 必须显式声明 `source_column[s]` 和 `output_column`
3. 同一 DataFrame 内 `output_column` 不允许重复（防覆盖）
4. 多字段 fetch 用 `output_columns: {field: col}` 映射；多 DataFrame 用 `output_dataframe` + 显式 `merge` step

**spec.md 不是规范**——曾经是双文件强制，现在已淘汰。人类阅读用 `docs/<factor>.md`（见 §6）。

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

1. **写研报输入**：`inputs/<factor>.md`（一段研报描述即可，越简洁越能暴露 prompt 设计强度）
2. **LLM 生成 spec**：`python -m core.spec_generator <factor>`，让 spec_schema 闭环校验通过
3. **跑全市场**：`python run.py <factor>`，得到 IC / ICIR / Sharpe / 单调性 / 覆盖率
4. **冒烟验证（推荐）**：单股 / 3 股端到端手算对照（纯 numpy + rqdatac），理解因子真实行为，10 分钟换避免误读全市场结果
5. **写沉淀文档**：`docs/<factor>.md`（**不可省略**），至少包含：
    - **因子定义**（数学 + 经济直觉）
    - **研报描述歧义** + 我们的工程选择 + 理由
    - **实现 N 步 spec** 高层逻辑
    - **复现结果**（IC / ICIR / Sharpe / 单调性 / 覆盖率）
    - **任何"非平凡的洞察"**（这一栏最值钱，例：npf_mrq_accs8 发现"前后 4 期相关性过滤其实是季节性过滤"）
    - **与基准的差距分析**（如有研报基准）
    - **未来改进路径**（v2 / v3 候选）

**参考模板**：
- `docs/npf_mrq_sue8.md` —— 含数学技巧（望远镜求和 + 方差恒等式）的因子
- `docs/npf_mrq_accs8.md` —— 含关键洞察（季节性过滤）+ 单股冒烟测试方法 + 多版本对照

> 知识沉淀是这个系统在 5000+ 因子尺度上的**核心价值**——spec.yaml 让因子可执行，docs/<factor>.md 让因子的研究决策可追溯、可复用、可教学。**任何"非平凡的发现"必须立刻写进 docs**，否则 6 个月后没人记得。

**沉淀文档约束**：每份 `docs/<factor>.md` **行数 ≤ 200**，找最精炼、最重点的表达，避免冗余啰嗦。宁可少写一句，不可多费一行。

---

## 7. Git 工作流

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

## 8. 目录速查

```
inputs/                  研报文字（每因子一份 .md）
prompts/                 LLM prompt（research_to_yaml.md = spec 生成 schema）
specs/<factor>/          spec.yaml + .llm_session.json（LLM 对话日志）
core/
  config.py              路径 + DEFAULT_START_DATE / DEFAULT_END_DATE
  spec_schema.py         spec.yaml 静态校验
  spec_generator.py      LLM 单段式生成 + 重试闭环
  operators/             算子库（fetch / compute / filter / rank / rolling /
                         transform / merge / row_aggregate / row_polyfit /
                         row_correlate）
  cleaning/              MAD + zscore + mask
  evaluation/            IC + 分层 + 绘图
  yolo_engine.py         spec yaml → 算子图执行
docs/<factor>.md         因子原理 + 工程经验沉淀（每因子一份）
output/<factor>/         评估产物 (png + report.md)，gitignore
run.py                   日常执行 CLI（默认 yolo + 评估）
```

`.gitignore` 已忽略 `output/`、`*.bak/`、`*.parquet`、`__pycache__/`、`.env`。
