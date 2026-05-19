# 因子复现 Agent —— CLI 必读

> 研报/文本 → 自动复现为可执行量化因子（基于米筐 RQData）

---

## 1. 环境与路径

```bash
source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate  # Python 3.11
```

| 变量 | 路径 |
|------|------|
| `FACTOR_OUTPUT_DIR` | `/nfs/ofs-prediction/peterzhenglinpeng/factor-replication/` |
| `OUTPUT_DIR` | `factor-repilcation-quant/output/` |
| `COMBO_MASK_PATH` | `.../backtest_engine/cache_dir/combo_mask_long.parquet` |
| `NEW_STOCK_MASK_PATH` | `.../backtest_engine/cache_dir/new_stock_mask_long.parquet` |
| `VWAP_POST_PATH` | `.../backtest_engine/cache_dir/vwap_post.parquet` |

预计算数据（mask/vwap）到 2026-05-15，评估时**零 API 调用**。

---

## 2. 核心工作流

```
inputs/ → specs/<factor>/{spec.md, spec.yaml} → confirm/<factor>/
  → YOLO (core/operators/) → raw_<factor>.parquet
  → 清洗 (core/cleaning/) → <factor>.parquet (完整时间范围)
  → 评估 (core/evaluation/) → evaluation.png + report.md
```

---

## 3. 常用命令

```bash
# 新增因子（手动模式，推荐）
python run.py --input "factor_name：描述" --mode manual

# 执行已确认因子（YOLO + 评估）
python run.py --factor roe_mrq_new --yolo-only --evaluate

# 仅评估已有因子（可任意指定区间）
python run.py --factor roe_mrq_new --evaluate-only --start-date 20160101 --end-date 20251231

# 批量执行
python run.py --batch-confirmed --start-date 20160101 --end-date 20251231
```

---

## 4. Spec YAML 核心结构

```yaml
factor:
  name: "roe_mrq_new"
  direction: 1
data_source:
  api: "get_factor"           # 或 get_pit_financials_ex
  fields: [...]
calculation_steps:
  - step: 1
    action: "fetch"           # fetch / compute / filter / transform / rank
    fields: [...]
  - step: 2
    action: "compute"
    formula: "roe = net_profit / total_equity"
universe:
  primary_index: "ALL"
```

---

## 5. 关键约束

1. **米筐认证**：YOLO 前必须 `rqdatac.init()` 成功
2. **PIT 模式**：财务数据用 `get_pit_financials_ex`，以 `info_date` 为公告日
3. **`_mrq_n` 字段是单季度值**：米筐 `_mrq_0` / `_mrq_4` 等后缀字段已经预计算为单季度值，**无需手动 diff**。例如 `net_profit_mrq_0` 就是最近一期单季度净利润
4. **清洗因子保留完整时间范围**：评估时根据 `start_date/end_date` 动态截取

---

## 6. Git 工作流

### Commit Message 规范（Conventional Commits）

```
<type>(<scope>): <subject>

<body>
```

| type | 用途 |
|------|------|
| `feat` | 新功能 |
| `fix` | 修复 bug |
| `docs` | 文档更新 |
| `refactor` | 重构（无功能变化） |
| `chore` | 构建/工具链改动 |

**例子：**
```
fix(pipeline): correct mask slicing with Timestamp range

docs(agents): add env paths and evaluation workflow

feat(evaluation): add skip_cleaning param
```

### 提交前检查清单

- [ ] `git status` 确认只提交意图中的文件
- [ ] `git diff --cached` 确认改动内容正确
- [ ] 提交信息符合 Conventional Commits 规范

---

## 7. 目录速查

```
inputs/              原始输入
doc/                 评估文档、可行性分析
prompts/             Prompt 模板
core/
  operators/         元操作注册中心
  cleaning/          因子清洗（MAD + zscore + mask）
  evaluation/        单因子评估（IC + 分层回测 + 绘图）
  config.py          路径配置
specs/               每个因子 spec.md + spec.yaml
confirm/             人机确认记录
output/              评估产物（png/md）
```

> `.gitignore` 已配置忽略 `output/`、`scripts/`、`*.parquet`、`__pycache__/`、`.env`。
