# reg_pe_hist 复现进展（暂存）

> 因子全名：经过历史增速变化调整的历史 PE 变化
> 类别：价值（嵌套截面回归剥离 → 估值情绪残差）
> 当前状态：**spec 已生成、算子已就位、待跑 YOLO 验证**

---

## 1. 因子定义（来自 `inputs/reg_pe_hist.md`）

三个 log 变量：
- `log_yoy = log(net_profit_mrq_0 / net_profit_mrq_4)` （要求两期均 > 0）
- `log_npttm = log(net_profitTTM)` （要求 > 0）
- `log_pe = log(pe_ratio_ttm)` （要求 > 0）

各做 60 个交易日 diff 后做嵌套截面回归：
1. 第一步：`delta_log_yoy ~ delta_log_npttm` → 残差 `residual_1`
   （单季同比变化中独立于 TTM 变化的部分）
2. 第二步：`delta_log_pe ~ residual_1` → 残差 `reg_pe_hist`
   （PE 变化中无法被独立利润动量解释的部分 = 估值情绪）

direction = -1（PE 异常上涨且无利润支撑 → 估值高估 → 预期低收益）。
基准：研报报告 IC=0.063，ICIR=0.85。

---

## 2. 已完成

### 2.1 算子建设
- 新增 `core/operators/cross_section_regress.py`：按 date 分组做截面 OLS，
  输出残差列；支持多 x、`min_samples` 阈值、NaN 传播；4 个单元测试通过。
- 注册到 `core/yolo_engine.py`。
- `core/spec_schema.py` 更新：
  - `COLUMN_ADDING_ACTIONS` 加入 `cross_section_regress`
  - `_collect_source_columns` 支持 `source_column_<role>` 单数后缀（如 `source_column_y`）。
- `prompts/research_to_yaml.md` 加入 `cross_section_regress` 文档。

### 2.2 Spec 生成（LLM 一遍过）
- `inputs/reg_pe_hist.md`：研报描述（含 log 比率、过滤条件、嵌套回归逻辑）。
- `python -m core.spec_generator reg_pe_hist` 一次通过 schema 校验。
- `specs/reg_pe_hist/spec.yaml`：10 步——
  1. fetch 4 字段 [net_profit_mrq_0, net_profit_mrq_4, net_profitTTM, pe_ratio_ttm]
  2. filter 全部 > 0
  3-5. 三个 log compute
  6-8. 三个 60 日 diff transform
  9. cross_section_regress 第一步 → residual_1
  10. cross_section_regress 第二步 → reg_pe_hist

### 2.3 已修复的坑
- **米筐 TTM 字段名是驼峰**：`net_profit_ttm` 不存在 → 拉回全 NaN → filter 把所有
  行 drop 掉 → evaluate 在空表上报 `KeyError 'icir'`。
  正确字段名是 `net_profitTTM`（驼峰、无下划线）。比率类（`pe_ratio_ttm` /
  `pb_ratio_lf`）仍是蛇形——这是米筐 API 的内部不一致。
  - 修复：spec.yaml 4 处替换 `net_profit_ttm` → `net_profitTTM`
  - 落入 `AGENTS.md` §5 与 `prompts/research_to_yaml.md` 规则 6.1，避免下次踩坑。

---

## 3. 待办（回来后继续）

### 3.1 立刻可跑
```bash
source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate
python run.py reg_pe_hist
```
（默认全流程：YOLO + 清洗 + 评估）

### 3.2 跑完之后
1. 看 `output/reg_pe_hist/report.md`：比对 IC / ICIR 与研报基准 0.063 / 0.85
2. 若 IC/ICIR 接近 → 写 `docs/reg_pe_hist.md`（按 AGENTS.md §6 的 5 步沉淀模板）
3. 若 IC/ICIR 偏离严重 → 排查方向：
   - filter 是否过严（log 要求三个字段全 > 0，可能损失大量样本）
   - cross_section_regress 的 `min_samples` 是否合理（默认 30）
   - `delta_log_pe` 的 60 日 diff 是否有 group_by=order_book_id（已确认 yes）
   - 是否需要按 date 截面去掉某些极端值（log 在边界爆炸）

### 3.3 沉淀文档要点（写 docs/reg_pe_hist.md 时）
- 嵌套截面回归的经济直觉：为何"先剥 TTM 再剥独立利润动量"而不是一次性双 x 回归？
  （答：作者想分离"利润 surprise"与"利润趋势"两层信息，残差化顺序传达因果假设）
- log 过滤副作用：要求三字段同时 > 0 会吃掉多少样本？（待统计）
- 对于亏损股的处理：被 filter 全掉，意味着此因子不覆盖亏损股——是否需要文档说明？

---

## 4. 引用文件

- `specs/reg_pe_hist/spec.yaml` —— 机器可执行因子定义
- `inputs/reg_pe_hist.md` —— 研报输入
- `core/operators/cross_section_regress.py` —— 新增算子
- `prompts/research_to_yaml.md` 规则 6.1 —— TTM 驼峰约定
- `AGENTS.md` §5 —— 同上约定（CLI 必读）
