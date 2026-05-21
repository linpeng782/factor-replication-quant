# 多因子论文的 spec 架构

> 一篇研报包含多个因子（如开源_微观_27 的 20 个因子）时，如何用本系统的 spec.yaml 模型表达？
> 写于 2026-05-21，源自对开源证券《市场微观结构系列（27）》和方正证券微观结构系列的分析。

---

## 1. 问题陈述

当前 spec.yaml 模型假设：**1 spec → 1 factor**。

但真实研报常常是"一个共享方法论 + N 个变体因子"。例：

| 论文 | 共享部分 | 因子变体数 |
|---|---|---|
| 开源_微观_27 | 分钟级"峰/岭/谷"分类 | 11 类 / 20 个 |
| 方正《适度冒险》 | 激增时刻 + 耀眼 5 分钟 | 1（含中间因子） |
| 方正《完整潮汐》 | 日内潮汐识别 | 3 |
| 方正《飞蛾扑火》 | 跳跃度 + 振幅修正 | 2-3 |

这类论文的算力开销集中在"共享部分"（拉分钟数据、特征工程），变体因子只是末端聚合不同。**naive 做法（每因子一份独立 spec）会重复跑共享部分 N 次，对分钟级数据是不可承受的。**

---

## 2. 三种候选架构

### 路 A：N 份独立 spec + fetch / 中间产物缓存层 ⭐ 推荐

每个因子一份 spec.yaml，**底层加 cache_key 机制**：第一次执行写缓存，后续命中直接读。

```yaml
- action: fetch
  api: get_minute
  fields: [volume, close]
  cache_key: minute_ohlcv_2013_2025          # ← 新增字段
  ...

- action: intraday_classify
  cache_key: peak_ridge_valley_classification # ← 中间特征也缓存
  ...

- action: row_aggregate         # 各因子在这里分叉
  source_columns: [...]
  output_column: peak_minute_count
```

**优点**：
- 每因子**独立 commit / 评估 / 审计** —— 5000 因子尺度的核心价值
- spec 模型不变，spec_schema 不扩展
- LLM 翻译压力小（每次只翻一个因子）
- 缓存层是一次性投资，**所有多因子论文受益**

**缺点**：
- spec 间共享步骤要复制粘贴（可用 yaml `&anchor`/`*ref` 缓解）
- 缓存层引入一次性工程成本（~半天）

### 路 B：多输出 spec（一份 spec 产 N factor）

扩展 `factor.column` 为 `factor.columns: list`，spec 末尾 pivot N 次。

**优点**：spec 间零重复。

**缺点**：
- 单因子失败 → 整 spec 跑废，违反"独立可审计"原则
- spec 极长（200+ 行），LLM 一次产 20 个 step 自洽性差
- spec_schema 需扩展（factor.column → factor.columns）

### 路 C：两层架构 —— 特征预处理脚本 + 轻量因子 spec

`tools/preprocess_<paper>.py` 一次性产出共享特征 parquet，每因子 spec 极轻（5 步内）。

**优点**：spec 全部干净；适合"特征工程极重 + 复用面广"的场景（适度冒险的"耀眼 5 分钟"窗口）。

**缺点**：特征预处理逃出 spec_schema 护城河；引入"特征注册中心"概念。

---

## 3. 推荐：路 A

理由（重要性排序）：

1. **5000 因子尺度下"独立可审计 / 可评估 / 可比较"是核心价值**。路 B 一份失败 20 个一起死，违反这个原则
2. **缓存层是基础设施投资，所有多因子论文受益** —— 比针对单篇论文的特殊架构 ROI 高
3. **spec 模型简单不变** —— spec_schema、LLM prompt、评估管线全部不变

路 C 仍是极端场景的 escape hatch（如适度冒险的"耀眼 5 分钟"算子表达不优雅时）。**A 默认，C 兜底。**

---

## 4. 配套需要的两件基础设施

### 4.1 工具：`factor_extractor`（前置 LLM）

把 1 篇长研报拆成 N 个因子各自的 `inputs/<factor_n>.md`：

```bash
python -m core.factor_extractor inputs/开源_微观_27.md
# 产出 inputs/peak_minute_count.md / inputs/ridge_minute_return.md / ...
```

**LLM 只做"列出所有因子 + 给每个写一段简短描述"**——不写完整公式，只指明"这个因子是什么 + 哪几段讲了它"。具体公式留给 spec_generator 阶段读原文细化。

工作量：~半天写 + 调 prompt。

### 4.2 算子层：fetch / 中间产物缓存

在 `core/operators/__init__.py` 加 `cache_key` 字段处理：

```python
# 伪代码
def with_cache(action_fn):
    def wrapped(ctx, step, fetcher):
        if 'cache_key' in step:
            cache_path = CACHE_DIR / f"{step['cache_key']}.parquet"
            if cache_path.exists():
                return load_to_ctx(cache_path)
        result = action_fn(ctx, step, fetcher)
        if 'cache_key' in step:
            write_cache(cache_path, result)
        return result
    return wrapped
```

工作量：~半天 + 测试。

---

## 5. 工作流（路 A + 两件基础设施）

```bash
# 1) LLM 把论文拆成 N 份单因子描述（rare，慢，~5 分钟一次性）
python -m core.factor_extractor inputs/开源_微观_27.md

# 2) 对每个因子分别生成 spec（~5 分钟 × N，可并行）
for f in inputs/peak_*.md inputs/ridge_*.md inputs/valley_*.md; do
    name=$(basename $f .md)
    python -m core.spec_generator $name
done

# 3) 全市场跑（第一个慢 30 分钟，后续秒过缓存）
for f in $(ls specs | grep -E 'peak|ridge|valley'); do
    python run.py $f
done

# 4) 各自评估（缓存命中后，每因子 1-2 分钟）
```

**算力账**：

| 阶段 | naive（无缓存） | 路 A 缓存方案 |
|---|---|---|
| fetch + 共享特征 | 30 分钟 × 20 = 10 小时 | **30 分钟一次** |
| 各因子聚合 | 1 分钟 × 20 | 1 分钟 × 20 |
| 评估 | 1 分钟 × 20 | 1 分钟 × 20 |
| **总计** | **~10.7 小时** | **~1 小时** |

10× 提速，且每因子独立。

---

## 6. 何时启动这个工程

不要立刻动手——按需求触发：

| 触发条件 | 应该做什么 |
|---|---|
| 团队明确要做 2+ 篇分钟级研报 | 启动 §4.1 + §4.2，约 1 天 |
| 只做单篇 + 单因子（如适度冒险） | 不需要，直接走现有"1 spec 1 factor"+ ad-hoc 预处理（路 C） |
| 完全不做分钟级方向 | 不启动，文档存档备查 |

---

## 7. 其他考虑

- **LLM 上下文不是瓶颈**：Kimi-k2.6 128k token，开源_微观_27 ~30k token 容量充裕。真实瓶颈是"一次推理产 N 个因子的自洽性"——`factor_extractor` 拆成 N 次单因子 prompt 规避
- **日频因子不受影响**：`cache_key` 是可选字段，不写不缓存；日频 fetch 本就 1 分钟内完成，不需要缓存

---

## 8. 与现有项目对齐的接入点

| 文档 / 代码 | 涉及的修改 |
|---|---|
| `core/spec_schema.py` | 不需要修改（cache_key 不进 schema 校验） |
| `core/operators/__init__.py` | 加 `with_cache` 装饰器 |
| `core/operators/fetch.py` | 在 fetch_minute 实现里默认带缓存能力 |
| `prompts/research_to_yaml.md` | 加 `cache_key` 字段使用说明 |
| `AGENTS.md §6` | "复现新因子的标准动作"末尾加一行 "多因子论文先跑 factor_extractor" |
| 新文件 `core/factor_extractor.py` | LLM 论文拆分工具 |
| 新文件 `prompts/paper_to_factors.md` | 拆分用 prompt |

---

