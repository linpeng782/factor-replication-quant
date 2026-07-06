# alpha158 自包含重构方案

> 2026-07-06。将 alpha158 从"脚本散在根目录 + 引擎散在 core/producers/"重构为自包含包。

---

## 1. 问题

当前 alpha158 代码分散在两个目录：

```
alpha158/                          ← 操作脚本（5 个）
  build.py / daily_update.py / smoke_replay.py / verify_repro.py / plot.py

core/producers/alpha158/           ← 计算引擎（5 个模块）
  factors.py / panel_operators.py / groups.py / adjusted_panels.py / __init__.py
```

**问题**：
- alpha158 不自包含——引擎在 core 里，脚本在根目录，新人不知道 alpha158 完整代码在哪
- `adjusted_panels.py`（后复权加载）是通用数据工具，不是 alpha158 专属逻辑，却放在 `core/producers/alpha158/` 里
- `core/operators/industry_co_momentum.py` 为了用 `adjusted_panels.py`，被迫依赖 `core/producers/alpha158/`——core 内部产生了不必要的耦合

## 2. 设计原则

1. **alpha158 自包含**：引擎 + 脚本 + 文档全部收敛到 `alpha158/` 一个目录
2. **后复权是通用基础设施**：`adjusted_panels.py` 抽到 `core/data/`，任何需要后复权价格的模块都能用，不绑定 alpha158
3. **依赖方向正确**：`core/data/` 是最底层（不依赖任何人），alpha158 依赖 `core/data/`，core 不依赖 alpha158
4. **删除 `core/producers/`**：alpha158 移走后变空，将来 alpha191 等也用同样的自包含模式

## 3. 目标目录结构

```
alpha158/                          ← 自包含包
  engine/                          ← 原 core/producers/alpha158/ 的因子引擎
    __init__.py                    ← 导出 Alpha158Panel, PanelOperators
    factors.py                     ← 158 因子定义
    panel_operators.py             ← 宽表算子底座
    groups.py                      ← 因子分组
  build.py                         ← 全量生产脚本
  daily_update.py                  ← 增量日更脚本
  smoke_replay.py                  ← 增量正确性对账
  verify_repro.py                  ← 复现校验
  plot.py                          ← 评估出图
  README.md                        ← 总览文档

core/
  data/                            ← 新建：通用数据工具
    __init__.py
    adjusted_panels.py             ← 后复权加载（通用基础设施）
  producers/                       ← 删除（alpha158 移走后变空）
```

## 4. 依赖方向

```
core/data/adjusted_panels.py       ← 最底层通用基础设施（不依赖任何人）
        ↑
alpha158/engine/                   ← 因子定义，依赖后复权加载
        ↑
alpha158/*.py                      ← 操作脚本，依赖 engine + core/data
        ↑
core/operators/industry_co_momentum.py  ← 只依赖 core/data，不依赖 alpha158
```

## 5. 移动清单

| 步骤 | 操作 | 说明 |
|---|---|---|
| 1 | 新建 `core/data/__init__.py` | 内容：`"""通用数据工具：后复权加载等。"""` |
| 2 | `git mv core/producers/alpha158/adjusted_panels.py core/data/adjusted_panels.py` | 后复权工具移到通用层 |
| 3 | `mkdir alpha158/engine/` | 新建引擎子目录 |
| 4 | `git mv core/producers/alpha158/factors.py alpha158/engine/factors.py` | 因子定义移到自包含包 |
| 5 | `git mv core/producers/alpha158/panel_operators.py alpha158/engine/panel_operators.py` | 宽表算子移到自包含包 |
| 6 | `git mv core/producers/alpha158/groups.py alpha158/engine/groups.py` | 因子分组移到自包含包 |
| 7 | 新建 `alpha158/engine/__init__.py` | 导出 Alpha158Panel, PanelOperators |
| 8 | 删除 `core/producers/alpha158/__init__.py` | 已移空 |
| 9 | 删除 `core/producers/__init__.py` | producers 目录废弃 |
| 10 | 删除 `core/producers/` 空目录 | 清理 |

## 6. 导入路径修改清单

### 6.1 alpha158 脚本（4 个文件，6 处修改）

| 文件 | 行号 | 当前导入 | 改为 |
|---|---|---|---|
| `alpha158/build.py` | 29 | `from core.producers.alpha158 import Alpha158Panel, adjusted_panels` | `from alpha158.engine import Alpha158Panel`<br>`from core.data import adjusted_panels` |
| `alpha158/build.py` | 30 | `from core.producers.alpha158.groups import factor_group` | `from alpha158.engine.groups import factor_group` |
| `alpha158/daily_update.py` | 37 | `from core.producers.alpha158 import Alpha158Panel` | `from alpha158.engine import Alpha158Panel` |
| `alpha158/daily_update.py` | 38 | `from core.producers.alpha158.groups import factor_group` | `from alpha158.engine.groups import factor_group` |
| `alpha158/verify_repro.py` | 31 | `from core.producers.alpha158 import Alpha158Panel` | `from alpha158.engine import Alpha158Panel` |
| `alpha158/verify_repro.py` | 32 | `from core.producers.alpha158 import adjusted_panels` | `from core.data import adjusted_panels` |
| `alpha158/smoke_replay.py` | 35 | `from core.producers.alpha158 import Alpha158Panel` | `from alpha158.engine import Alpha158Panel` |

### 6.2 core 模块（1 个文件，1 处修改）

| 文件 | 行号 | 当前导入 | 改为 |
|---|---|---|---|
| `core/operators/industry_co_momentum.py` | 26 | `from core.producers.alpha158.adjusted_panels import load_adjusted_panels` | `from core.data.adjusted_panels import load_adjusted_panels` |

### 6.3 scripts（2 个文件，2 处修改）

| 文件 | 行号 | 当前导入 | 改为 |
|---|---|---|---|
| `scripts/comomentum_cmc.py` | 57 | `from core.producers.alpha158.adjusted_panels import load_adjusted_panels` | `from core.data.adjusted_panels import load_adjusted_panels` |
| `scripts/comomentum_family_verify.py` | 55 | `from core.producers.alpha158.adjusted_panels import load_adjusted_panels` | `from core.data.adjusted_panels import load_adjusted_panels` |

### 6.4 注释修改（3 处）

| 文件 | 行号 | 当前注释 | 改为 |
|---|---|---|---|
| `alpha158/build.py` | 7 | `（见 core.producers.alpha158.groups）` | `（见 alpha158.engine.groups）` |
| `alpha158/verify_repro.py` | 8 | `core.producers.alpha158.loader 加载后复权宽表面板` | `core.data.adjusted_panels 加载后复权宽表面板` |
| `config/factors_input.py` | 39 | `复权读时实时算（core.producers.alpha158.loader）` | `复权读时实时算（core.data.adjusted_panels）` |

### 6.5 `adjusted_panels` 调用方式注意（2 处）

`build.py` 和 `verify_repro.py` 当前用**模块名.函数名**方式调用 `adjusted_panels`：

```python
# build.py 第 56 行 / verify_repro.py 第 59 行
panels = adjusted_panels.load_adjusted_panels(...)
```

因此 import 改法有两种选择，**必须跟调用方式对齐**：

| 方案 | import 写法 | 调用写法 | 改动量 |
|---|---|---|---|
| A（保持模块名调用） | `from core.data import adjusted_panels` | `adjusted_panels.load_adjusted_panels(...)` 不变 | 只改 import 行 |
| B（直接导入函数） | `from core.data.adjusted_panels import load_adjusted_panels` | `load_adjusted_panels(...)` 去掉模块名前缀 | 改 import + 改调用行 |

推荐 **方案 A**（改动量最小，只改 import 行，调用行不动）。

> ⚠️ 如果用方案 B，6.1 表里 `build.py` 和 `verify_repro.py` 的 import 写法需对应调整，且调用处也要去掉 `adjusted_panels.` 前缀。

### 6.6 不改的文件

| 文件 | 原因 |
|---|---|
| `alpha158/daily_update.py` 的 `adjusted_panels` 相关 | 不导入 adjusted_panels，只导入 Alpha158Panel + groups |
| `alpha158/plot.py` | 不导入 core.producers.alpha158 |
| `alpha158/README.md` | 后续单独更新（步骤 8） |

## 7. `alpha158/engine/__init__.py` 内容

```python
"""Alpha158 因子引擎（158 个量价技术因子）。

- panel_operators.PanelOperators : 宽表算子底座（Ref/Mean/Std/Slope/Rsquare/...）
- factors.Alpha158Panel          : 158 因子定义（9 K线 + 4 价格 + 23×5 rolling + 6×5 量）
- groups.factor_group            : 因子名 → 组映射（kline/price/rolling/volume）

后复权数据加载见 core.data.adjusted_panels（通用基础设施，非 alpha158 专属）。
入口脚本见 alpha158/ 目录上层（build/daily_update/smoke_replay/verify_repro/plot）。
"""
from .factors import Alpha158Panel
from .panel_operators import PanelOperators

__all__ = ["Alpha158Panel", "PanelOperators"]
```

## 8. `alpha158/README.md` 更新

第 5 行和第 19-21 行需要更新：

**当前**：
```
> **算子引擎**因被 core 算子共享，留在 `core/producers/alpha158/`（见下）。
...
引擎（**不在这**，勿找）：`core/producers/alpha158/`
- `factors.py` 158 因子定义 · `groups.py` 分组(kline/price/rolling/volume) ·
  `adjusted_panels.py` 读时后复权（**被 `core/operators/industry_co_momentum.py` 共享**，故留在 core）· `panel_operators.py` 宽表算子。
```

**改为**：
```
> **算子引擎**在 `alpha158/engine/`（本目录子目录，自包含）。
...
引擎：`alpha158/engine/`
- `factors.py` 158 因子定义 · `groups.py` 分组(kline/price/rolling/volume) · `panel_operators.py` 宽表算子。
- 后复权加载：`core/data/adjusted_panels.py`（通用基础设施，非 alpha158 专属，被 `core/operators/industry_co_momentum.py` 共享）。
```

## 9. 验证步骤

| 步骤 | 命令 | 预期结果 |
|---|---|---|
| 1 | `python -c "from core.data.adjusted_panels import load_adjusted_panels; print('OK')"` | OK |
| 2 | `python -c "from alpha158.engine import Alpha158Panel; print('OK')"` | OK |
| 3 | `python -c "from alpha158.engine.groups import factor_group; print('OK')"` | OK |
| 4 | `python -c "from alpha158.build import *; print('OK')"` | 无报错 |
| 5 | `python -c "from core.operators.industry_co_momentum import *; print('OK')"` | 无报错 |
| 6 | `PYTHONPATH=. python alpha158/smoke_replay.py --all` | 增量正确性对账通过 |
| 7 | `PYTHONPATH=. python alpha158/verify_repro.py` | 复现校验通过 |

## 10. 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 漏改某个 import | 低 | 导入报错 | 步骤 6 验证全部导入 |
| `adjusted_panels.py` 内部有对 `core.producers` 的反向依赖 | 极低 | 移动后报错 | 已读全文，只依赖 `config`，无反向依赖 |
| `factors.py` / `panel_operators.py` 内部有对 `adjusted_panels` 的依赖 | 低 | 移动后报错 | 已确认 factors 依赖 panel_operators，不依赖 adjusted_panels |
| `__pycache__` 残留导致旧导入生效 | 低 | 测试通过但实际失败 | 移动后清理 `__pycache__` |

## 11. 执行顺序

按依赖关系从底层往上改，确保每一步都可验证：

1. 新建 `core/data/` + 移 `adjusted_panels.py` → 验证 `from core.data.adjusted_panels import ...`
2. 新建 `alpha158/engine/` + 移 3 个引擎文件 → 验证 `from alpha158.engine import ...`
3. 改 `alpha158/*.py` 4 个脚本的 import → 验证 `from alpha158.build import *`
4. 改 `core/operators/` + `scripts/` 3 个文件的 import → 验证 industry_co_momentum
5. 删除 `core/producers/` → 验证无残留引用
6. 更新 `alpha158/README.md` + 2 处注释
7. 跑 `smoke_replay.py --all` + `verify_repro.py` 全量验证
