"""
因子标识符 → 路径解析（sources/<publisher>/<group>/{specs,docs,inputs,input.md} 布局）。

使用约定：
- 裸因子名 'peak_minute_count' → 扫 sources/*/*/specs/<factor>/spec.yaml，唯一才返回
- 限定路径 'kysec/paper_27_microstructure/peak_minute_count' → 直接拼路径
- input 解析：先看 group_dir/inputs/<factor>.md（fundamental 风格），再看 group_dir/input.md（券商 paper 风格）
"""

from __future__ import annotations

import functools
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SOURCES_DIR = PROJECT_ROOT / "sources"


def _is_qualified(arg: str) -> bool:
    return "/" in arg


def _qualified_to_spec_path(arg: str) -> Path:
    """'kysec/paper_27_microstructure/peak_minute_count' → sources/.../specs/peak_minute_count/spec.yaml"""
    parts = arg.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"限定路径需 3 段 '<publisher>/<group>/<factor>'，收到 {arg!r}"
        )
    publisher, group, factor = parts[0], parts[1], "/".join(parts[2:])
    return SOURCES_DIR / publisher / group / "specs" / factor / "spec.yaml"


@functools.cache
def _scan_registry() -> dict[str, Path]:
    """扫 sources/*/*/specs/*/spec.yaml；返回 {factor_name: spec_path}。重名保留全部为 list 方便诊断。"""
    registry: dict[str, list[Path]] = {}
    if not SOURCES_DIR.exists():
        return {}
    for spec_path in SOURCES_DIR.glob("*/*/specs/*/spec.yaml"):
        factor = spec_path.parent.name
        registry.setdefault(factor, []).append(spec_path)
    out: dict[str, Path] = {}
    for factor, paths in registry.items():
        if len(paths) == 1:
            out[factor] = paths[0]
        else:
            # 重名时存第一个，但 resolve_spec_path 会重新检测并报错
            out[factor] = paths[0]
    return out


def _all_matches_for_bare_name(name: str) -> list[Path]:
    if not SOURCES_DIR.exists():
        return []
    return sorted(SOURCES_DIR.glob(f"*/*/specs/{name}/spec.yaml"))


def resolve_spec_path(arg: str) -> Path:
    """
    把因子标识符解析为 spec.yaml 绝对路径。

    arg:
      - 'peak_minute_count' (裸名)：注册表查找，唯一才返回
      - 'kysec/paper_27_microstructure/peak_minute_count' (限定)：直接拼
    """
    if _is_qualified(arg):
        path = _qualified_to_spec_path(arg)
        if not path.exists():
            raise FileNotFoundError(f"spec 不存在: {path}")
        return path

    matches = _all_matches_for_bare_name(arg)
    if len(matches) == 0:
        registry = _scan_registry()
        sample = ", ".join(sorted(registry.keys())[:10])
        raise FileNotFoundError(
            f"factor {arg!r} 未找到。已扫描的因子（前 10 个）：[{sample}]\n"
            f"提示：用限定路径 '<publisher>/<group>/<factor>'，或确认 sources/ 下已有该因子。"
        )
    if len(matches) > 1:
        qualified = [
            "/".join(p.relative_to(SOURCES_DIR).parts[:2] + (arg,)) for p in matches
        ]
        raise ValueError(
            f"factor {arg!r} 在多处定义：{qualified}\n"
            f"请用限定路径 '<publisher>/<group>/<factor>' 消歧。"
        )
    return matches[0]


def resolve_input_path(spec_path: Path) -> Path:
    """
    给 spec.yaml 路径，找研报输入。

    路径结构：sources/<pub>/<group>/specs/<factor>/spec.yaml
                                       ^ parents[2] = group_dir

    优先级：
      1. group_dir/inputs/<factor>.md   (fundamental 系列：每因子一份)
      2. group_dir/input.md             (券商 paper：paper-level 共享)
      3. raise
    """
    factor = spec_path.parent.name
    group_dir = spec_path.parents[2]
    per_factor = group_dir / "inputs" / f"{factor}.md"
    if per_factor.exists():
        return per_factor
    paper_level = group_dir / "input.md"
    if paper_level.exists():
        return paper_level
    raise FileNotFoundError(
        f"未找到研报输入：{per_factor} 或 {paper_level} 都不存在。"
    )


def resolve_group_dir_for_new_spec(arg: str) -> Path:
    """
    spec_generator 写新 spec 时调用：解析出 group_dir（spec.yaml 还不存在时）。

    arg 必须是限定路径 'publisher/group/factor'，否则 raise（提示用户加 --group）。
    """
    if not _is_qualified(arg):
        existing = sorted(p.parent.name for p in SOURCES_DIR.glob("*/*"))
        raise ValueError(
            f"新 spec 必须用限定路径 '<publisher>/<group>/<factor>'，收到裸名 {arg!r}。\n"
            f"已存在的 group：{existing[:20]}"
        )
    parts = arg.split("/")
    publisher, group = parts[0], parts[1]
    return SOURCES_DIR / publisher / group


def factor_name_from_arg(arg: str) -> str:
    """'kysec/paper_27/peak_minute_count' → 'peak_minute_count'；'peak_minute_count' → 'peak_minute_count'。"""
    return arg.rsplit("/", 1)[-1]


def resolve_source(arg: str) -> str:
    """因子标识符 → 来源（sources/<source>/... 顶层 publisher，如 cxl / kysec / founder）。

    仅取 namespace 的顶层（source）；实际落盘是 source/group 两级，见 resolve_namespace：
    factors/<stage>/<source>/<group>/<factor>.parquet。
    来源从 spec 路径推导（不在 spec 内容里）——改来源 = 挪 sources/ 目录，不动 spec。
    """
    spec_path = resolve_spec_path(arg)
    return spec_path.relative_to(SOURCES_DIR).parts[0]


def resolve_source_safe(factor_name: str, default: str = "_misc") -> str:
    """惰性解析来源；解析不到（如临时/probe 因子无 spec）时回退 default，不抛错。"""
    try:
        return resolve_source(factor_name)
    except (FileNotFoundError, ValueError):
        return default


def resolve_namespace(arg: str) -> str:
    """因子标识符 → '<publisher>/<group>'（数据/产出分桶，完整镜像 sources/ 两级）。

    group = 研报/系列（拥有 ~10-20 个因子的自然单元）。落盘：
      factors/<stage>/<publisher>/<group>/<factor>.parquet
      output/<publisher>/<group>/<factor>/

    dquant 后端时，以下 source 产物统一映射到 <source>-dquant/（raw/cleaned/neu/output
    四处自动并行隔离，不覆盖 rq 基线）：
      · cxl   —— 基本面因子，随 FUNDAMENTAL_BACKEND=dquant
      · kysec —— 分钟因子，随 _IS_MINUTE_DQUANT（MINUTE_DATA_BACKEND=dquant）
    ML 训练 SOURCES 配置（如 "kysec-dquant/paper_27_microstructure"）与此映射对齐。
    """
    spec_path = resolve_spec_path(arg)
    ns = "/".join(spec_path.relative_to(SOURCES_DIR).parts[:2])
    import config
    if config.FUNDAMENTAL_BACKEND == "dquant" and ns.startswith("cxl/"):
        ns = "cxl-dquant/" + ns[len("cxl/"):]
    if config._IS_MINUTE_DQUANT and ns.startswith("kysec/"):
        ns = "kysec-dquant/" + ns[len("kysec/"):]
    return ns


def resolve_namespace_safe(factor_name: str, default: str = "_misc/_misc") -> str:
    """惰性解析 namespace；解析不到时回退 default，不抛错。"""
    try:
        return resolve_namespace(factor_name)
    except (FileNotFoundError, ValueError):
        return default


_TEMPORAL_TRANSFORM_METHODS = {"diff", "shift", "yoy", "qoq"}
# 带 window 参数的滚动类算子（warmup 由 window 决定）
_ROLLING_ACTIONS = {
    "rolling", "rolling_weighted_mean", "rolling_sorted_subset",
    "rolling_group_ratio", "rolling_ts_regress",
}
# 自建列、不消费 ctx 已有列的 action → 产出列 warmup 记 0（跨日回看由各自内部负责）：
#   fetch / load_panel        —— 直接读源数据/外部面板
#   minute_* / industry_*     —— 分钟聚合的跨日 warmup 由 minute_engine 的 reducer.warmup
#                                + per-stock superset 缓存自管，不属于 L3 尾窗职责
_BASE_ACTIONS = {"fetch", "load_panel", "industry_co_momentum"}
# _collect_source_columns 认不出（键名不以 source_column 开头）、但确实引用 ctx 列的键。
# 漏了会低估 warmup —— reg_pb_gshe 的 group_column=ep_group 就携带 252 日窗口，
# 曾在本函数重构时把 W2 从 252 打到 1（已由下方 floor 护栏兜住）。
_EXTRA_SOURCE_KEYS = ("weight_column", "mask_column", "group_column", "change_on")


def _step_output_columns(step: dict) -> list[str]:
    """该 step 产出的列名（output_column / output_columns / minute features / merge columns）。"""
    if step.get("output_columns"):
        return list(step["output_columns"].values())
    if step.get("action") == "merge":
        return list(step.get("columns") or [])
    if step.get("features"):                       # minute_* 聚合：features 即产出列
        return list(step["features"])
    out = step.get("output_column")
    return [out] if out else []


def _step_input_columns(step: dict) -> list[str]:
    """该 step 消费的列名（source_column[s] / source_column_<role> / weight / mask）。"""
    from core.spec_schema import _collect_source_columns

    try:
        cols = list(_collect_source_columns(step, loc="warmup"))
    except Exception:                              # 无 source_* 声明（自建列类 action）
        cols = []
    cols += [step[k] for k in _EXTRA_SOURCE_KEYS if step.get(k)]
    return cols


def _step_own_lookback(step: dict) -> int:
    """该 step 自身引入的回看长度（交易日）；非时序算子为 0。"""
    act = step.get("action")
    if act in _ROLLING_ACTIONS and step.get("window") is not None:
        return int(step["window"])
    if act == "transform" and step.get("method") in _TEMPORAL_TRANSFORM_METHODS:
        default = 4 if step.get("method") == "yoy" else 1
        return int(step.get("periods", default))
    return 0


def max_warmup_window(spec_yaml: dict) -> int:
    """W = 算出 factor.column 所需的回看长度（交易日），**按依赖链求和**；无时序步骤默认 1。

    L3 增量回读窗口由此推导：fetch_start = last − (W + buffer) 交易日。

    做法：沿 calculation_steps 顺序传播「每列所需回看」符号表 w[col]：
      · 自建列 action（fetch / load_panel / minute_* / industry_co_momentum）→ w[out] = 0
      · 时序算子   → w[out] = max(w[输入列]) + 自身窗口（rolling.window / transform.periods）
      · 非时序算子 → w[out] = max(w[输入列])            （compute / rank / row_* / merge 等）
    最终取 w[factor.column]。并行链天然取 max，串行链天然求和。

    ⚠️ 2026-08 修复：旧实现取「spec 内单个最大 window」，对**同一链上叠加多个时序算子**的 spec
    会低估。首个触发者 dongwu/pct_turn20：链为 rolling(40)→shift(1)→rolling(20)，真实 warmup
    61 日而旧实现返回 40 → 增量补数时前 5 个新交易日整天全空，且不报错不告警
    （见 sources/dongwu/paper_07_stable_turnover/docs/pct_turn20.md）。
    over-warmup 是安全的（只是回读窗口变长、增量稍慢），under-warmup 会静默产出空洞。
    """
    steps = spec_yaml.get("calculation_steps") or []
    w: dict[str, int] = {}

    for step in steps:
        act = step.get("action")
        if act == "filter":                        # 只删行、不增列
            continue
        outs = _step_output_columns(step)
        if act in _BASE_ACTIONS or (act or "").startswith("minute_"):
            for c in outs:
                w[c] = 0
            continue
        if act == "merge":                         # 跨表搬列，warmup 随列继承（已在符号表里）
            continue
        base = max((w.get(c, 0) for c in _step_input_columns(step)), default=0)
        for c in outs:
            w[c] = base + _step_own_lookback(step)

    # floor 护栏：结果**永不低于**「spec 内单个最大窗口」（= 旧实现语义）。
    # 依赖链靠 source_column 等键推导，若将来新增算子用了本函数不认识的列引用键，
    # 链会断在那里、warmup 被低估 → 静默产出空洞（最难查的一类 bug）。
    # 有此 floor，最坏情况退化成旧行为，绝不会比旧实现更差。
    floor = max([1] + [_step_own_lookback(s) for s in steps])

    factor_column = (spec_yaml.get("factor") or {}).get("column")
    if factor_column in w:
        return max(floor, w[factor_column])
    # 目标列没追踪到（异常 spec）→ 全部时序窗口求和，宁可 over-warmup
    return max(floor, sum(_step_own_lookback(s) for s in steps))


def incremental_safe(spec_yaml: dict) -> bool:
    """该 spec 是否可用【有界尾窗】增量。

    返回 False（→ 应全量重算）当任一时序算子的**日历回看无界**：
      (a) `rolling` 带 `change_on`：变化日 rolling——按 change_on 值变化点采样后再 rolling(window)，
          window 是【变化点个数】而非日历日（季频基本面 8 期≈8 季≈504 日，且变化间隔随股而异）。
      (b) `filter` 出现在 `rolling` / 时序 `transform` 之前：filter 删行（op_filter 走
          df.query().reset_index），其后 window/periods 按【过滤后行数】计 → 日历回看不再有界。
    这类因子改全量重算：从冻结 PIT 源重算是确定性的、历史不漂移，成本可接受（截面计算便宜）。
    """
    seen_filter = False
    for step in (spec_yaml.get("calculation_steps") or []):
        act = step.get("action")
        if act == "filter":
            seen_filter = True
        elif act == "rolling":
            if step.get("change_on"):          # 变化日 rolling：日历回看无界
                return False
            if seen_filter:                    # filter→rolling：window 按过滤后行数计
                return False
        elif act == "transform" and step.get("method") in _TEMPORAL_TRANSFORM_METHODS:
            if seen_filter:
                return False
    return True


def resolve_output_dir(arg: str) -> Path:
    """
    评估产物目录：<project_root>/output/<factor>/

    平铺在项目根下，便于：
    - 用户 ls output/ 一眼浏览所有因子
    - tar output/ 一行打包所有评估产物
    - glob output/*/evaluation_*.png 批量读取

    按 <publisher>/<group> 分桶（镜像 factors/ 与 sources/），便于按研报浏览：
      output/kysec/paper_27_microstructure/peak_minute_count/
    """
    spec_path = resolve_spec_path(arg)
    namespace = "/".join(spec_path.relative_to(SOURCES_DIR).parts[:2])
    return PROJECT_ROOT / "output" / namespace / spec_path.parent.name
