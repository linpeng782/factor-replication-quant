"""
factor-panel 命名空间迁移：扁平 → <producer>/<factor>.parquet

分类规则：
  - REPLICATION_BASE 下任何 .parquet           → spec
  - ALPHA_ENGINE_BASE 下，文件名 MARS_ 前缀     → mars
  - ALPHA_ENGINE_BASE 下，其余                  → alpha158

迁移映射：
  factor-panel/X.parquet           → factor-panel/<producer>/X.parquet
  cleaned-factor-panel/X.parquet   → cleaned-factor-panel/<producer>/X.parquet
  raw_factor/X.parquet             → factor-panel/spec/X.parquet
  cleaned_factor/X.parquet         → cleaned-factor-panel/spec/X.parquet

用法：
  python scripts/migrate_factor_panel_to_namespaced.py            # dry-run
  python scripts/migrate_factor_panel_to_namespaced.py --execute  # 真动
  python scripts/reverse_factor_panel_migration.py <manifest>     # 回滚

回滚：每次 --execute 都把 (src, dst) 对落 manifest_<timestamp>.json，
reverse 脚本按照 manifest 倒着 mv 即可。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ── 路径常量 ────────────────────────────────────────────────────
ALPHA_ENGINE_BASE = Path("/nfs/ofs-prediction/peterzhenglinpeng/my-alpha-engine")
ALPHA_ENGINE_REPO = Path("/nfs/volume-1593-1/peterzhenglinpeng/my-alpha-engine")
REPLICATION_BASE = Path("/nfs/ofs-prediction/peterzhenglinpeng/factor-replication")

FACTOR_PANEL = ALPHA_ENGINE_BASE / "factor-panel"
CLEANED_PANEL = ALPHA_ENGINE_BASE / "cleaned-factor-panel"
REPL_RAW = REPLICATION_BASE / "raw_factor"
REPL_CLEAN = REPLICATION_BASE / "cleaned_factor"

# (src_dir, dst_root_panel)：dst 下还要加一层 <producer>/
_SOURCES = [
    (FACTOR_PANEL, FACTOR_PANEL),
    (CLEANED_PANEL, CLEANED_PANEL),
    (REPL_RAW, FACTOR_PANEL),
    (REPL_CLEAN, CLEANED_PANEL),
]


def classify(parquet: Path) -> str:
    """parquet 全路径 → producer ∈ {alpha158, mars, spec}"""
    if REPLICATION_BASE in parquet.parents:
        return "spec"
    return "mars" if parquet.name.startswith("MARS_") else "alpha158"


def build_plan() -> list[dict]:
    plan: list[dict] = []
    for src_dir, dst_root in _SOURCES:
        if not src_dir.exists():
            print(f"  ⚠️  源目录不存在，跳过: {src_dir}", file=sys.stderr)
            continue
        # 只扫"顶层"的 parquet；如果已经有 <producer>/ 子目录里的文件就不再迁
        for parquet in sorted(src_dir.glob("*.parquet")):
            producer = classify(parquet)
            plan.append({
                "src": str(parquet),
                "dst": str(dst_root / producer / parquet.name),
                "producer": producer,
                "panel_kind": "raw" if "cleaned" not in dst_root.name else "cleaned",
                "factor": parquet.stem,
            })
    return plan


def load_registry() -> dict[str, set[str]] | None:
    """从 my-alpha-engine 加载预期因子集；import 失败则返回 None（不阻断）"""
    if not ALPHA_ENGINE_REPO.exists():
        return None
    sys.path.insert(0, str(ALPHA_ENGINE_REPO))
    try:
        from core.factors import list_factor_names  # type: ignore
        return {
            "alpha158": set(list_factor_names("alpha158", windows=[5, 10, 20, 30, 60])),
            "mars": set(list_factor_names("mars", windows=[5, 10, 20, 30, 60])),
        }
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️  注册表 import 失败（继续走前缀启发式）: {e}", file=sys.stderr)
        return None
    finally:
        if str(ALPHA_ENGINE_REPO) in sys.path:
            sys.path.remove(str(ALPHA_ENGINE_REPO))


def validate(plan: list[dict], registry: dict[str, set[str]] | None) -> list[str]:
    issues: list[str] = []

    # 1. 目标无重复
    dsts = [p["dst"] for p in plan]
    dups = [d for d, n in Counter(dsts).items() if n > 1]
    if dups:
        issues.append(f"目标路径重复 ({len(dups)} 个)：{dups[:3]}")

    # 2. 目标不应已存在（除非允许覆盖；这里强制不允许）
    existing = [p["dst"] for p in plan if Path(p["dst"]).exists()]
    if existing:
        issues.append(f"目标已存在 ({len(existing)} 个)，可能上次部分执行：{existing[:3]}")

    # 3. 注册表交叉验证（只查 alpha-engine 侧 raw panel，cleaned 应跟 raw 同集合）
    if registry is not None:
        raw_alpha = {p["factor"] for p in plan
                     if p["panel_kind"] == "raw" and p["producer"] == "alpha158"
                     and ALPHA_ENGINE_BASE in Path(p["src"]).parents}
        raw_mars = {p["factor"] for p in plan
                    if p["panel_kind"] == "raw" and p["producer"] == "mars"}
        for producer, expected, actual in [
            ("alpha158", registry["alpha158"], raw_alpha),
            ("mars", registry["mars"], raw_mars),
        ]:
            missing = expected - actual
            extra = actual - expected
            if missing:
                issues.append(f"{producer}: 注册表说有但磁盘缺 {len(missing)} 个: {sorted(missing)[:3]}")
            if extra:
                issues.append(f"{producer}: 磁盘多出注册表 {len(extra)} 个: {sorted(extra)[:3]}")

    # 4. raw 和 cleaned 因子集应该一致（per producer）
    for producer in {"alpha158", "mars", "spec"}:
        raw = {p["factor"] for p in plan if p["panel_kind"] == "raw" and p["producer"] == producer}
        cleaned = {p["factor"] for p in plan if p["panel_kind"] == "cleaned" and p["producer"] == producer}
        if raw != cleaned:
            only_raw = sorted(raw - cleaned)[:3]
            only_cleaned = sorted(cleaned - raw)[:3]
            issues.append(
                f"{producer}: raw 与 cleaned 因子集不一致；"
                f"only_raw={only_raw} only_cleaned={only_cleaned}"
            )

    return issues


def print_summary(plan: list[dict]) -> None:
    print(f"\n[summary] 共 {len(plan)} 个文件")
    grid: dict[tuple[str, str], int] = defaultdict(int)
    for item in plan:
        grid[(item["producer"], item["panel_kind"])] += 1
    print(f"  {'producer':10s} {'kind':8s} count")
    for (producer, kind), count in sorted(grid.items()):
        print(f"  {producer:10s} {kind:8s} {count}")


def print_sample(plan: list[dict], n: int = 5) -> None:
    print(f"\n[sample] 头 {n} 条 + 尾 3 条：")
    for item in plan[:n]:
        print(f"  {item['src']}\n    → {item['dst']}")
    if len(plan) > n + 3:
        print(f"  ... ({len(plan) - n - 3} 条略) ...")
        for item in plan[-3:]:
            print(f"  {item['src']}\n    → {item['dst']}")


def execute(plan: list[dict], manifest_path: Path) -> None:
    """每步 mv 后立刻 flush manifest，便于异常时反向回滚"""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # 先建所有 dst 父目录
    parents = {Path(p["dst"]).parent for p in plan}
    for d in parents:
        d.mkdir(parents=True, exist_ok=True)

    completed: list[dict] = []
    failure: str | None = None
    try:
        for i, item in enumerate(plan, 1):
            src, dst = Path(item["src"]), Path(item["dst"])
            if dst.exists():
                failure = f"目标已存在（疑似 race）: {dst}"
                break
            if not src.exists():
                failure = f"源不存在（疑似已被移走）: {src}"
                break
            src.rename(dst)
            completed.append(item)
            if i % 50 == 0 or i == len(plan):
                print(f"  [{i}/{len(plan)}] mv done")
    finally:
        manifest_path.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "total_planned": len(plan),
            "completed_count": len(completed),
            "failure": failure,
            "completed": completed,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nmanifest: {manifest_path}")
        if failure:
            print(f"⚠️  失败：{failure}")
            print(f"已迁移 {len(completed)}/{len(plan)}；可用 reverse 脚本回滚。")
            sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true",
                        help="真执行；默认 dry-run")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="manifest 输出路径；默认 scripts/migration_manifest_<ts>.json")
    args = parser.parse_args()

    print("[1/3] 构建迁移计划...")
    plan = build_plan()
    if not plan:
        print("  没有发现要迁移的文件，退出。")
        return
    print_summary(plan)

    print("\n[2/3] 校验...")
    registry = load_registry()
    issues = validate(plan, registry)
    if issues:
        print("  ⚠️  发现问题：")
        for x in issues:
            print(f"   - {x}")
        if args.execute:
            print("\n--execute 中止：先修问题再跑。")
            sys.exit(1)
    else:
        print("  ✅ 校验通过")

    print_sample(plan)

    if not args.execute:
        print("\n这是 dry-run。--execute 才真动。")
        return

    print(f"\n[3/3] 执行迁移 ({len(plan)} 个文件)...")
    manifest = args.manifest or (
        Path(__file__).parent
        / f"migration_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    execute(plan, manifest)
    print("\n✅ 迁移完成")


if __name__ == "__main__":
    main()
