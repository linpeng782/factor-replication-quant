"""
反向回滚：读 migrate_factor_panel_to_namespaced.py 写的 manifest，倒着 mv 回去。

用法：
  python scripts/reverse_factor_panel_migration.py path/to/manifest.json            # dry-run
  python scripts/reverse_factor_panel_migration.py path/to/manifest.json --execute  # 真动
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--execute", action="store_true",
                        help="真执行；默认 dry-run")
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"manifest 不存在: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    completed = data.get("completed", [])
    print(f"[plan] manifest 记录 {len(completed)} 条已完成 mv，准备倒序回滚")

    if not completed:
        print("  无内容，退出。")
        return

    # 倒序回滚：先回滚晚的，避免 dst 子目录还有别的文件挡道
    completed_rev = list(reversed(completed))

    print("\n[sample] 头 3 条 + 尾 2 条：")
    for item in completed_rev[:3]:
        print(f"  {item['dst']}\n    → {item['src']}")
    if len(completed_rev) > 5:
        print(f"  ... ({len(completed_rev) - 5} 条略) ...")
    for item in completed_rev[-2:]:
        print(f"  {item['dst']}\n    → {item['src']}")

    # 校验
    issues = []
    for item in completed_rev:
        if not Path(item["dst"]).exists():
            issues.append(f"dst 已不存在（不能 mv 回）: {item['dst']}")
            break
        if Path(item["src"]).exists():
            issues.append(f"src 已存在（无法回滚到）: {item['src']}")
            break
    if issues:
        print("\n  ⚠️  问题：")
        for x in issues:
            print(f"   - {x}")
        sys.exit(1)
    print("\n  ✅ 校验通过")

    if not args.execute:
        print("\n这是 dry-run。--execute 才真动。")
        return

    print(f"\n[execute] 反向 mv {len(completed_rev)} 个文件...")
    for i, item in enumerate(completed_rev, 1):
        Path(item["dst"]).rename(item["src"])
        if i % 50 == 0 or i == len(completed_rev):
            print(f"  [{i}/{len(completed_rev)}] reverted")

    print("\n✅ 回滚完成")


if __name__ == "__main__":
    main()
