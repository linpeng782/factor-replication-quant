"""
output/<factor>/ → sources/<pub>/<group>/output/<factor>/

冲突策略（仅当目标文件已存在时）：keep-new
  - 如果两边都有同名文件，**保留新位置**（新代码刚生成的，风格最新）
  - 旧位置独有的文件搬过去（如老 report.md）
  - 旧位置目录最后清空 + rmdir
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.spec_resolver import resolve_output_dir  # noqa: E402

OLD_OUTPUT = ROOT / "output"


def main() -> None:
    if not OLD_OUTPUT.exists():
        print("output/ 不存在，无事可做")
        return

    factor_dirs = sorted(p for p in OLD_OUTPUT.iterdir() if p.is_dir())
    print(f"要迁移 {len(factor_dirs)} 个因子的 output 子目录\n")

    moved = 0
    skipped_files = 0
    failed: list[str] = []

    for old_factor_dir in factor_dirs:
        factor = old_factor_dir.name
        try:
            new_dir = resolve_output_dir(factor)
        except FileNotFoundError as e:
            failed.append(f"{factor}: {e}")
            print(f"  ✗ {factor}: 找不到对应 sources/ group（{e}）")
            continue

        new_dir.mkdir(parents=True, exist_ok=True)
        for f in old_factor_dir.iterdir():
            if not f.is_file():
                continue
            target = new_dir / f.name
            if target.exists():
                # 保留新位置；记录跳过
                f.unlink()
                skipped_files += 1
                continue
            shutil.move(str(f), str(target))

        # 清空后删除空目录
        try:
            old_factor_dir.rmdir()
            moved += 1
            print(f"  ✓ {factor}  →  {new_dir.relative_to(ROOT)}")
        except OSError as e:
            failed.append(f"{factor}: rmdir failed ({e})")
            print(f"  ⚠ {factor}: 目录非空无法删除（{e}）")

    # 顶层 output/ 如果空了，也删掉
    if not any(OLD_OUTPUT.iterdir()):
        OLD_OUTPUT.rmdir()
        print(f"\n顶层 output/ 已删除")
    else:
        print(f"\n顶层 output/ 还有残留：{[p.name for p in OLD_OUTPUT.iterdir()]}")

    print(f"\n汇总：迁移 {moved}，冲突保留新版 {skipped_files}，失败 {len(failed)}")
    if failed:
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
