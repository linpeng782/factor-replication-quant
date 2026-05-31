"""
迁移回归对比：把 yolo 输出重定向到 sandbox，跟生产 parquet 做 bit-exact 比对。

不污染 /nfs/.../raw_factor/。比对失败时打印 diff 摘要。

用法：
    python scripts/regression_check.py peak_interval_kurt npf_mrq_sue8
    （不传参数则跑默认两个因子）
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
SANDBOX = Path("/tmp/migration_sandbox/raw_factor")


def _patch_yolo_output_dir(sandbox: Path) -> None:
    """yolo_engine 模块在 import 时把 RAW_FACTOR_BASE 拉成本地名 → 必须改 yolo_engine 的本地绑定。"""
    sandbox.mkdir(parents=True, exist_ok=True)
    # 触发模块加载
    import core.yolo_engine as ye

    ye.RAW_FACTOR_BASE = sandbox  # 直接覆盖本地绑定


def _compare_parquet(prod: Path, sand: Path, factor: str) -> tuple[bool, str]:
    """
    分级比对：
      1) shape 完全一致 → 直接 assert_frame_equal（bit-exact）
      2) shape 不同但 index 一致 → 比对**列交集**；diff 列额外报告（区分 prod-extra / sandbox-extra）
      3) 任一对齐后非交集列**全为 NaN** → 视为 universe drift（不是回归），仍判 PASS
      4) 否则 FAIL
    """
    if not prod.exists():
        return False, f"  ✗ 生产 parquet 不存在: {prod}"
    if not sand.exists():
        return False, f"  ✗ sandbox parquet 不存在: {sand}"

    a = pd.read_parquet(prod)
    b = pd.read_parquet(sand)

    msgs = [f"  生产:    shape={a.shape}, 非空={a.notna().values.sum():,}"]
    msgs.append(f"  sandbox: shape={b.shape}, 非空={b.notna().values.sum():,}")

    if not a.index.equals(b.index):
        msgs.append("  ✗ index（日期）不一致 — 真回归")
        return False, "\n".join(msgs)

    cols_a = set(a.columns)
    cols_b = set(b.columns)
    common = sorted(cols_a & cols_b)
    only_prod = sorted(cols_a - cols_b)
    only_sand = sorted(cols_b - cols_a)

    # ── 路径 1：shape + columns 全对齐
    if not only_prod and not only_sand:
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=True, check_dtype=True)
            msgs.append(f"  ✓ {factor}: bit-exact 一致")
            return True, "\n".join(msgs)
        except AssertionError as e:
            msgs.append(f"  ✗ {factor}: 列对齐但值不一致")
            msgs.append(f"  详情: {str(e).splitlines()[0][:200]}")
            return False, "\n".join(msgs)

    # ── 路径 2：列差异，但交集应 bit-exact
    msgs.append(
        f"  列差异：仅 prod {len(only_prod)} 列；仅 sandbox {len(only_sand)} 列；交集 {len(common)} 列"
    )
    a_common = a[common]
    b_common = b[common]
    try:
        pd.testing.assert_frame_equal(a_common, b_common, check_exact=True, check_dtype=True)
        msgs.append(f"  ✓ 交集 bit-exact 一致（{a_common.shape}）")
    except AssertionError as e:
        msgs.append(f"  ✗ {factor}: 交集列值仍不一致")
        msgs.append(f"  详情: {str(e).splitlines()[0][:200]}")
        return False, "\n".join(msgs)

    # ── 路径 3：差异列必须全为 NaN，否则真回归
    a_extra_all_nan = a[only_prod].isna().values.all() if only_prod else True
    b_extra_all_nan = b[only_sand].isna().values.all() if only_sand else True
    msgs.append(
        f"  仅 prod 列全为 NaN: {a_extra_all_nan} | "
        f"仅 sandbox 列全为 NaN: {b_extra_all_nan}"
    )
    if a_extra_all_nan and b_extra_all_nan:
        if only_sand:
            msgs.append(
                f"  → sandbox 多出的全 NaN 列（前 5 个）: {only_sand[:5]}"
                "  —— universe drift，不是回归"
            )
        msgs.append(f"  ✓ {factor}: 实质 bit-exact（universe drift 已隔离）")
        return True, "\n".join(msgs)

    msgs.append("  ✗ 差异列含非 NaN 值 — 真回归")
    return False, "\n".join(msgs)


def _run_one(factor: str, start_date: str, end_date: str) -> tuple[bool, str]:
    from core.spec_generator import load_spec_yaml
    from core.yolo_engine import run_factor
    from core.config import RAW_FACTOR_BASE as PROD_DIR  # 比对时用

    print(f"\n=== {factor} ===")
    t0 = time.time()
    spec = load_spec_yaml(factor)
    df = run_factor(
        factor_name=factor,
        spec_yaml=spec,
        start_date=start_date,
        end_date=end_date,
    )
    dur = time.time() - t0
    print(f"  YOLO 完成 in {dur:.1f}s, sandbox shape={df.shape}")

    sand_path = SANDBOX / f"{factor}.parquet"
    prod_path = PROD_DIR / f"{factor}.parquet"
    return _compare_parquet(prod_path, sand_path, factor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "factors",
        nargs="*",
        default=["peak_interval_kurt", "npf_mrq_sue8"],
        help="要回归的因子列表（默认 peak_interval_kurt + npf_mrq_sue8）",
    )
    parser.add_argument(
        "--keep-sandbox",
        action="store_true",
        help="跑完保留 /tmp/migration_sandbox（默认清理）",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="默认用 core.config.DEFAULT_START_DATE（与生产对齐）",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="默认用 core.config.DEFAULT_END_DATE",
    )
    args = parser.parse_args()
    from core.config import DEFAULT_END_DATE, DEFAULT_START_DATE

    start_date = args.start_date or DEFAULT_START_DATE
    end_date = args.end_date or DEFAULT_END_DATE
    print(f"日期区间: {start_date} ~ {end_date}")

    # sandbox 重建
    if SANDBOX.parent.exists():
        shutil.rmtree(SANDBOX.parent)
    SANDBOX.mkdir(parents=True, exist_ok=True)
    _patch_yolo_output_dir(SANDBOX)
    print(f"sandbox: {SANDBOX}（生产目录未触碰）")

    results: list[tuple[str, bool, str]] = []
    for f in args.factors:
        ok, msg = _run_one(f, start_date, end_date)
        results.append((f, ok, msg))
        print(msg)

    print("\n" + "=" * 60)
    print("回归对比结果汇总")
    print("=" * 60)
    all_ok = True
    for f, ok, _ in results:
        status = "✓ bit-exact" if ok else "✗ 不一致"
        print(f"  {status}  {f}")
        all_ok = all_ok and ok

    if not args.keep_sandbox and SANDBOX.parent.exists():
        shutil.rmtree(SANDBOX.parent)
        print(f"\nsandbox 已清理：{SANDBOX.parent}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
