"""
回归对比：把当前 baseline 跑出的结果与 git 中的"金标"逐数值比对。

Phase 5 用法：
    1. 重跑 regression_baseline_replication.py 和 regression_baseline_alpha_engine.py
       (它们会覆写 *.parquet)
    2. 但是！对比前要先把"金标"从 git 取出到一个临时目录：
         git show <baseline_commit>:scripts/regression_baselines/replication_ic_summary.parquet \
             > /tmp/golden/replication_ic_summary.parquet
       (或者保留 baseline 副本到 _golden/)
    3. 跑本脚本

为了简化使用，本脚本接受两个目录参数：旧 baseline 和新 baseline，逐文件对比。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def compare_parquet(old: Path, new: Path, label: str, atol: float = 0.0) -> bool:
    """逐数值对比；atol=0 → 严格相等"""
    if not old.exists():
        print(f"  ⚠️  [{label}] 金标缺失: {old}")
        return False
    if not new.exists():
        print(f"  ⚠️  [{label}] 新结果缺失: {new}")
        return False

    df_old = pd.read_parquet(old)
    df_new = pd.read_parquet(new)
    try:
        pd.testing.assert_frame_equal(
            df_old, df_new,
            check_exact=(atol == 0.0),
            atol=atol if atol > 0 else 0,
            rtol=0,
        )
        print(f"  ✅ [{label}] {df_old.shape} 完全相等")
        return True
    except AssertionError as e:
        print(f"  ❌ [{label}] DIFF:")
        print(f"     {str(e).splitlines()[0]}")
        # 找出哪些 cell 不等
        if df_old.shape == df_new.shape:
            common_cols = [c for c in df_old.columns if c in df_new.columns]
            for col in common_cols:
                if df_old[col].dtype.kind in "fi" and df_new[col].dtype.kind in "fi":
                    diff = (df_old[col] - df_new[col]).abs()
                    if (diff > atol).any():
                        worst = diff.idxmax()
                        print(f"     - {col}@{worst}: old={df_old.loc[worst, col]} new={df_new.loc[worst, col]}")
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--golden", type=Path, required=True,
                   help="金标 baseline 目录（提取共享库前的快照）")
    p.add_argument("--new", type=Path, required=True,
                   help="新 baseline 目录（提取后重跑的结果）")
    p.add_argument("--atol", type=float, default=0.0,
                   help="绝对容差，默认 0 严格相等")
    args = p.parse_args()

    files = [
        ("replication_ic_summary", "replication_ic_summary.parquet"),
        ("replication_layered_summary", "replication_layered_summary.parquet"),
        ("replication_cleaned_fingerprint", "replication_cleaned_fingerprint.parquet"),
        ("alpha_engine_ic_summary", "alpha_engine_ic_summary.parquet"),
        ("alpha_engine_layered_summary", "alpha_engine_layered_summary.parquet"),
    ]

    print(f"\n=== regression compare ===")
    print(f"  golden: {args.golden}")
    print(f"  new:    {args.new}")
    print(f"  atol:   {args.atol}\n")

    all_pass = True
    for label, fname in files:
        ok = compare_parquet(args.golden / fname, args.new / fname, label, args.atol)
        all_pass = all_pass and ok

    print()
    if all_pass:
        print("✅ 全部一致——抽 alpha-shared 安全")
        sys.exit(0)
    else:
        print("❌ 有差异——回查 alpha-shared 是否引入数值漂移")
        sys.exit(1)


if __name__ == "__main__":
    main()
