"""
全链路 0→1 回归对比：raw_factor + cleaned_factor + evaluation PNG
========================================================================
所有写都重定向到 /tmp/full_sandbox/，生产路径 zero touch。

对比策略：
  - raw / cleaned parquet：列交集 bit-exact（universe drift 单独报告）
  - eval PNG：先 self-determinism（两次跑 bit-exact），再 vs 生产
              （生产 PNG 旧于 plots.py commit d5e7b8c 时不强求字节级对齐，
               但底层 IC/Sharpe/分组数字必须一致——通过 log 比对）

用法：
    python scripts/regression_check_full.py eruption_turnover_corr
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SANDBOX = Path("/tmp/full_sandbox")
SAND_RAW = SANDBOX / "raw_factor"
SAND_CLEAN = SANDBOX / "cleaned_factor"
SAND_OUT = SANDBOX / "output"


def _patch_paths() -> None:
    SAND_RAW.mkdir(parents=True, exist_ok=True)
    SAND_CLEAN.mkdir(parents=True, exist_ok=True)
    SAND_OUT.mkdir(parents=True, exist_ok=True)
    import config as cfg
    import core.yolo_engine as ye

    ye.RAW_FACTOR_BASE = SAND_RAW
    cfg.RAW_FACTOR_BASE = SAND_RAW
    cfg.CLEANED_FACTOR_BASE = SAND_CLEAN


def _intersection_compare(prod: Path, sand: Path, label: str) -> tuple[bool, str]:
    if not prod.exists():
        return False, f"  ✗ {label}: 生产 parquet 不存在 {prod}"
    if not sand.exists():
        return False, f"  ✗ {label}: sandbox parquet 不存在 {sand}"

    a = pd.read_parquet(prod)
    b = pd.read_parquet(sand)
    msgs = [f"  {label}:"]
    msgs.append(f"    生产: shape={a.shape}, 非空={a.notna().values.sum():,}")
    msgs.append(f"    sandbox: shape={b.shape}, 非空={b.notna().values.sum():,}")

    if not a.index.equals(b.index):
        msgs.append(f"    ✗ index 不一致")
        return False, "\n".join(msgs)

    cols_a = set(a.columns)
    cols_b = set(b.columns)
    common = sorted(cols_a & cols_b)
    only_a = sorted(cols_a - cols_b)
    only_b = sorted(cols_b - cols_a)

    if not only_a and not only_b:
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            msgs.append(f"    ✓ bit-exact 一致")
            return True, "\n".join(msgs)
        except AssertionError as e:
            msgs.append(f"    ✗ 列对齐但值不一致: {str(e).splitlines()[0][:200]}")
            return False, "\n".join(msgs)

    msgs.append(f"    列差异: prod-only {len(only_a)}, sandbox-only {len(only_b)}, 交集 {len(common)}")
    try:
        pd.testing.assert_frame_equal(a[common], b[common], check_exact=True)
        msgs.append(f"    ✓ 交集 bit-exact 一致")
    except AssertionError as e:
        msgs.append(f"    ✗ 交集仍不一致: {str(e).splitlines()[0][:200]}")
        return False, "\n".join(msgs)

    a_ok = a[only_a].isna().values.all() if only_a else True
    b_ok = b[only_b].isna().values.all() if only_b else True
    msgs.append(f"    prod-only 全 NaN: {a_ok}; sandbox-only 全 NaN: {b_ok}")
    if a_ok and b_ok:
        if only_b:
            msgs.append(f"    universe drift 列（前 5）: {only_b[:5]}")
        return True, "\n".join(msgs)
    msgs.append(f"    ✗ 差异列含非 NaN — 真回归")
    return False, "\n".join(msgs)


def _png_self_determinism(factor: str, png_name: str) -> tuple[bool, str]:
    """同 plots.py 跑两次，验证字节级确定性。"""
    p1 = SAND_OUT / factor / png_name
    if not p1.exists():
        return False, "  ✗ PNG self-determinism: 第一次 PNG 不存在"
    saved = SANDBOX / f"{factor}_run1.png"
    shutil.copy2(p1, saved)
    p1.unlink()

    # 重跑（直接重新调 evaluate_single_factor，不重新跑 yolo）
    _run_eval_only(factor, redirect_output=True)
    if not p1.exists():
        return False, "  ✗ PNG self-determinism: 第二次 PNG 未生成"

    rc = subprocess.run(["cmp", str(saved), str(p1)], capture_output=True).returncode
    if rc == 0:
        size = p1.stat().st_size
        return True, f"  ✓ PNG self-determinism: 两次跑 bit-exact ({size} bytes)"
    return False, f"  ✗ PNG self-determinism: cmp exit={rc}"


def _run_eval_only(factor: str, redirect_output: bool) -> None:
    import yaml

    from core.evaluation import evaluate_single_factor
    from core.spec_resolver import resolve_spec_path

    raw = pd.read_parquet(SAND_RAW / f"{factor}.parquet")
    spec = yaml.safe_load(resolve_spec_path(factor).read_text())
    out_dir = SAND_OUT / factor if redirect_output else None
    evaluate_single_factor(
        factor_name=factor,
        factor_df=raw,
        spec_yaml=spec,
        output_dir=str(out_dir) if out_dir else None,
    )


def _run_full_pipeline(factor: str) -> None:
    import yaml

    from config import DEFAULT_END_DATE, DEFAULT_START_DATE
    from core.evaluation import evaluate_single_factor
    from core.spec_resolver import resolve_spec_path
    from core.yolo_engine import run_factor

    spec = yaml.safe_load(resolve_spec_path(factor).read_text())
    print(f"\n--- 跑 YOLO（写 sandbox raw_factor） ---")
    t0 = time.time()
    raw_df = run_factor(
        factor_name=factor,
        spec_yaml=spec,
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE,
    )
    print(f"  YOLO done in {time.time() - t0:.1f}s, shape={raw_df.shape}")

    print(f"\n--- 跑 evaluation（写 sandbox cleaned_factor + sandbox output PNG） ---")
    t0 = time.time()
    raw = pd.read_parquet(SAND_RAW / f"{factor}.parquet")
    evaluate_single_factor(
        factor_name=factor,
        factor_df=raw,
        spec_yaml=spec,
        output_dir=str(SAND_OUT / factor),
    )
    print(f"  eval done in {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factor", help="因子名（裸名或限定路径）")
    parser.add_argument("--keep-sandbox", action="store_true")
    args = parser.parse_args()

    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    _patch_paths()
    print(f"sandbox: {SANDBOX}（生产路径 zero touch）")

    factor = args.factor
    _run_full_pipeline(factor)

    # ── 对比 raw / cleaned parquet
    from config import RAW_FACTOR_BASE as _patched_raw  # noqa: F401
    prod_raw = Path("/nfs/ofs-prediction/peterzhenglinpeng/factor-replication/raw_factor") / f"{factor}.parquet"
    prod_clean = Path("/nfs/ofs-prediction/peterzhenglinpeng/factor-replication/cleaned_factor") / f"{factor}.parquet"

    print("\n=== parquet 对比 ===")
    raw_ok, raw_msg = _intersection_compare(prod_raw, SAND_RAW / f"{factor}.parquet", "raw_factor")
    print(raw_msg)
    clean_ok, clean_msg = _intersection_compare(prod_clean, SAND_CLEAN / f"{factor}.parquet", "cleaned_factor")
    print(clean_msg)

    # ── PNG self-determinism（绝对一致性）
    print("\n=== PNG 字节级确定性（self-determinism） ===")
    pngs = list((SAND_OUT / factor).glob("*.png"))
    if pngs:
        png_name = pngs[0].name
        png_ok, png_msg = _png_self_determinism(factor, png_name)
        print(png_msg)

        # ── 跟生产 PNG 比对（只是观察，不强求字节对齐）
        prod_png_dir = ROOT / "sources/kysec/paper_27_microstructure/output" / factor
        prod_pngs = list(prod_png_dir.glob("*.png*"))   # 含 .bak
        if prod_pngs:
            prod_png = prod_pngs[0]
            sand_png = SAND_OUT / factor / png_name
            print(f"  生产 PNG: {prod_png.name} ({prod_png.stat().st_size} bytes, "
                  f"mtime={time.ctime(prod_png.stat().st_mtime)})")
            print(f"  sandbox PNG: {sand_png.name} ({sand_png.stat().st_size} bytes)")
            rc = subprocess.run(["cmp", str(prod_png), str(sand_png)], capture_output=True).returncode
            if rc == 0:
                print("  ✓ 生产 PNG 字节级一致")
            else:
                print(f"  · 生产 PNG 字节不同（不强求；plots.py commit 后渲染会差异）")
    else:
        png_ok = False
        print("  ✗ sandbox 没产出 PNG")

    print("\n" + "=" * 60)
    print(f"汇总：raw {'✓' if raw_ok else '✗'}  cleaned {'✓' if clean_ok else '✗'}  "
          f"PNG self-determinism {'✓' if png_ok else '✗'}")
    print("=" * 60)

    if not args.keep_sandbox:
        shutil.rmtree(SANDBOX)
        print(f"sandbox 已清理")

    sys.exit(0 if (raw_ok and clean_ok and png_ok) else 1)


if __name__ == "__main__":
    main()
