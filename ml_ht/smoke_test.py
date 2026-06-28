"""端到端冒烟测试：长表按日期 predicate 切小窗 → 跑 2 epoch → predict + 信号。

设计原则：
1. 不侵入主代码（所有常量 monkeypatch，per `dataset.SPLIT` 因为 preprocess 未用故无需 patch）
2. 切片覆盖 train/valid/test 三个 hardcoded 日期区间，确保所有 split 都有非零样本
3. 缓存切好的小长表/标签，避免重复读 7.6G

切片窗口 [2017-06-01, 2020-02-29]：
    train: 2017-06 ~ 2017-11      (~6 个月)
    valid: 2018-01 ~ 2019-11      (~全部 2018-2019)
    test:  2020-01 ~ 2020-02      (~2 个月)
    之间 embargo 2017-12 / 2019-12 由 preprocess 截断保留。

用法: python -m ml_ht.smoke_test
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# 让 `core` 包和 `ml_ht` 包都可被 import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import ml_ht.dataset as ds
import ml_ht.run as runmod

# ---------------- 配置 ----------------

CACHE = Path("/tmp/opencode/ml_ht_smoke")
CACHE.mkdir(parents=True, exist_ok=True)

SMOKE_LONG = CACHE / "smoke_long.parquet"
SMOKE_LABEL = CACHE / "smoke_label.parquet"

DATE_LO = pd.Timestamp("2017-06-01")
DATE_HI = pd.Timestamp("2020-02-29")


# ---------------- 切片缓存 ----------------

def _slice_long_table() -> None:
    if SMOKE_LONG.exists():
        print(f"[smoke] use cached  {SMOKE_LONG}  ({SMOKE_LONG.stat().st_size/1e6:.1f} MB)")
        return
    print(f"[smoke] slicing long table {DATE_LO.date()} ~ {DATE_HI.date()} ...")
    df = pd.read_parquet(
        ds.LONG_TABLE_PATH,
        filters=[("date", ">=", DATE_LO), ("date", "<=", DATE_HI)],
    )
    df.to_parquet(SMOKE_LONG)
    print(f"[smoke] wrote {df.shape} -> {SMOKE_LONG.stat().st_size/1e6:.1f} MB")
    del df


def _slice_labels() -> None:
    if SMOKE_LABEL.exists():
        print(f"[smoke] use cached  {SMOKE_LABEL}  ({SMOKE_LABEL.stat().st_size/1e6:.1f} MB)")
        return
    print(f"[smoke] slicing labels {DATE_LO.date()} ~ {DATE_HI.date()} ...")
    lw = pd.read_parquet(
        ds.LABEL_PATH,
        filters=[("date", ">=", DATE_LO), ("date", "<=", DATE_HI)],
    )
    lw.to_parquet(SMOKE_LABEL)
    print(f"[smoke] wrote {lw.shape} -> {SMOKE_LABEL.stat().st_size/1e6:.1f} MB")
    del lw


# ---------------- monkeypatch + 跑 ----------------

def main() -> int:
    _slice_long_table()
    _slice_labels()

    # 转向到 /tmp
    ds.LONG_TABLE_PATH = SMOKE_LONG
    ds.LABEL_PATH = SMOKE_LABEL
    runmod.MODEL_DIR = CACHE / "models"
    runmod.SIGNAL_DIR = CACHE / "signals"
    runmod.RUNS_DIR = CACHE / "runs"

    #极小训练量，只验通路
    sys.argv = [
        "ml_ht.run",
        "--train",
        "--predict",
        "--device", "cuda",
        "--lr", "1e-3",
        "--max-epochs", "2",
        "--patience", "1",
        "--batch-size", "4096",
    ]

    runmod.main()

    # 复盘产出
    runs_dir = CACHE / "runs"
    if runs_dir.exists():
        subdirs = sorted(runs_dir.iterdir())
        if subdirs:
            run_dir = subdirs[-1]
            print(f"\n[smoke] === run_dir 内容 ===")
            for f in sorted(run_dir.iterdir()):
                print(f"  {f.name}  ({f.stat().st_size} bytes)")
            hist = run_dir / "history.jsonl"
            if hist.exists():
                print(f"\n[smoke] === history.jsonl 首/末行 ===")
                lines = hist.read_text().splitlines()
                for ln in lines[:1] + (["..."] if len(lines) > 2 else []) + lines[-1:]:
                    print(" ", ln)
            tr = run_dir / "test_report.json"
            if tr.exists():
                import json
                print(f"\n[smoke] === test_report.json ===")
                print(json.dumps(json.loads(tr.read_text()), ensure_ascii=False, indent=2))
    sig_dir = CACHE / "signals"
    if sig_dir.exists():
        files = sorted(sig_dir.glob("*.txt"))
        print(f"\n[smoke] 信号文件数: {len(files)}")
        for f in files[:2] + (["..."] if len(files) > 4 else []) + files[-2:]:
            if f == Path("..."):
                print("  ...")
            else:
                with open(f) as fp:
                    head = [next(fp).strip() for _ in range(3)]
                print(f"  {f.name}  ({f.stat().st_size} B)  head: {head}")
    return 0


if __name__ == "__main__":
    sys.exit(main())