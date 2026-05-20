"""
因子复现 CLI

用法:
    python run.py FACTOR                       # 默认全流程: YOLO + 评估
    python run.py FACTOR --yolo-only           # 只跑 YOLO，不评估
    python run.py FACTOR --evaluate-only       # 只评估（读已有 raw parquet）
    python run.py FACTOR --spec-only -i FILE   # 从研报生成 spec.yaml（不执行）
    python run.py --all                        # 对 specs/* 全部跑全流程
    python run.py --all --evaluate-only        # 对 specs/* 全部仅评估

可选参数:
    --start-date YYYYMMDD       默认见 core/config.DEFAULT_START_DATE
    --end-date YYYYMMDD         默认见 core/config.DEFAULT_END_DATE
    --workers N                 覆盖环境变量 FETCHER_WORKERS（多线程 fetch 并发）
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

from core.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    RAW_FACTOR_DIR,
)
from core.evaluation import evaluate_single_factor
from core.spec_generator import generate_spec_from_research, load_spec_yaml
from core.yolo_engine import run_factor


# ── 单因子三种执行模式 ─────────────────────────────────────


def _load_existing_raw(factor_name: str) -> pd.DataFrame:
    path = RAW_FACTOR_DIR / f"{factor_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"raw 因子不存在: {path}（先 'python run.py {factor_name} --yolo-only' 生成）"
        )
    logger.info(f"📂 加载已有因子: {path}")
    return pd.read_parquet(path)


def run_one(
    factor_name: str,
    mode: str,
    start_date: str,
    end_date: str,
) -> None:
    """
    单因子执行主路径。
    mode ∈ {'full', 'yolo', 'evaluate-only'}
    """
    logger.info("=" * 60)
    logger.info(f"🎯 {factor_name} | mode={mode}")
    logger.info("=" * 60)

    spec_yaml = load_spec_yaml(factor_name)

    if mode == "evaluate-only":
        factor_df = _load_existing_raw(factor_name)
    else:
        factor_df = run_factor(
            factor_name=factor_name,
            spec_yaml=spec_yaml,
            start_date=start_date,
            end_date=end_date,
        )

    if mode in ("full", "evaluate-only"):
        evaluate_single_factor(
            factor_name=factor_name,
            factor_df=factor_df,
            start_date=start_date,
            end_date=end_date,
            spec_yaml=spec_yaml,
        )

    logger.info(f"✅ {factor_name} 完成")


def run_spec_gen(factor_name: str, input_path: Path) -> None:
    """从研报文字生成 spec.yaml（不执行 YOLO）"""
    logger.info("=" * 60)
    logger.info(f"🤖 LLM 生成 spec: {factor_name} ← {input_path}")
    logger.info("=" * 60)
    text = input_path.read_text(encoding="utf-8")
    generate_spec_from_research(text, factor_name)


# ── 工具函数 ──────────────────────────────────────────────


def list_all_factors() -> List[str]:
    """返回 specs/ 下所有具备 spec.yaml 的因子名（已排序）"""
    specs_dir = Path("specs")
    if not specs_dir.exists():
        return []
    return sorted(
        p.name
        for p in specs_dir.iterdir()
        if p.is_dir() and (p / "spec.yaml").exists()
    )


# ── CLI 入口 ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="因子复现 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "factor", nargs="?", help="因子名（specs/<name>/spec.yaml 必须存在）"
    )
    parser.add_argument(
        "--all", action="store_true", help="对 specs/* 全部因子执行（与 <factor> 互斥）"
    )
    parser.add_argument(
        "--input", "-i", type=Path, help="(--spec-only 必填) 研报文字文件路径"
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--yolo-only", action="store_true", help="只跑 YOLO，不评估"
    )
    mode_group.add_argument(
        "--evaluate-only", action="store_true", help="只评估（读已有 raw）"
    )
    mode_group.add_argument(
        "--spec-only", action="store_true", help="只走 LLM 从研报生成 spec.yaml"
    )

    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--workers",
        type=int,
        help="覆盖 FETCHER_WORKERS（多线程 fetch 并发，默认 12）",
    )

    args = parser.parse_args()

    # --workers 覆盖环境变量
    if args.workers is not None:
        os.environ["FETCHER_WORKERS"] = str(args.workers)

    # ── --spec-only 路径 ──
    if args.spec_only:
        if not args.factor:
            parser.error("--spec-only 需要一个 <factor> 位置参数")
        if not args.input:
            parser.error("--spec-only 需要 --input <研报文件>")
        if not args.input.exists():
            parser.error(f"input 文件不存在: {args.input}")
        if args.all:
            parser.error("--spec-only 不能与 --all 同时使用")
        run_spec_gen(args.factor, args.input)
        return

    # ── 其余三种模式: yolo / evaluate-only / full ──
    if args.yolo_only:
        mode = "yolo"
    elif args.evaluate_only:
        mode = "evaluate-only"
    else:
        mode = "full"

    # 因子列表
    if args.all:
        factors = list_all_factors()
        if not factors:
            parser.error("specs/ 目录下没有任何 spec.yaml")
        logger.info(f"批量执行 {len(factors)} 个因子: {factors}")
    elif args.factor:
        factors = [args.factor]
    else:
        parser.error("必须指定 <factor> 或 --all")

    failed: List[str] = []
    for f in factors:
        try:
            run_one(f, mode, args.start_date, args.end_date)
        except Exception as exc:
            logger.exception(f"❌ {f} 失败: {exc}")
            failed.append(f)

    if failed:
        logger.warning(f"⚠️  失败 {len(failed)}/{len(factors)}: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
