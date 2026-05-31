"""
因子复现 CLI（仅日常执行；spec 生成请用 `python -m core.spec_generator`）

用法:
    python run.py FACTOR                 # 默认全流程: YOLO + 评估
    python run.py FACTOR --yolo-only     # 只跑 YOLO，不评估
    python run.py FACTOR --evaluate-only # 只评估（读已有 raw parquet）

FACTOR 两种形态都接受：
    peak_minute_count                                   # 裸名，自动扫 sources/ 定位（要求全局唯一）
    kysec/paper_27_microstructure/peak_minute_count     # 限定路径，重名时消歧

可选参数:
    --start-date YYYYMMDD   默认 core.config.DEFAULT_START_DATE
    --end-date YYYYMMDD     默认 core.config.DEFAULT_END_DATE
    --workers N             覆盖环境变量 FETCHER_WORKERS（多线程 fetch 并发）

约定:
    spec 文件:    sources/<publisher>/<group>/specs/<FACTOR>/spec.yaml
    研报输入:     sources/<publisher>/<group>/{input.md, inputs/<FACTOR>.md}（与 run.py 无关）
    因子产出:    factors/{raw,cleaned,neu}/<source>/<FACTOR>.parquet（source 由 spec 路径推导）
    评估输出:    output/<FACTOR>/evaluation_<range>__{cleaned,neu}.png
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from core.config import (
    DEFAULT_END_DATE,
    DEFAULT_EVAL_END_DATE,
    DEFAULT_EVAL_START_DATE,
    DEFAULT_START_DATE,
    RAW_FACTOR_BASE,
)
from core.evaluation import evaluate_single_factor
from core.spec_generator import load_spec_yaml
from core.spec_resolver import resolve_source_safe
from core.yolo_engine import run_factor


def _load_existing_raw(factor_name: str) -> pd.DataFrame:
    path = RAW_FACTOR_BASE / resolve_source_safe(factor_name) / f"{factor_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"raw 因子不存在: {path}（先 'python run.py {factor_name} --yolo-only' 生成）"
        )
    logger.info(f"📂 加载已有因子: {path}")
    return pd.read_parquet(path)


def run_one(
    factor_name: str,
    mode: str,
    fetch_start: str,
    fetch_end: str,
    eval_start: str,
    eval_end: str,
) -> None:
    """
    单因子执行主路径。
    mode ∈ {'full', 'yolo', 'evaluate-only'}

    fetch_start/fetch_end : 米筐拉数据 + panel 落盘窗口（默认 DEFAULT_START_DATE/END_DATE）
    eval_start/eval_end   : IC / 分层评估窗口，从 panel 中截取（默认 DEFAULT_EVAL_*）

    fetch ⊋ eval：fetch 区间通常覆盖 eval 区间，前面多出来的部分是 warm-up（防止
    rolling / yoy / qoq 因子在评估区间起点是 NaN）。
    """
    logger.info("=" * 60)
    logger.info(f"🎯 {factor_name} | mode={mode}")
    logger.info(f"   fetch: {fetch_start} ~ {fetch_end}")
    logger.info(f"   eval : {eval_start} ~ {eval_end}")
    logger.info("=" * 60)

    spec_yaml = load_spec_yaml(factor_name)

    if mode == "evaluate-only":
        factor_df = _load_existing_raw(factor_name)
    else:
        factor_df = run_factor(
            factor_name=factor_name,
            spec_yaml=spec_yaml,
            start_date=fetch_start,
            end_date=fetch_end,
        )

    if mode in ("full", "evaluate-only"):
        evaluate_single_factor(
            factor_name=factor_name,
            factor_df=factor_df,
            start_date=eval_start,
            end_date=eval_end,
            spec_yaml=spec_yaml,
        )

    logger.info(f"✅ {factor_name} 完成")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="因子复现 CLI（仅日常执行；spec 生成请用 python -m core.spec_generator）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "factor",
        help="因子名（裸名或 pub/group/factor 限定路径；spec 在 sources/<pub>/<group>/specs/<factor>/）",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--yolo-only", action="store_true", help="只跑 YOLO，不评估"
    )
    mode_group.add_argument(
        "--evaluate-only", action="store_true", help="只评估（读已有 raw）"
    )

    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"fetch 起始日（panel 落盘窗口起点；默认 {DEFAULT_START_DATE}）",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help=f"fetch 结束日（默认 {DEFAULT_END_DATE}）",
    )
    parser.add_argument(
        "--eval-start-date",
        default=DEFAULT_EVAL_START_DATE,
        help=f"评估起始日（IC/ICIR 区间；默认 {DEFAULT_EVAL_START_DATE}）",
    )
    parser.add_argument(
        "--eval-end-date",
        default=DEFAULT_EVAL_END_DATE,
        help=f"评估结束日（默认 {DEFAULT_EVAL_END_DATE}）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="覆盖 FETCHER_WORKERS（多线程 fetch 并发，默认 12）",
    )

    args = parser.parse_args()

    if args.workers is not None:
        os.environ["FETCHER_WORKERS"] = str(args.workers)

    # ── 运行日志落盘：sources/<pub>/<group>/output/<factor>/run_<timestamp>.log ──
    try:
        from core.spec_resolver import resolve_output_dir
        log_dir = resolve_output_dir(args.factor)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"run_{ts}.log"
        logger.add(log_path, level="DEBUG", encoding="utf-8",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
        logger.info(f"📝 运行日志: {log_path}")
    except Exception:
        pass  # log 落盘失败不影响主流程

    if args.yolo_only:
        mode = "yolo"
    elif args.evaluate_only:
        mode = "evaluate-only"
    else:
        mode = "full"

    try:
        run_one(
            args.factor,
            mode,
            fetch_start=args.start_date,
            fetch_end=args.end_date,
            eval_start=args.eval_start_date,
            eval_end=args.eval_end_date,
        )
    except Exception as exc:
        logger.exception(f"❌ {args.factor} 失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
