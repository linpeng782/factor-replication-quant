"""
因子复现 Agent 系统 —— 主入口

完整 pipeline：
    输入 → 解析 → Spec生成 → 人机确认 → YOLO执行 → 因子清洗 → 单因子检验 → 报告输出

用法：
    # 手动模式（推荐，配合 kimi CLI 使用）
    python run.py --input "roe_pyoy_mrq：单季度ROE同比（仅保留分母>0的因子值）" --mode manual

    # 自动模式（需要配置 LLM API Key）
    python run.py --input "..." --mode auto

    # 直接执行已确认的因子（跳过 Spec 生成和确认）
    python run.py --factor roe_pyoy_mrq --yolo-only

    # 执行 + 自动评估
    python run.py --factor roe_pyoy_mrq --yolo-only --evaluate

    # 仅评估（已有因子值）
    python run.py --factor roe_pyoy_mrq --evaluate-only

    # 批量执行所有已确认的因子
    python run.py --batch-confirmed
"""

import argparse
import sys
from pathlib import Path

import yaml
import pandas as pd

from core.spec_generator import load_spec_yaml
from core.spec_validator import is_confirmed, validate_and_confirm
from core.yolo_engine import run_factor
from core.report_generator import generate_report
from core.evaluation import evaluate_single_factor


def main():
    parser = argparse.ArgumentParser(description="因子复现 Agent 系统")
    parser.add_argument("--input", "-i", type=str, help="输入文本或文件路径")
    parser.add_argument("--factor", "-f", type=str, help="指定因子名称（直接执行模式）")
    parser.add_argument("--mode", "-m", type=str, default="manual", choices=["manual", "auto"],
                        help="Spec 生成模式: manual（手动贴到 LLM）或 auto（自动调用 API）")
    parser.add_argument("--yolo-only", action="store_true",
                        help="跳过 Spec 生成和确认，直接执行 YOLO")
    parser.add_argument("--evaluate", action="store_true",
                        help="YOLO 执行后自动进行因子评估（清洗 + IC + 分层）")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="跳过 YOLO，仅对已有因子值进行评估")
    parser.add_argument("--batch-confirmed", action="store_true",
                        help="批量执行所有已确认的因子")
    parser.add_argument("--start-date", type=str, default="20160101", help="回测开始日期")
    parser.add_argument("--end-date", type=str, default="20251231", help="回测结束日期")
    parser.add_argument("--trade-date", type=str, help="单交易日（用于构建股票池）")
    parser.add_argument("--auto-confirm", action="store_true",
                        help="自动跳过人机确认（仅对已确认过的因子有效）")

    args = parser.parse_args()

    # ── 模式1: 批量执行已确认因子 ─────────────────────
    if args.batch_confirmed:
        specs_dir = Path("specs")
        if not specs_dir.exists():
            print("❌ specs/ 目录不存在")
            sys.exit(1)

        confirmed_factors = []
        for factor_dir in specs_dir.iterdir():
            if factor_dir.is_dir() and is_confirmed(factor_dir.name):
                confirmed_factors.append(factor_dir.name)

        print(f"发现 {len(confirmed_factors)} 个已确认因子: {confirmed_factors}")
        for name in confirmed_factors:
            print(f"\n{'='*60}")
            run_single_factor(
                factor_name=name,
                yolo_only=True,
                evaluate=args.evaluate,
                start_date=args.start_date,
                end_date=args.end_date,
                trade_date=args.trade_date,
            )
        return

    # ── 模式2: 直接执行指定因子 ───────────────────────
    if args.factor:
        # evaluate-only 模式：跳过 YOLO，直接评估
        if args.evaluate_only:
            run_single_factor(
                factor_name=args.factor,
                yolo_only=False,
                evaluate=True,
                skip_yolo=True,
                start_date=args.start_date,
                end_date=args.end_date,
                trade_date=args.trade_date,
            )
            return

        run_single_factor(
            factor_name=args.factor,
            yolo_only=args.yolo_only,
            evaluate=args.evaluate,
            start_date=args.start_date,
            end_date=args.end_date,
            trade_date=args.trade_date,
        )
        return

    # 没指定 --factor 也没 --batch-confirmed → 报错
    print("❌ 请指定 --factor <name> 或 --batch-confirmed")
    print("   研报 → spec.yaml 现走 LLM 生成链路：")
    print("   python -m core.spec_generator <factor_name> --input <research.md>")
    parser.print_help()
    sys.exit(1)


def run_single_factor(
    factor_name: str,
    yolo_only: bool = False,
    evaluate: bool = False,
    skip_yolo: bool = False,
    start_date: str = "20160101",
    end_date: str = "20251231",
    trade_date: str = None,
):
    """执行单个因子的完整或部分流程"""
    print(f"\n{'='*60}")
    print(f"🎯 执行因子: {factor_name}")
    print(f"{'='*60}")

    # 加载 Spec
    try:
        spec_yaml = load_spec_yaml(factor_name)
    except FileNotFoundError:
        print(f"❌ Spec 文件不存在: specs/{factor_name}/spec.yaml")
        return

    # 如果不是 yolo_only，需要检查确认状态
    if not yolo_only and not skip_yolo:
        if not is_confirmed(factor_name):
            print(f"⚠️ 因子 {factor_name} 未确认，请先运行确认流程")
            return

    factor_df = None

    # YOLO 执行
    if not skip_yolo:
        try:
            factor_df = run_factor(
                factor_name=factor_name,
                spec_yaml=spec_yaml,
                start_date=start_date,
                end_date=end_date,
                trade_date=trade_date,
            )
        except Exception as e:
            print(f"❌ YOLO 执行失败: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        # evaluate-only 模式：从磁盘读取已有因子值
        from core.config import RAW_FACTOR_DIR
        factor_path = RAW_FACTOR_DIR / f"{factor_name}.parquet"
        if not factor_path.exists():
            print(f"❌ 因子值文件不存在: {factor_path}")
            return
        print(f"📂 加载已有因子值: {factor_path}")
        factor_df = pd.read_parquet(factor_path)

    # 因子评估
    evaluation_results = None
    if evaluate and factor_df is not None:
        try:
            evaluation_results = evaluate_single_factor(
                factor_name=factor_name,
                factor_df=factor_df,
                start_date=start_date,
                end_date=end_date,
                spec_yaml=spec_yaml,
            )
        except Exception as e:
            print(f"⚠️ 因子评估失败: {e}")
            import traceback
            traceback.print_exc()

    # 生成报告
    try:
        generate_report(
            factor_name=factor_name,
            factor_df=factor_df,
            spec_yaml=spec_yaml,
            evaluation_results=evaluation_results,
        )
    except Exception as e:
        print(f"⚠️ 报告生成失败: {e}")

    print(f"\n✅ 因子 {factor_name} 全流程执行完毕")


if __name__ == "__main__":
    main()
