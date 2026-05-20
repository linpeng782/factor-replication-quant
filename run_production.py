"""
因子生产入口脚本

用法:
    python run_production.py --factor roe_mrq_new --start 20100101 --end 20251231
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.yolo_engine import run_factor
from core.validation import analyze_factor


def main():
    parser = argparse.ArgumentParser(description="因子生产")
    parser.add_argument("--factor", default="pe_ttm_delta60", help="因子名称（对应 specs/<factor>/spec.yaml），默认 roe_mrq_new")
    parser.add_argument("--start", default="20100101", help="开始日期，默认 20100101")
    parser.add_argument("--end", default="20251231", help="结束日期，默认 20251231")
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"生产因子: {args.factor}")
    print(f"日期范围: {args.start} ~ {args.end}")
    print(f"=" * 60)

    result = run_factor(
        factor_name=args.factor,
        start_date=args.start,
        end_date=args.end,
    )

    # 检查报告
    factor_dir = Path(f"/nfs/ofs-prediction/peterzhenglinpeng/factor-replication/{args.factor}")
    analyze_factor(result, args.factor, factor_dir)

    print(f"\n生产完成!")
    print(f"产出路径: /nfs/ofs-prediction/peterzhenglinpeng/factor-replication/{args.factor}/{args.factor}.parquet")
    print(f"数据维度: {result.shape} (日期 × 股票)")
    print(f"非空值数: {result.notna().sum().sum()}")


if __name__ == "__main__":
    main()
