"""
roic_ttm_ind_rnk8 集成测试
用少量股票快速验证 Spec 流程
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.operators import OpRegistry
from core.operators import fetch, compute, filter, rank, transform, rolling
from core.yolo_engine import DataFetcher


def test_full_pipeline():
    """用3只股票验证完整计算链路"""
    fetcher = DataFetcher()
    ctx = {
        "_start_date": "2024-01-01",
        "_end_date": "2024-06-30",
        "_start_quarter": "2024q1",
        "_end_quarter": "2024q2",
        "_universe": ["000002.XSHE", "600519.XSHG", "000001.XSHE"],
    }

    print("=" * 60)
    print("roic_ttm_ind_rnk8 集成测试 (3只股票)")
    print("=" * 60)

    # Step 1: fetch ROIC_TTM
    print("\n▶ Step 1: 获取ROIC_TTM")
    step1 = {
        "action": "fetch",
        "api": "get_factor",
        "fields": ["return_on_invested_capital_ttm"],
        "output": "roic_data",
    }
    op_fetch = OpRegistry.get("fetch")
    df1 = op_fetch(ctx, step1, fetcher)
    print(f"   → shape={df1.shape}, columns={list(df1.columns)}")
    print(f"   → 非空ROIC_TTM: {df1['return_on_invested_capital_ttm'].notna().sum()}")

    # Step 2: fetch 行业分类
    print("\n▶ Step 2: 获取中信行业分类")
    step2 = {
        "action": "fetch",
        "api": "custom",
        "command": "__internal__zx2019_industry",
        "columns": ["first_industry_name", "order_book_id", "start_date"],
        "output": "industry_data",
    }
    df2 = op_fetch(ctx, step2, fetcher)
    print(f"   → shape={df2.shape}, columns={list(df2.columns)}")
    print(f"   → 日期范围: {df2['date'].min()} ~ {df2['date'].max()}")

    # Step 3: merge
    print("\n▶ Step 3: Merge行业分类")
    step3 = {
        "action": "compute",
        "input": "roic_data",
        "formula": "roic_ttm = return_on_invested_capital_ttm",
        "merge_columns": ["first_industry_name"],
        "output": "merged_data",
    }
    op_compute = OpRegistry.get("compute")
    df3 = op_compute(ctx, step3, fetcher)
    print(f"   → shape={df3.shape}, columns={list(df3.columns)}")
    print(f"   → 有行业分类的行数: {df3['first_industry_name'].notna().sum()}")

    # Step 4: rank
    print("\n▶ Step 4: 行业内排名")
    step4 = {
        "action": "rank",
        "method": "industry_rank",
        "input": "merged_data",
        "rank_column": "roic_ttm",
        "group_column": "first_industry_name",
        "pct": False,
        "ascending": False,
        "rank_method": "min",
        "output_col": "ind_rank",
        "output": "ranked_data",
    }
    op_rank = OpRegistry.get("rank")
    df4 = op_rank(ctx, step4, fetcher)
    print(f"   → shape={df4.shape}")
    print(f"   → ind_rank 统计: min={df4['ind_rank'].min()}, max={df4['ind_rank'].max()}, 非空={df4['ind_rank'].notna().sum()}")

    # Step 5: rolling
    print("\n▶ Step 5: 过去8期排名最小值")
    step5 = {
        "action": "rolling",
        "input": "ranked_data",
        "columns": ["ind_rank"],
        "window": 8,
        "min_periods": 4,
        "agg": "min",
        "group_by": "order_book_id",
        "on": "return_on_invested_capital_ttm",
        "fill_method": "ffill",
        "output": "rolled_data",
    }
    op_rolling = OpRegistry.get("rolling")
    df5 = op_rolling(ctx, step5, fetcher)
    print(f"   → shape={df5.shape}")
    print(f"   → ind_rank (rolled) 统计: min={df5['ind_rank'].min()}, max={df5['ind_rank'].max()}, 非空={df5['ind_rank'].notna().sum()}")

    # Step 6: rename
    print("\n▶ Step 6: 重命名")
    step6 = {
        "action": "compute",
        "input": "rolled_data",
        "formula": "roic_ttm_ind_rnk8 = ind_rank",
        "output": "factor",
    }
    df6 = op_compute(ctx, step6, fetcher)
    print(f"   → shape={df6.shape}")
    print(f"   → 最终列: {list(df6.columns)}")

    # 验证最终结果
    factor_df = df6[["order_book_id", "date", "roic_ttm_ind_rnk8"]].dropna()
    print(f"\n✅ 最终因子值: {len(factor_df)} 行")
    print(factor_df.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("集成测试通过 ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()
