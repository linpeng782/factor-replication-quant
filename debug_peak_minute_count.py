"""
单因子调试脚本：peak_minute_count（paper_27 微观结构系列）

功能：
  1. 加载 spec.yaml
  2. 用 YoloEngine 按步骤执行（仅 YOLO，不评估）
  3. 在关键节点加中文 debug 打印，方便一步一步看
  4. 保存中间产物到 debug_output/ 目录

用法：
  直接修改下面 FACTOR / START_DATE / END_DATE 等参数，然后运行：
      python debug_peak_minute_count.py

注意：
  不调用命令行参数，所有配置在脚本内改。
  本脚本只跑 YOLO（不评估），如需评估请改 MODE = "evaluate" 或运行 run.py --evaluate-only。
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

# ============================================================
# 在这里手动改参数
# ============================================================
FACTOR = "peak_minute_count"           # 因子名（裸名即可，全局唯一）
START_DATE = "20100101"                # 数据 fetch 起始日
END_DATE = "20260627"                  # 数据 fetch 结束日
REBUILD = True                         # 是否全量重算（True=覆盖旧文件，False=尝试增量）
SAVE_INTERMEDIATE = True               # 是否保存中间产物
# ============================================================

# 设置数据后端（dquant 分钟）
os.environ["MINUTE_DATA_BACKEND"] = "dquant"

# 调试输出目录
DEBUG_DIR = Path(__file__).parent / "debug_output"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# 日志落盘
logger.add(DEBUG_DIR / "debug_peak_minute_count.log", level="DEBUG", encoding="utf-8")

# 导入核心模块
from core.config import RAW_FACTOR_BASE, DEFAULT_START_DATE, DEFAULT_END_DATE
from core.spec_resolver import (
    resolve_spec_path,
    factor_name_from_arg,
    resolve_namespace_safe,
    incremental_safe,
    max_warmup_window,
)
from core.yolo_engine import DataFetcher, YoloEngine, incremental_append
from core.operators import Context
from core.operators import fetch, compute, transform, rolling, merge  # noqa: F401
from core.operators import (
    minute_intraday_aggregate,
    minute_pricejump_aggregate,
    minute_tide,
    minute_smartmoney,
    minute_dazzle,
    minute_apm_segments,
    load_panel,
    industry_co_momentum,
    cross_section_regress,
    row_aggregate,
    row_polyfit,
    row_correlate,
    rolling_ts_regress,
    rank,
    filter as _filter_mod,
)

# 触发所有 reducer 注册
import core.operators.minute_engine  # noqa: F401


def load_spec(factor_name: str) -> dict:
    """加载 spec.yaml。"""
    spec_path = resolve_spec_path(factor_name)
    logger.info(f"[调试] 加载 spec: {spec_path}")
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    logger.info(f"[调试] spec 因子名: {spec['factor']['name']}, 列: {spec['factor']['column']}")
    logger.info(f"[调试] steps 数量: {len(spec.get('calculation_steps', []))}")
    return spec


def step_by_step_run(spec: dict, start_date: str, end_date: str, rebuild: bool = True):
    """
    一步一步手动执行 spec 中的每个 step，每步结束后打印上下文并保存中间产物。
    """
    factor_name = spec["factor"]["name"]
    factor_column = spec["factor"]["column"]

    fetcher = DataFetcher()
    ctx = Context(factor_name=factor_name)
    ctx.start_date = start_date
    ctx.end_date = end_date
    ctx.trade_date = None
    ctx.start_quarter = f"{start_date[:4]}q1"
    ctx.end_quarter = f"{end_date[:4]}q4"

    # 构建股票池
    universe_cfg = spec.get("universe", {})
    logger.info("[调试] 开始构建股票池...")
    from core.yolo_engine import build_universe

    ctx.universe = build_universe(universe_cfg, start_date, fetcher)
    logger.info(f"[调试] 股票池大小: {len(ctx.universe)} 只")
    logger.info(f"[调试] 前 5 只: {ctx.universe[:5]}")

    # 保存股票池
    if SAVE_INTERMEDIATE:
        pd.Series(ctx.universe, name="order_book_id").to_csv(
            DEBUG_DIR / "step00_universe.csv", index=False
        )

    # 手动执行每个 step
    steps = spec.get("calculation_steps", [])
    for i, step in enumerate(steps, 1):
        action = step["action"]
        label = f"step#{i} {step.get('name') or action}"
        logger.info(f"\n[调试] ▶▶▶ 执行 {label} [action={action}]")

        from core.operators import OpRegistry

        op_func = OpRegistry.get(action)
        before = ctx.schema_snapshot()
        op_func(ctx, step, fetcher)
        ctx.log_schema_diff(before, step_label=f"step#{i}")

        # 保存当前 step 后的主表
        if SAVE_INTERMEDIATE and ctx.has_df("data"):
            df = ctx.get_df("data")
            logger.info(f"[调试] {label} 后主表 shape={df.shape}, 列={list(df.columns)}")
            sample_path = DEBUG_DIR / f"step{i:02d}_{action}_sample.csv"
            df.head(1000).to_csv(sample_path, index=False)
            logger.info(f"[调试] 已保存前 1000 行到: {sample_path}")

    # 主表 → 宽表
    logger.info("\n[调试] 所有 steps 执行完毕，准备 pivot 成宽表...")
    if not ctx.has_df("data"):
        raise RuntimeError("主表 'data' 不存在")
    data = ctx.get_df("data")
    if factor_column not in data.columns:
        raise RuntimeError(f"factor.column={factor_column} 不在主表中，现有列: {list(data.columns)}")

    logger.info(f"[调试] 主表 info: shape={data.shape}, 列={list(data.columns)}")
    logger.info(f"[调试] 因子列非空数: {data[factor_column].notna().sum()}")
    logger.info(f"[调试] 日期范围: {data['date'].min()} ~ {data['date'].max()}")
    logger.info(f"[调试] 股票数: {data['order_book_id'].nunique()}")

    wide = data.pivot(index="date", columns="order_book_id", values=factor_column)
    wide.index = pd.to_datetime(wide.index)
    logger.info(f"[调试] 宽表 shape={wide.shape}, 日期数={len(wide)}, 股票数={len(wide.columns)}")
    logger.info(f"[调试] 宽表非空单元数: {wide.notna().values.sum():,}")
    logger.info(f"[调试] 宽表覆盖率: {wide.notna().mean().mean():.2%}")

    # 落盘
    out_dir = RAW_FACTOR_BASE / resolve_namespace_safe(factor_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{factor_name_from_arg(factor_name)}.parquet"

    if not rebuild and out_path.exists():
        old_idx = pd.to_datetime(pd.read_parquet(out_path, columns=[]).index)
        last = old_idx.max() if len(old_idx) else None
    else:
        last = None

    combined = incremental_append(out_path, wide, last, rebuild=rebuild)
    logger.info(f"[调试] 已写入: {out_path}, shape={combined.shape}")
    logger.info(f"[调试] 完成！")

    return combined


def main():
    logger.info("=" * 60)
    logger.info("[调试] 启动 peak_minute_count 单因子调试")
    logger.info(f"[调试] FACTOR={FACTOR}, START={START_DATE}, END={END_DATE}, REBUILD={REBUILD}")
    logger.info("=" * 60)

    spec = load_spec(FACTOR)
    combined = step_by_step_run(spec, START_DATE, END_DATE, rebuild=REBUILD)

    logger.info(f"[调试] 最终宽表 shape={combined.shape}")
    logger.info(f"[调试] 最终宽表日期范围: {combined.index.min().date()} ~ {combined.index.max().date()}")


if __name__ == "__main__":
    main()
