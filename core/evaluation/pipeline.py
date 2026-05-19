"""
单因子评估 Pipeline
------------------------------------------------------------
从已生产的因子值出发，自动完成：
  1. 加载 mask + 因子清洗
  2. 加载 vwap + 构建 forward returns
  3. IC 计算 + 分层回测
  4. direction 自动判断
  5. 生成 evaluation.png + 汇总指标

输入:
  - factor_values.parquet (wide 面板, date × order_book_id)
  - combo_mask_long.parquet / new_stock_mask_long.parquet
  - vwap_post.parquet

输出:
  - cleaned.parquet
  - evaluation.png
  - 增强版 report.md
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

import core.config as config
from core.cleaning import load_filter_masks, prepare_factor
from core.evaluation.ic import compute_ic_report, compute_ic_series
from core.evaluation.returns import build_forward_returns
from core.evaluation.layered import layered_backtest
from core.evaluation.plots import plot_factor_report


def evaluate_single_factor(
    factor_name: str,
    factor_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    spec_yaml: Optional[dict] = None,
    output_dir: Optional[Path] = None,
    ic_horizons: tuple = (5, 10, 20),
    primary_ic_horizon: int = 5,
    layer_rebalance: int = 5,
    layer_groups: int = 5,
    mad_n: float = 3.0,
    plot: bool = True,
    skip_cleaning: bool = False,
) -> dict:
    """
    单因子端到端评估

    参数:
        factor_name        : 因子名
        factor_df          : (T, N) 原始因子面板（wide 格式，date × order_book_id）
        start_date, end_date: 评估区间（用于截断 mask 和 vwap）
        spec_yaml          : Spec YAML dict（用于读取 evaluation 配置，可为 None）
        output_dir         : 输出目录，默认 config.OUTPUT_DIR / factor_name
        ic_horizons        : IC 持有期列表
        primary_ic_horizon : direction 判断 + 主图使用的 horizon
        layer_rebalance    : 分层调仓周期（天）
        layer_groups       : 分层组数
        mad_n              : MAD 去极值阈值
        plot               : 是否生成 evaluation.png
        skip_cleaning      : 如果为 True，跳过清洗步骤，直接使用传入的 factor_df 作为已清洗因子
                             （用于 raw 文件损坏时，直接用已有的清洗后因子重新评估）

    返回:
        dict: {
            'success': bool,
            'direction': int,
            'ic_summary': pd.DataFrame,
            'layered_summary': pd.DataFrame,
            'cleaned_factor': pd.DataFrame,
            'error': str (仅 success=False)
        }
    """
    try:
        # 从 spec_yaml 读取配置（如有）
        if spec_yaml:
            ev_cfg = spec_yaml.get("evaluation", {})
            ic_horizons = ev_cfg.get("horizons", ic_horizons)
            primary_ic_horizon = ev_cfg.get("primary_horizon", primary_ic_horizon)
            layer_rebalance = ev_cfg.get("layer_rebalance", layer_rebalance)
            layer_groups = ev_cfg.get("layer_groups", layer_groups)
            mad_n = ev_cfg.get("mad_n", mad_n)

        report_dir = Path(output_dir or config.OUTPUT_DIR / factor_name)
        report_dir.mkdir(parents=True, exist_ok=True)
        config.RAW_FACTOR_DIR.mkdir(parents=True, exist_ok=True)
        config.CLEANED_FACTOR_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info(f"📊 开始评估因子: {factor_name}")
        logger.info(f"  factor shape: {factor_df.shape}")
        logger.info(f"  date range: {factor_df.index.min().date()} ~ {factor_df.index.max().date()}")
        logger.info(f"  IC horizons: {ic_horizons}, primary: {primary_ic_horizon}d")
        logger.info(f"  Layered: {layer_groups}组 × {layer_rebalance}日调仓")
        logger.info("=" * 60)

        # ==================== 1. 加载完整 mask & 清洗因子（保留完整时间范围）====================
        if skip_cleaning:
            logger.info("[1/4] skip_cleaning=True，直接使用传入的已清洗因子...")
            factor_clean = factor_df.copy()
        else:
            logger.info("[1/4] 加载完整 mask & 清洗因子...")
            # 先加载完整 mask（不截断），确保清洗后的因子保留原始数据的完整时间范围
            pre_mask_full, post_mask_full = load_filter_masks(
                reindex_columns=factor_df.columns,
            )

            # 用 factor_df 的时间范围截取 mask，避免 mask 比 factor 长导致大量 NaN
            start_dt = factor_df.index.min()
            end_dt = factor_df.index.max()
            pre_mask = pre_mask_full.loc[start_dt:end_dt]
            post_mask = post_mask_full.loc[start_dt:end_dt]

            factor_clean = prepare_factor(
                factor=factor_df,
                pre_mask=pre_mask,
                post_mask=post_mask,
                mad_n=mad_n,
            )

            # 保存清洗后因子到外部目录（完整时间范围）
            cleaned_path = factor_dir / f"{factor_name}.parquet"
            factor_clean.to_parquet(cleaned_path)
            logger.info(
                f"  -> 清洗后因子已保存: {cleaned_path} "
                f"(shape={factor_clean.shape}, "
                f"range={factor_clean.index.min().date()} ~ {factor_clean.index.max().date()})"
            )

        # ==================== 2. 加载 vwap & 构建 forward returns ====================
        logger.info("[2/4] 加载 vwap & 构建 forward returns...")
        vwap_long = pd.read_parquet(config.VWAP_POST_PATH, columns=["order_book_id", "datetime", "post_vwap"])
        vwap_long["datetime"] = pd.to_datetime(vwap_long["datetime"])
        vwap_wide = (
            vwap_long.set_index(["datetime", "order_book_id"])["post_vwap"]
            .unstack(level="order_book_id")
            .sort_index()
        )

        # 评估时截取：根据 start/end_date 从清洗因子中截断
        factor_eval = factor_clean.copy()
        if start_date:
            factor_eval = factor_eval.loc[factor_eval.index >= pd.Timestamp(start_date)]
        if end_date:
            factor_eval = factor_eval.loc[factor_eval.index <= pd.Timestamp(end_date)]

        # 对齐 vwap 到评估区间的 factor
        vwap_aligned = vwap_wide.reindex(index=factor_eval.index, columns=factor_eval.columns)
        factor_clean = factor_eval  # 后续分析用截取后的因子

        # 构建 forward returns（同时截断 factor_clean 到 vwap 有效日期）
        forward_returns = build_forward_returns(vwap_aligned, horizons=ic_horizons)
        # 1d return 用于分层回测
        return_1d = build_forward_returns(vwap_aligned, horizons=(1,))[1]

        logger.info(f"  -> vwap aligned shape: {vwap_aligned.shape}")

        # ==================== 3. Direction 判断 ====================
        logger.info("[3/4] Direction 判断 & IC / 分层回测...")
        probe = compute_ic_series(
            factor_clean, forward_returns[primary_ic_horizon], method="spearman"
        )
        probe_mean = float(probe.dropna().mean())
        direction = -1 if probe_mean < 0 else 1
        factor_for_analysis = factor_clean if direction == 1 else -factor_clean
        logger.info(f"  -> raw IC mean ({primary_ic_horizon}d) = {probe_mean:+.4f}, direction = {direction:+d}")

        # ==================== 4. IC Report ====================
        ic_summary_df, ic_series = compute_ic_report(
            factor_for_analysis, forward_returns, method="spearman"
        )

        # ==================== 5. Layered Backtest ====================
        layered_result = layered_backtest(
            factor_for_analysis,
            return_1d,
            n=layer_rebalance,
            g=layer_groups,
        )



        # ==================== 6. 绘图 ====================
        if plot:
            plot_path = report_dir / "evaluation.png"
            plot_factor_report(
                factor_name=factor_name,
                factor_clean=factor_clean,  # 分布图用原始因子，不翻转
                ic_series_dict=ic_series,
                ic_summary_df=ic_summary_df,
                layered_result=layered_result,
                output_path=plot_path,
                primary_ic_horizon=primary_ic_horizon,
                direction=direction,
            )

        logger.info("=" * 60)
        logger.info(f"✅ 因子 {factor_name} 评估完成")
        logger.info("=" * 60)

        return {
            "success": True,
            "direction": direction,
            "ic_summary": ic_summary_df,
            "layered_summary": layered_result["summary"],
            "cleaned_factor": factor_clean,
            "monotonicity": layered_result.get("monotonicity", np.nan),
        }

    except Exception as e:
        logger.exception(f"[{factor_name}] 评估失败: {e}")
        return {
            "success": False,
            "error": str(e),
        }
