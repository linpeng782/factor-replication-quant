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
from alpha_shared.cleaning.mask_loader import load_filter_masks
from alpha_shared.cleaning.preprocess import prepare_factor
from alpha_shared.evaluation.ic import compute_ic_report, compute_ic_series
from alpha_shared.evaluation.returns import build_forward_returns
from alpha_shared.evaluation.layered import layered_backtest
from alpha_shared.neu import neutralize
from core.eval_plots import plot_factor_report


def _analyze_and_plot(
    factor_eval: pd.DataFrame,
    *,
    factor_name: str,
    variant: str,
    forward_returns: dict,
    return_1d: pd.DataFrame,
    ic_horizons: tuple,
    primary_ic_horizon: int,
    layer_rebalance: int,
    layer_groups: int,
    report_dir: Path,
    start_date: Optional[str],
    end_date: Optional[str],
    plot: bool,
) -> dict:
    """对单个因子变体（cleaned / neu）做 direction 判断 + IC + 分层 + 出图。

    出图命名：evaluation_<start>_<end>__<variant>.png（两版并存，互不覆盖）。
    """
    probe = compute_ic_series(
        factor_eval, forward_returns[primary_ic_horizon], method="spearman"
    )
    direction = -1 if float(probe.dropna().mean()) < 0 else 1
    factor_for_analysis = factor_eval if direction == 1 else -factor_eval

    ic_summary_df, ic_series = compute_ic_report(
        factor_for_analysis, forward_returns, method="spearman"
    )
    layered_result = layered_backtest(
        factor_for_analysis, return_1d, n=layer_rebalance, g=layer_groups
    )

    if plot:
        if start_date and end_date:
            tag = f"{str(start_date).replace('-', '')}_{str(end_date).replace('-', '')}"
        else:
            tag = "full"
        plot_path = report_dir / f"evaluation_{tag}__{variant}.png"
        plot_factor_report(
            factor_name=f"{factor_name} [{variant}]",
            factor_clean=factor_eval,  # 分布图用本变体因子，不翻转
            ic_series_dict=ic_series,
            ic_summary_df=ic_summary_df,
            layered_result=layered_result,
            output_path=plot_path,
            primary_ic_horizon=primary_ic_horizon,
            direction=direction,
        )

    return {
        "direction": direction,
        "ic_summary": ic_summary_df,
        "layered_summary": layered_result["summary"],
        "monotonicity": layered_result.get("monotonicity", np.nan),
    }


def evaluate_single_factor(
    factor_name: str,
    factor_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    spec_yaml: Optional[dict] = None,
    output_dir: Optional[Path] = None,
    namespace: Optional[str] = None,
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

        # 命名空间分桶：cleaned/neu/output 落 <stage>/<source>/<group>/<factor>
        # 显式传入 namespace（如 alpha158/kline，无 spec 的批量因子用）→ 直接用；
        # 否则回退到从 spec 路径推导（cxl/kysec/founder 等研报因子，行为不变）。
        if namespace is None:
            from core.spec_resolver import resolve_namespace_safe
            namespace = resolve_namespace_safe(factor_name)

        if output_dir is not None:
            report_dir = Path(output_dir)
        elif namespace not in (None, "_misc/_misc"):
            report_dir = config.OUTPUT_DIR / namespace / factor_name
        else:
            from core.spec_resolver import resolve_output_dir

            report_dir = resolve_output_dir(factor_name)
        report_dir.mkdir(parents=True, exist_ok=True)

        cleaned_dir = config.CLEANED_FACTOR_BASE / namespace
        neu_dir = config.NEU_FACTOR_BASE / namespace
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        neu_dir.mkdir(parents=True, exist_ok=True)

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
            # mask 路径在此注入（alpha_shared.load_filter_masks 自身不依赖任何 config）
            pre_mask_full, post_mask_full = load_filter_masks(
                combo_mask_path=config.COMBO_MASK_PATH,
                new_stock_mask_path=config.NEW_STOCK_MASK_PATH,
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
            cleaned_path = cleaned_dir / f"{factor_name}.parquet"
            factor_clean.to_parquet(cleaned_path)
            logger.info(
                f"  -> 清洗后因子已保存: {cleaned_path} "
                f"(shape={factor_clean.shape}, "
                f"range={factor_clean.index.min().date()} ~ {factor_clean.index.max().date()})"
            )

        # ==================== 1b. 行业市值中性化（强制：清洗 → 中性化 → 再标准化）====================
        logger.info("[1b/4] 行业市值中性化（行业 one-hot + log市值，取残差）...")
        industry = pd.read_parquet(config.INDUSTRY_PANEL_ZX_PATH)
        industry.index = pd.to_datetime(industry.index)
        size = pd.read_parquet(config.MARKET_CAP_PANEL_PATH)
        size.index = pd.to_datetime(size.index)
        factor_neu = neutralize(factor_clean, industry, size, restandardize=True)
        neu_path = neu_dir / f"{factor_name}.parquet"
        factor_neu.to_parquet(neu_path)
        logger.info(f"  -> 中性化因子已保存: {neu_path} (shape={factor_neu.shape})")

        # ==================== 2. 加载 forward returns + 截取评估区间 ====================
        logger.info("[2/4] 加载 forward returns（labels 直读）...")

        def _truncate(df):
            if start_date:
                df = df.loc[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df.loc[df.index <= pd.Timestamp(end_date)]
            return df

        cleaned_eval = _truncate(factor_clean)
        neu_eval = _truncate(factor_neu)

        # 直接读 labels parquet——跟 alpha-engine 用同一文件，bit-exact 一致
        # 缺失 horizon 时 fall back 到 vwap_panel 现算（非 bit-exact 但仍 PIT）
        def _load_forward_return(h):
            label_path = config.LABELS_DIR / f"forward_return_{h}d.parquet"
            if label_path.exists():
                df = pd.read_parquet(label_path)
                df.index = pd.to_datetime(df.index)
                return df
            logger.warning(
                f"forward_return_{h}d.parquet 不存在，从 vwap_panel.parquet 现算"
            )
            vwap_wide = pd.read_parquet(config.VWAP_PANEL_PATH)
            vwap_wide.index = pd.to_datetime(vwap_wide.index)
            return build_forward_returns(vwap_wide.sort_index(), horizons=[h])[h]

        # cleaned 与 neu 同 index×columns，forward returns 对齐到该网格一次即可
        forward_returns = {
            h: _load_forward_return(h).reindex(
                index=cleaned_eval.index, columns=cleaned_eval.columns
            )
            for h in ic_horizons
        }
        return_1d = _load_forward_return(1).reindex(
            index=cleaned_eval.index, columns=cleaned_eval.columns
        )
        logger.info(
            f"  -> 加载 horizons={list(forward_returns.keys())} + 1d，shape={return_1d.shape}"
        )

        # ==================== 3-6. 两版各自评估 + 出图（cleaned 对照 / neu 生产）====================
        logger.info("[3/4] 评估 cleaned（清洗对照版）...")
        _kw = dict(
            factor_name=factor_name, forward_returns=forward_returns, return_1d=return_1d,
            ic_horizons=ic_horizons, primary_ic_horizon=primary_ic_horizon,
            layer_rebalance=layer_rebalance, layer_groups=layer_groups,
            report_dir=report_dir, start_date=start_date, end_date=end_date, plot=plot,
        )
        cleaned_res = _analyze_and_plot(cleaned_eval, variant="cleaned", **_kw)
        logger.info("[4/4] 评估 neu（行业市值中性化生产版）...")
        neu_res = _analyze_and_plot(neu_eval, variant="neu", **_kw)

        logger.info("=" * 60)
        logger.info(
            f"✅ {factor_name} 完成 | "
            f"cleaned: dir={cleaned_res['direction']:+d} "
            f"IC{primary_ic_horizon}d={cleaned_res['ic_summary'].loc[f'{primary_ic_horizon}d','ic_mean']:+.4f} "
            f"mono={cleaned_res['monotonicity']:+.3f}  ||  "
            f"neu: dir={neu_res['direction']:+d} "
            f"IC{primary_ic_horizon}d={neu_res['ic_summary'].loc[f'{primary_ic_horizon}d','ic_mean']:+.4f} "
            f"mono={neu_res['monotonicity']:+.3f}"
        )
        logger.info("=" * 60)

        # 顶层返回 = 中性化（生产）版；附 cleaned 对照
        return {
            "success": True,
            "direction": neu_res["direction"],
            "ic_summary": neu_res["ic_summary"],
            "layered_summary": neu_res["layered_summary"],
            "monotonicity": neu_res["monotonicity"],
            "neutralized_factor": neu_eval,
            "cleaned_factor": cleaned_eval,
            "cleaned": cleaned_res,
            "neutralized": neu_res,
        }

    except Exception as e:
        logger.exception(f"[{factor_name}] 评估失败: {e}")
        return {
            "success": False,
            "error": str(e),
        }
