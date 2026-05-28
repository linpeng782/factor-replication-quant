"""
Factor Inventory：所有 panel 因子在统一配置下的"评估总账"。

每次跑会在 factor_inventory/<时间戳>/ 下产出：
  inventory.parquet    197 行 × 30+ 列的核心表
  inventory.csv        human-readable 镜像
  run_meta.json        本次 refresh 的元信息
  plots/<producer>/<factor>.png   每因子一张评估图

并维护 factor_inventory/latest 软链接指向最新的时间戳目录。

底层用 alpha-engine 的 evaluate_all（ProcessPool 并行 + 共享 forward_returns；PIT canonical）。

用法：
    python scripts/build_factor_inventory.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

# 把 alpha-engine 加进 sys.path，import 它的 evaluate_all
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALPHA_ENGINE_ROOT = Path("/nfs/volume-1593-1/peterzhenglinpeng/my-alpha-engine")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ALPHA_ENGINE_ROOT))

# 必须在 sys.path 设好之后再 import alpha-engine 的东西
from alpha_engine import config as engine_config  # noqa: E402
from alpha_engine.analysis.evaluation.pipeline import evaluate_all  # noqa: E402

# ── 参数（手改） ─────────────────────────────────────────────
START = "2016-01-01"
END = "2025-12-31"
IC_HORIZONS = (2, 5, 10, 20)
PRIMARY_IC_HORIZON = 5
LAYER_GROUPS = 5
LAYER_REBALANCE = 5

# 跳过 mars（用户判断质量较低）；只评估 alpha158 + spec
PRODUCERS = ("alpha158", "spec")

N_WORKERS = 100  # 并行 worker 数（128C 机器；留 28 核给系统 + alpha-engine 内部 fork）

INVENTORY_ROOT = PROJECT_ROOT / "factor_inventory"


def _list_factors(producer: str) -> list[str]:
    """扫 cleaned-factor-panel/<producer>/*.parquet 拿到因子名列表。"""
    panel_dir = engine_config.CLEANED_FACTOR_PANEL_DIR / producer
    if not panel_dir.exists():
        logger.warning(f"  {panel_dir} 不存在，跳过")
        return []
    return sorted(p.stem for p in panel_dir.glob("*.parquet"))


def _load_spec_origins() -> dict[str, tuple[str, str]]:
    """扫 sources/<pub>/<group>/specs/<factor>/spec.yaml 建 factor → (pub, group)。

    路径形如 sources/kysec/paper_27_microstructure/specs/peak_minute_count/spec.yaml
    → factor='peak_minute_count', pub='kysec', group='paper_27_microstructure'
    """
    sources = PROJECT_ROOT / "sources"
    out: dict[str, tuple[str, str]] = {}
    for spec_yaml in sources.glob("*/*/specs/*/spec.yaml"):
        parts = spec_yaml.parts
        # 倒数 5..1: pub, group, 'specs', factor, 'spec.yaml'
        pub, group, _, factor = parts[-5], parts[-4], parts[-3], parts[-2]
        if factor in out and out[factor] != (pub, group):
            logger.warning(f"  spec 因子重名: {factor} 在 {out[factor]} 与 ({pub},{group})")
        out[factor] = (pub, group)
    logger.info(f"  spec origins: {len(out)} 个 spec 因子的 (pub, group) 已索引")
    return out


def _compute_missing_rates(
    factor_names: list[str],
    factor_to_producer: dict[str, str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """对每个因子在 [start, end] 评估窗内算 overall + listed 两个缺失率。

    - missing_rate_overall : 总 NaN cell 数 / 总 cell 数（混合"未上市"+"算法缺失"）
    - missing_rate_listed  : 真业务缺失。用 vwap_panel.notna() 当上市 mask，
                             仅在"已上市 (date, stock)"对里算 NaN 占比。
                             这才是诊断算法健康度的关键指标——不受 panel 时间窗稀释。

    数据源：cleaned-factor-panel（与 evaluate_all 看到的一致），不是 raw factor-panel。
    """
    logger.info(f"[missing_rate] 加载 vwap_panel 作上市 mask: {engine_config.VWAP_PANEL_PATH}")
    vwap = pd.read_parquet(engine_config.VWAP_PANEL_PATH)
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    vwap = vwap.loc[(vwap.index >= start_ts) & (vwap.index <= end_ts)]

    rows: dict[str, tuple[float, float]] = {}
    t0 = time.time()
    for i, fname in enumerate(factor_names):
        producer = factor_to_producer[fname]
        path = engine_config.CLEANED_FACTOR_PANEL_DIR / producer / f"{fname}.parquet"
        if not path.exists():
            rows[fname] = (float("nan"), float("nan"))
            continue
        df = pd.read_parquet(path)
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        if df.empty:
            rows[fname] = (float("nan"), float("nan"))
            continue

        listed = vwap.reindex(index=df.index, columns=df.columns).notna()

        total = df.size
        na = int(df.isna().sum().sum())
        listed_cells = int(listed.sum().sum())
        na_in_listed = int((listed & df.isna()).sum().sum())

        overall = na / total if total else float("nan")
        listed_rate = na_in_listed / listed_cells if listed_cells else float("nan")
        rows[fname] = (overall, listed_rate)

        del df, listed

    t = time.time() - t0
    logger.info(f"[missing_rate] {len(factor_names)} 个因子完成，耗时 {t:.1f}s")
    return pd.DataFrame.from_dict(
        rows,
        orient="index",
        columns=["missing_rate_overall", "missing_rate_listed"],
    ).rename_axis("factor")


def _build_inventory(
    ic_summary: pd.DataFrame,
    layered_summary: pd.DataFrame,
    missing_rates: pd.DataFrame,
    factor_to_producer: dict[str, str],
    spec_origins: dict[str, tuple[str, str]],
    eval_ts: str,
) -> pd.DataFrame:
    """
    把 evaluate_all 产出的 ic_summary + layered_summary + 缺失率合并成 inventory。

    ic_summary 列示例：direction, ic_mean_2d, ic_std_2d, icir_2d, ic_t_2d,
                      pct_positive_2d, ... (4 horizons)
    layered_summary 列示例：direction, monotonicity, ann_return_G1..G5,
                          sharpe_LongShort, ... (5 groups + LongShort)
    missing_rates 列：missing_rate_overall, missing_rate_listed
    """
    ic = ic_summary.set_index("factor")
    layered = layered_summary.set_index("factor")

    # 选取 layered 关键列（避免太宽）
    layered_keep = [
        "monotonicity",
        "ann_return_G1",
        "ann_return_G5",
        "ann_return_LongShort",
        "ann_vol_LongShort",
        "sharpe_LongShort",
        "mean_turnover_LongShort",
    ]
    layered_keep = [c for c in layered_keep if c in layered.columns]
    layered_sub = layered[layered_keep]

    # join：direction 在 ic 里已经有了，layered 里的 direction 跟 ic 一致，去重
    inv = ic.join(layered_sub, how="left").join(missing_rates, how="left")

    # 加元数据列
    inv["producer"] = inv.index.map(factor_to_producer)
    # pub / group：spec 因子从 sources/<pub>/<group>/specs/ 路径推断；非 spec 行为 NaN
    inv["pub"] = inv.index.map(lambda f: spec_origins.get(f, (None, None))[0])
    inv["group"] = inv.index.map(lambda f: spec_origins.get(f, (None, None))[1])
    inv["time_window_start"] = START
    inv["time_window_end"] = END
    inv["eval_timestamp"] = eval_ts

    # 列重排：身份 → 来源 → 缺失率 → IC → layered → meta
    # 缺失率放前面是因为它常作为"先筛掉低质量因子"的第一道闸
    id_cols = ["producer", "pub", "group", "direction"]
    missing_cols = ["missing_rate_overall", "missing_rate_listed"]
    ic_cols = sorted([c for c in inv.columns if c.startswith(("ic_mean", "ic_std", "icir", "ic_t", "pct_positive", "n_days_"))])
    layered_cols = [c for c in inv.columns if c in layered_keep]
    meta_cols = ["time_window_start", "time_window_end", "eval_timestamp"]

    ordered = id_cols + missing_cols + ic_cols + layered_cols + meta_cols
    inv = inv[[c for c in ordered if c in inv.columns]]

    return inv


def main():
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = INVENTORY_ROOT / eval_ts
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    logger.info("=" * 72)
    logger.info(f"Factor Inventory 构建：{eval_ts}")
    logger.info(f"  时间窗:    {START} ~ {END}")
    logger.info(f"  horizons:  {IC_HORIZONS}, primary={PRIMARY_IC_HORIZON}")
    logger.info(f"  layered:   groups={LAYER_GROUPS}, rebalance={LAYER_REBALANCE}")
    logger.info(f"  producers: {PRODUCERS}（跳过 mars）")
    logger.info(f"  workers:   {N_WORKERS}")
    logger.info(f"  output:    {out_dir}")
    logger.info("=" * 72)

    # 1. 收集因子列表
    factor_to_producer: dict[str, str] = {}
    factor_list: list[str] = []
    for prod in PRODUCERS:
        names = _list_factors(prod)
        logger.info(f"  {prod}: {len(names)} 个因子")
        for n in names:
            if n in factor_to_producer:
                logger.warning(f"重名因子: {n} 在 {factor_to_producer[n]} 与 {prod}")
            factor_to_producer[n] = prod
            factor_list.append(n)
    logger.info(f"  合计: {len(factor_list)} 个因子待评估\n")

    # 2. 跑 evaluate_all（PIT canonical labels 源 + ProcessPool 并行）
    t0 = time.time()
    # 让 plots 按 producer 分子目录：先扁平产出到 plots_dir/，跑完再 mv 到子目录
    flat_plots_dir = plots_dir
    (flat_plots_dir / "plots").mkdir(parents=True, exist_ok=True)  # evaluate_all 会创建 plots/ 子目录

    result = evaluate_all(
        factor_names=factor_list,
        start=START,
        end=END,
        output_dir=flat_plots_dir,
        ic_horizons=IC_HORIZONS,
        primary_ic_horizon=PRIMARY_IC_HORIZON,
        layer_rebalance=LAYER_REBALANCE,
        layer_groups=LAYER_GROUPS,
        n_workers=N_WORKERS,
        plot=True,
    )
    t_eval = time.time() - t0
    logger.info(f"\n[evaluate_all] 耗时 {t_eval:.1f}s")

    # 3. 读 evaluate_all 落盘的 csv，整理成 inventory
    ic_csv = flat_plots_dir / "ic_summary.csv"
    layered_csv = flat_plots_dir / "layered_summary.csv"
    if not ic_csv.exists() or not layered_csv.exists():
        logger.error(f"evaluate_all 没产出 ic_summary.csv 或 layered_summary.csv")
        sys.exit(1)

    ic_summary = pd.read_csv(ic_csv)
    layered_summary = pd.read_csv(layered_csv)

    # 3.5 算缺失率（overall + listed）
    missing_rates = _compute_missing_rates(factor_list, factor_to_producer, START, END)

    # 3.6 索引 spec 因子来源（pub, group），从 sources/<pub>/<group>/specs/<factor>/ 推断
    spec_origins = _load_spec_origins()

    inventory = _build_inventory(
        ic_summary, layered_summary, missing_rates, factor_to_producer, spec_origins, eval_ts
    )

    # 4. 写 inventory.parquet + inventory.csv
    inventory.to_parquet(out_dir / "inventory.parquet")
    # CSV 用合适的浮点格式
    inventory.to_csv(out_dir / "inventory.csv", float_format="%.6f")
    logger.info(f"\n  inventory shape: {inventory.shape}")
    logger.info(f"  → {out_dir / 'inventory.parquet'}")
    logger.info(f"  → {out_dir / 'inventory.csv'}")

    # 5. 把 evaluate_all 扁平产出的 plots/*.png 按 producer 分子目录
    flat_plot_files = list((flat_plots_dir / "plots").glob("*.png"))
    if flat_plot_files:
        for prod in PRODUCERS:
            (plots_dir / prod).mkdir(parents=True, exist_ok=True)
        for png in flat_plot_files:
            prod = factor_to_producer.get(png.stem)
            if prod is None:
                logger.warning(f"  unknown producer for {png.stem}, leaving in flat")
                continue
            png.rename(plots_dir / prod / png.name)
        # 删空的 plots/ 子目录
        try:
            (flat_plots_dir / "plots").rmdir()
        except OSError:
            pass

    # 把 csv 也搬出去（avoiding duplication）
    for csv_name in ("ic_summary.csv", "layered_summary.csv"):
        csv_path = flat_plots_dir / csv_name
        if csv_path.exists():
            csv_path.unlink()

    # 6. 写 run_meta.json
    # evaluate_all 返回 dict 但 key 名不固定；统一从 inventory 行数推断成功数
    n_success = len(inventory)
    meta = {
        "eval_timestamp": eval_ts,
        "time_window": {"start": START, "end": END},
        "ic_horizons": list(IC_HORIZONS),
        "primary_ic_horizon": PRIMARY_IC_HORIZON,
        "layer_groups": LAYER_GROUPS,
        "layer_rebalance": LAYER_REBALANCE,
        "producers": list(PRODUCERS),
        "n_factors": len(factor_list),
        "n_succeeded": n_success,
        "n_failed": len(factor_list) - n_success,
        "eval_seconds": round(t_eval, 1),
        "n_workers": N_WORKERS,
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 7. 维护 latest 软链接
    latest = INVENTORY_ROOT / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(eval_ts, target_is_directory=True)
    logger.info(f"\n  latest → {eval_ts}")

    # 8. 打印 top 20 by |icir_5d|
    if "icir_5d" in inventory.columns:
        top = inventory.reindex(inventory["icir_5d"].abs().sort_values(ascending=False).index).head(20)
        logger.info(f"\n=== Top 20 by |icir_5d| ===")
        print(top[["producer", "direction", "ic_mean_5d", "icir_5d", "ic_mean_20d", "icir_20d", "monotonicity", "sharpe_LongShort"]].to_string())

    logger.info("\n✅ inventory 构建完成")


if __name__ == "__main__":
    main()
