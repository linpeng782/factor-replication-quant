"""
回归 baseline：replication 侧（spec engine pipeline）

抽 alpha-shared 共享库前的"金标"快照。
跑 evaluate_single_factor 对 2 个 spec 因子，把 IC summary + 分层 summary +
cleaned_factor 的标量摘要保存成 parquet。
抽完后用同样脚本再跑一次，与 baseline 逐数值比对（容差 0）。

覆盖 primitive：load_filter_masks / prepare_factor / build_forward_returns
            / compute_ic_series / compute_ic_report / layered_backtest
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import config
from core.evaluation import evaluate_single_factor

OUT_DIR = PROJECT_ROOT / "scripts" / "regression_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FACTORS = ["peak_minute_count", "npf_mrq_sue8"]
START = "20200101"
END = "20241231"
IC_HORIZONS = (2, 5, 10, 20)
PRIMARY_IC_HORIZON = 5
LAYER_GROUPS = 5
LAYER_REBALANCE = 5


def _cleaned_factor_fingerprint(df: pd.DataFrame) -> dict:
    """对清洗后因子取一组对 mask_loader / prepare_factor 改动敏感的标量。"""
    arr = df.values.astype(np.float64)
    finite = arr[np.isfinite(arr)]
    return {
        "shape_T": df.shape[0],
        "shape_N": df.shape[1],
        "n_finite": int(np.isfinite(arr).sum()),
        "sum": float(np.nansum(arr)),
        "sum_sq": float(np.nansum(arr * arr)),
        "min": float(finite.min()) if finite.size else float("nan"),
        "max": float(finite.max()) if finite.size else float("nan"),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "std": float(finite.std(ddof=0)) if finite.size else float("nan"),
    }


def main():
    logger.info("=" * 72)
    logger.info("回归 baseline (replication): 5 个 primitive 全覆盖")
    logger.info(f"  factors={FACTORS}  window={START}~{END}")
    logger.info("=" * 72)

    ic_rows = []
    layered_rows = []
    fingerprint_rows = []

    for factor_name in FACTORS:
        logger.info(f"\n--- {factor_name} ---")
        raw_path = config.RAW_FACTOR_DIR / f"{factor_name}.parquet"
        factor_df = pd.read_parquet(raw_path)

        result = evaluate_single_factor(
            factor_name=factor_name,
            factor_df=factor_df,
            start_date=START,
            end_date=END,
            ic_horizons=IC_HORIZONS,
            primary_ic_horizon=PRIMARY_IC_HORIZON,
            layer_groups=LAYER_GROUPS,
            layer_rebalance=LAYER_REBALANCE,
            output_dir=OUT_DIR / "_eval_artifacts" / factor_name,
            plot=False,
            skip_cleaning=False,
        )
        if not result.get("success"):
            logger.error(f"{factor_name} 失败: {result}")
            continue

        # IC summary：列展平成 (factor, horizon×metric)
        ic_summary: pd.DataFrame = result["ic_summary"]
        ic_dict = {"factor": factor_name, "direction": result["direction"]}
        for h in ic_summary.index:
            for col in ic_summary.columns:
                ic_dict[f"{col}_{h}d"] = ic_summary.loc[h, col]
        ic_rows.append(ic_dict)

        # Layered summary：列展平成 (factor, group×metric)
        layered_summary: pd.DataFrame = result["layered_summary"]
        layered_dict = {
            "factor": factor_name,
            "direction": result["direction"],
            "monotonicity": result.get("monotonicity"),
        }
        for grp in layered_summary.index:
            for col in layered_summary.columns:
                layered_dict[f"{col}_{grp}"] = layered_summary.loc[grp, col]
        layered_rows.append(layered_dict)

        # cleaned_factor 指纹
        fp = _cleaned_factor_fingerprint(result["cleaned_factor"])
        fp["factor"] = factor_name
        fingerprint_rows.append(fp)

    ic_df = pd.DataFrame(ic_rows).set_index("factor").sort_index()
    layered_df = pd.DataFrame(layered_rows).set_index("factor").sort_index()
    fp_df = pd.DataFrame(fingerprint_rows).set_index("factor").sort_index()

    ic_df.to_parquet(OUT_DIR / "replication_ic_summary.parquet")
    layered_df.to_parquet(OUT_DIR / "replication_layered_summary.parquet")
    fp_df.to_parquet(OUT_DIR / "replication_cleaned_fingerprint.parquet")

    logger.success(f"\n✅ baseline 写入 {OUT_DIR}")
    logger.info(f"  ic_summary       shape={ic_df.shape}")
    logger.info(f"  layered_summary  shape={layered_df.shape}")
    logger.info(f"  cleaned_fingerprint shape={fp_df.shape}")
    logger.info(f"\nfingerprint preview:\n{fp_df}")


if __name__ == "__main__":
    main()
