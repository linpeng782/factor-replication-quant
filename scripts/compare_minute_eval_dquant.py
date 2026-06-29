"""
paper_27 分钟因子下游评估对比：dquant vs rq
============================================================
对每个因子，分别用 rq raw(factors/raw) 与 dquant raw(factors/raw-dquant) 跑
core.evaluation.evaluate_single_factor，并排比较「选股效果」指标：
  - IC5d 均值 / ICIR（neu 生产版）
  - 分层单调性 monotonicity
  - 多空年化（top 组 - bottom 组 ann_return）

目的：换数据源(rq→dquant)+换复权口径(jy) 后，因子的选股效果是否与 rq 一致。
注意：rq 基线 superset 缓存存在历史陈旧（已查明），但仅影响 ~0.0024% 单元，
对全截面 IC 等聚合指标影响可忽略，故 rq 因子仍是合理的下游参照。

cleaned/neu 因子 parquet 重定向到临时目录（不污染生产）；
但 dquant 侧评估 plot=True，标准 2×2 报告图（evaluation_<range>__{cleaned,neu}.png）
直接落到 output-dquant/<ns>/<factor>/（复用 core.eval_plots.plot_factor_report）。

用法：
  python scripts/compare_minute_eval_dquant.py
  python scripts/compare_minute_eval_dquant.py --factors peak_minute_count ridge_minute_count
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.config as config  # noqa: E402

NS = "kysec/paper_27_microstructure"
RQ_BASE = config._FACTORS / "raw"          # noqa: SLF001
DQ_BASE = config._FACTORS / "raw-dquant"   # noqa: SLF001


def _ls_spread(layered_summary: pd.DataFrame) -> float:
    """多空年化 = 最高组 - 最低组 ann_return（layered 已做 direction 调整，G_g 为强侧）。"""
    grows = [i for i in layered_summary.index if str(i).startswith("G")]
    if len(grows) < 2:
        return float("nan")
    top, bot = grows[-1], grows[0]
    return float(layered_summary.loc[top, "ann_return"] - layered_summary.loc[bot, "ann_return"])


def _install_mask_cache():
    """各因子共享同一列网格，mask/行业/市值/labels 加载昂贵且重复 → 进程内缓存一次。
    仅替换 core.evaluation 命名空间内的引用，不改生产代码。"""
    import core.evaluation as ev

    _orig_masks = ev.load_filter_masks
    _mask_cache = {}

    def _cached_masks(combo_mask_path, new_stock_mask_path, reindex_columns):
        key = tuple(reindex_columns)
        if key not in _mask_cache:
            _mask_cache[key] = _orig_masks(
                combo_mask_path=combo_mask_path,
                new_stock_mask_path=new_stock_mask_path,
                reindex_columns=reindex_columns,
            )
        return _mask_cache[key]

    ev.load_filter_masks = _cached_masks

    _orig_rp = ev.pd.read_parquet
    _pq_cache = {}

    def _cached_rp(path, *args, **kwargs):
        # 仅缓存无额外参数的整表读取（行业/市值/labels 面板），其余透传
        if not args and not kwargs:
            sp = str(path)
            if sp not in _pq_cache:
                _pq_cache[sp] = _orig_rp(path)
            return _pq_cache[sp].copy()
        return _orig_rp(path, *args, **kwargs)

    ev.pd.read_parquet = _cached_rp


def _eval_one(factor_name, factor_df, spec_yaml, out_dir, eval_start, eval_end, primary_h, plot=False):
    from core.evaluation import evaluate_single_factor
    res = evaluate_single_factor(
        factor_name=factor_name, factor_df=factor_df,
        start_date=eval_start, end_date=eval_end, spec_yaml=spec_yaml,
        namespace=NS, output_dir=out_dir, plot=plot,
    )
    if not res.get("success"):
        return None
    neu = res["neutralized"]; cln = res["cleaned"]
    hi = f"{primary_h}d"
    icn = neu["ic_summary"]; icc = cln["ic_summary"]
    hkey = hi if hi in icn.index else icn.index[0]
    return {
        "neu_ic": float(icn.loc[hkey, "ic_mean"]),
        "neu_icir": float(icn.loc[hkey, "icir"]),
        "neu_mono": float(neu["monotonicity"]),
        "neu_ls": _ls_spread(neu["layered_summary"]),
        "cln_ic": float(icc.loc[hkey, "ic_mean"]),
        "cln_icir": float(icc.loc[hkey, "icir"]),
        "hkey": hkey,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", nargs="*", default=None, help="指定因子裸名；默认全 23")
    ap.add_argument("--eval-start", default=config.DEFAULT_EVAL_START_DATE)
    ap.add_argument("--eval-end", default=config.DEFAULT_EVAL_END_DATE)
    a = ap.parse_args()

    # 重定向 cleaned/neu 因子 parquet 到临时目录，避免污染生产
    tmp = Path(tempfile.mkdtemp(prefix="eval_cmp_"))
    config.CLEANED_FACTOR_BASE = tmp / "cleaned"
    config.NEU_FACTOR_BASE = tmp / "neu"
    config.OUTPUT_DIR = tmp / "output"
    # dquant 侧标准报告图落地目录（真实 output-dquant）
    out_dquant = Path(__file__).resolve().parent.parent / "output-dquant" / NS
    logger.info(f"临时产物目录（用后即弃）: {tmp}")
    logger.info(f"dquant 报告图落地: {out_dquant}")

    _install_mask_cache()  # mask/行业/市值/labels 进程内缓存（各因子列网格一致）

    names = a.factors or sorted(p.stem for p in (DQ_BASE / NS).glob("*.parquet"))
    logger.info(f"评估对比 {len(names)} 因子 | 窗口 {a.eval_start}~{a.eval_end}")

    from core.spec_generator import load_spec_yaml
    rows = []
    for n in names:
        rq_p = RQ_BASE / NS / f"{n}.parquet"; dq_p = DQ_BASE / NS / f"{n}.parquet"
        if not (rq_p.exists() and dq_p.exists()):
            logger.warning(f"{n}: 缺 rq 或 dquant raw，跳过"); continue
        try:
            spec = load_spec_yaml(f"{NS}/{n}")
        except Exception:
            spec = None
        primary_h = (spec or {}).get("evaluation", {}).get("primary_horizon", 5)
        rq_df = pd.read_parquet(rq_p); dq_df = pd.read_parquet(dq_p)
        rq_df.index = pd.to_datetime(rq_df.index); dq_df.index = pd.to_datetime(dq_df.index)
        logger.info(f"  评估 {n} (rq) ...")
        r = _eval_one(n, rq_df, spec, tmp / "o_rq" / n, a.eval_start, a.eval_end, primary_h, plot=False)
        logger.info(f"  评估 {n} (dquant，出标准报告图 → output-dquant) ...")
        d = _eval_one(n, dq_df, spec, out_dquant / n, a.eval_start, a.eval_end, primary_h, plot=True)
        if r is None or d is None:
            logger.warning(f"{n}: 评估失败，跳过"); continue
        rows.append((n, r, d))

    # ── 汇总表（neu 生产版为主）──
    logger.info("=" * 122)
    logger.info(
        f"{'因子':<32}{'IC(rq)':>9}{'IC(dq)':>9}{'ΔIC':>9}"
        f"{'ICIR(rq)':>10}{'ICIR(dq)':>10}{'mono(rq)':>10}{'mono(dq)':>10}{'LS(rq)':>9}{'LS(dq)':>9}"
    )
    logger.info("-" * 122)
    dic = []
    for n, r, d in rows:
        dic.append(abs(d["neu_ic"] - r["neu_ic"]))
        logger.info(
            f"{n:<32}{r['neu_ic']:>9.4f}{d['neu_ic']:>9.4f}{d['neu_ic']-r['neu_ic']:>+9.4f}"
            f"{r['neu_icir']:>10.3f}{d['neu_icir']:>10.3f}"
            f"{r['neu_mono']:>10.3f}{d['neu_mono']:>10.3f}"
            f"{r['neu_ls']:>9.3f}{d['neu_ls']:>9.3f}"
        )
    logger.info("-" * 122)
    if dic:
        logger.info(f"neu IC 绝对差: 中位={np.median(dic):.5f}  最大={np.max(dic):.5f}  均值={np.mean(dic):.5f}")
    logger.success(f"完成 {len(rows)} 因子下游评估对比（临时产物在 {tmp}，可删）")


if __name__ == "__main__":
    main()
