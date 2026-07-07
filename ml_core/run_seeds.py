"""
多 seed 批量训练 —— 以 train_config.yaml 为基底，逐 seed 覆盖 run_id / lgbm.seed 后全流程训练
============================================================
动机：单模型对标签微噪声敏感（早停位置漂移 → top-k 选股换血），多 seed 训练 + rank 平均
集成可互相抵消这类噪声。Stage-1 选因子与 Stage-2 训练都随 seed 变（去相关更强，方案 B）。

用法：改下方 SEEDS / MAX_PARALLEL → python -m ml_core.run_seeds
产物：每 seed 一套 ML_MODELS_DIR/<run_id>/ + ML_PREDICTIONS_DIR/<run_id>/（pred_panel_live + signals）
之后跑 ml_core.ensemble_seeds 做 rank 平均集成。
"""
from __future__ import annotations

import copy
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

from loguru import logger

# ── 手动参数（按需修改）──
SEEDS = [43, 44, 45, 46]          # 42 已有：lgbm_a158_p27_shap_dqlabels（Stage-1/2 全 42，等价成员，直接复用）
MAX_PARALLEL = 2                  # 并行进程数（每进程 LGBM 64 线程，2×64=128 核吃满）
RUN_ID_TMPL = "lgbm_a158_p27_shap_dq_s{seed}"

LOG_DIR = Path(__file__).parent / "logs"


def _train_one(seed: int) -> str:
    """单 seed 全流程：覆盖配置 → 训练 → 推理 → 导信号（子进程内执行）。"""
    from ml_core.run import _run, load_config

    c = copy.deepcopy(load_config())
    run_id = RUN_ID_TMPL.format(seed=seed)
    c["run_id"] = run_id
    c.setdefault("lgbm", {})["seed"] = seed

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sink = logger.add(LOG_DIR / f"{run_id}_{ts}.log", level="INFO",
                      format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}")
    logger.info(f"[seeds] === seed={seed} run_id={run_id} 开始 ===")
    try:
        _run(c)
        logger.success(f"[seeds] === seed={seed} 完成 ===")
        return run_id
    finally:
        logger.remove(sink)


def main() -> None:
    logger.info(f"[seeds] 批量训练 seeds={SEEDS} | 并行={MAX_PARALLEL}")
    ctx = get_context("spawn")    # spawn：子进程干净导入，避免 fork 带出父进程线程/日志状态
    with ctx.Pool(MAX_PARALLEL) as pool:
        done = pool.map(_train_one, SEEDS)
    logger.success(f"[seeds] 全部完成：{done}")


if __name__ == "__main__":
    main()
