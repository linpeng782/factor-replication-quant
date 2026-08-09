"""
ml_core.run_rolling_batch —— 批量滚动训练（逐年回退，并行启动）
============================================================
逐年回退训练：year 从大到小，每个 year 训练 year-years_back~year，预测 year+1。
2 路并行（每路 64 线程，128 核机器刚好不超卖），跑完一批接下一批。

每个 year 用独立的临时 config（避免并行写冲突），跑完自动清理。

用法：python -m ml_core.run_rolling_batch
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from loguru import logger

REPO = Path(__file__).parent.parent
VENV_PYTHON = "/nfs/ofs-prediction/peterzhenglinpeng-code/peterdidi/bin/python"
BASE_CONFIG = Path(__file__).parent / "rolling_config.yaml"
LOG_DIR = Path(__file__).parent / "logs"


def _run_one(year: int) -> int:
    """为指定 year 生成独立 config，启动一个 run_rolling 子进程（用 ML_CORE_ROLLING_CONFIG 环境变量指定）。"""
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    cfg["year"] = year
    cfg["run_id"] = f"lgbm_rolling_{year}"
    cfg["split_mode"] = "stock"

    # 独立临时 config 文件（避免并行写冲突）
    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_rolling_{year}.yaml", delete=False, mode="w", encoding="utf-8",
        dir=str(REPO / "ml_core"),
    )
    tmp.write(yaml.dump(cfg, allow_unicode=True, default_flow_style=False, sort_keys=False))
    tmp.close()
    tmp_path = Path(tmp.name)

    logger.info(f"[batch] 启动 year={year} → run_id=lgbm_rolling_{year} | config={tmp_path.name}")
    import os
    ret = subprocess.run(
        [VENV_PYTHON, "-m", "ml_core.run_rolling"],
        cwd=str(REPO),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO),
             "ML_CORE_ROLLING_CONFIG": str(tmp_path)},
    )
    log_path = LOG_DIR / f"batch_{year}.log"
    log_path.write_text(ret.stdout, encoding="utf-8")
    tmp_path.unlink(missing_ok=True)  # 清理临时 config
    if ret.returncode != 0:
        logger.error(f"[batch] year={year} 失败 returncode={ret.returncode}，日志 → {log_path}")
    else:
        logger.success(f"[batch] year={year} 完成，日志 → {log_path}")
    return ret.returncode


def _already_done(year: int) -> bool:
    """检查该 year 的模型 + 信号是否已存在（跳过已完成）。"""
    import config
    model_dir = config.ML_MODELS_DIR / f"lgbm_rolling_{year}"
    sig_dir = config.ML_PREDICTIONS_DIR / f"lgbm_rolling_{year}" / "signals"
    return model_dir.exists() and sig_dir.exists() and any(sig_dir.glob("*.txt"))


def main() -> None:
    all_years = [2025, 2024, 2023, 2022, 2021, 2020, 2019]
    n_parallel = 2
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 跳过已完成的 year
    years = [y for y in all_years if not _already_done(y)]
    skipped = [y for y in all_years if y not in years]
    if skipped:
        logger.info(f"跳过已完成: {skipped}")
    if not years:
        logger.success("全部年份已完成，无需训练")
        return
    logger.info(f"=== 批量滚动训练 {len(years)} 年 | {n_parallel} 路并行 | years={years} ===")
    failed = []
    for i in range(0, len(years), n_parallel):
        batch = years[i:i + n_parallel]
        logger.info(f"--- 第 {i//n_parallel+1} 批: {batch} ---")
        with ProcessPoolExecutor(max_workers=n_parallel) as ex:
            futs = {ex.submit(_run_one, y): y for y in batch}
            for fut in as_completed(futs):
                y = futs[fut]
                if fut.result() != 0:
                    failed.append(y)
    if failed:
        logger.error(f"=== 失败年份: {failed} ===")
        sys.exit(1)
    logger.success(f"=== 全部 {len(years)} 年完成 ===")


if __name__ == "__main__":
    main()
