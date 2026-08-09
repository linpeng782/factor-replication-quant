#!/usr/bin/env python
"""
东吴因子逐因子增量实验 —— 并行编排器
============================================================
从 ablate124（124因子基线，不含东吴）出发，每次只追加1个东吴因子，
4个实验并行跑（训练+推理+回测），最后汇总对比。

每个实验内部：训练(ml_core.run) → 回测(batch_runner.py) 串行
4个实验之间：并行（ProcessPoolExecutor）

用法：
    python -m ml_core.run_dw_incremental
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ==================== 路径常量 ====================
REPO = Path("/nfs/ofs-prediction/peterzhenglinpeng-code/factor-replication-quant-new")
BT_REPO = Path("/nfs/ofs-prediction/peterzhenglinpeng-code/daily-realtime-backtest-pipeline")
VENV_PY = "/nfs/ofs-prediction/peterzhenglinpeng-code/peterdidi/bin/python"
DATA_ROOT = "/nfs/ofs-prediction/peterzhenglinpeng"
TMP_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/tmp/dw_exp")

# ==================== 实验定义 ====================
BASELINE_RUN_ID = "lgbm_shap107_no_p27_csrank5_dq"  # 107因子基线（无东吴无p27）
BASE_SOURCES = [
    "alpha158-dquant",
    "style-dquant/size",
    "kysec-dquant/paper_67_long_momentum",
    "htsec/paper_04_momentum",
]
NUM_THREADS = 32  # 4×32=128核

# 4个东吴因子 → 各自的 source
DW_SOURCE_MAP = {
    "turn20":     "dongwu/paper_07_stable_turnover",
    "str20":      "dongwu/paper_07_stable_turnover",
    "pct_turn20": "dongwu/paper_07_stable_turnover",
    "gtr20":      "dongwu/paper_13_gtr",
}

EXPERIMENTS = [
    {"factor": "turn20",     "run_id": "lgbm_shap108_nop27_add_turn20_csrank5_dq"},
    {"factor": "str20",      "run_id": "lgbm_shap108_nop27_add_str20_csrank5_dq"},
    {"factor": "pct_turn20", "run_id": "lgbm_shap108_nop27_add_pct_turn20_csrank5_dq"},
    {"factor": "gtr20",      "run_id": "lgbm_shap108_nop27_add_gtr20_csrank5_dq"},
]


# ==================== 生成临时 config ====================
def _load_base_train_config() -> dict:
    """读当前 train_config.yaml 作为模板"""
    with open(REPO / "ml_core" / "train_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_base_bt_config() -> dict:
    """读回测 config.yaml 作为模板"""
    with open(BT_REPO / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def gen_train_config(exp: dict, base: dict) -> Path:
    """为单个实验生成 train_config yaml，返回文件路径"""
    cfg = copy.deepcopy(base)
    factor = exp["factor"]
    run_id = exp["run_id"]
    dw_source = DW_SOURCE_MAP[factor]

    cfg["run_id"] = run_id
    # sources = 基础5个 + 该因子对应的东吴源
    cfg["sources"] = BASE_SOURCES + [dw_source]
    # feature_set：从 ablate124 继承124个因子，只追加这一个东吴因子
    cfg["feature_set"] = {
        "inherit_from": BASELINE_RUN_ID,
        "append": [factor],
    }
    # 并行跑时降低线程数
    cfg.setdefault("lgbm", {})["num_threads"] = NUM_THREADS
    # select_method 必须为 null（固定因子集与选因子互斥）
    cfg["select_method"] = None

    path = TMP_DIR / f"train_config_{factor}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return path


def gen_bt_config(exp: dict, base: dict) -> Path:
    """为单个实验生成回测 config yaml（只改 signal_dir），返回文件路径"""
    cfg = copy.deepcopy(base)
    run_id = exp["run_id"]
    cfg["basic_params"]["signal_dir"] = f"${{DATA_ROOT}}/ml/predictions/{run_id}/signals"
    # 并行跑关闭 MySQL
    cfg["mysql_config"]["enabled"] = False

    path = TMP_DIR / f"bt_config_{exp['factor']}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return path


# ==================== 单个实验执行 ====================
def run_one_experiment(exp: dict, train_cfg_path: str, bt_cfg_path: str) -> dict:
    """串行跑一个实验：训练+推理+导信号 → 回测。返回结果 dict。"""
    factor = exp["factor"]
    run_id = exp["run_id"]
    t0 = time.time()
    result = {"factor": factor, "run_id": run_id, "success": True, "stages": {}}

    # ---- 阶段1：训练 + 推理 + 导信号 ----
    env = os.environ.copy()
    env["ML_CONFIG_PATH"] = train_cfg_path
    train_cmd = [VENV_PY, "-u", "-m", "ml_core.run"]
    proc = subprocess.run(
        train_cmd, cwd=str(REPO), env=env,
        capture_output=True, text=True, timeout=3600,
    )
    result["stages"]["train"] = {
        "returncode": proc.returncode,
        "tail": (proc.stderr or proc.stdout or "")[-2000:],
    }
    if proc.returncode != 0:
        result["success"] = False
        result["elapsed"] = time.time() - t0
        return result

    # ---- 阶段2：回测 ----
    env_bt = os.environ.copy()
    env_bt["BACKTEST_CONFIG_PATH"] = bt_cfg_path
    bt_cmd = [VENV_PY, "-u", "batch_runner.py"]
    proc_bt = subprocess.run(
        bt_cmd, cwd=str(BT_REPO), env=env_bt,
        capture_output=True, text=True, timeout=1800,
    )
    result["stages"]["backtest"] = {
        "returncode": proc_bt.returncode,
        "tail": (proc_bt.stderr or proc_bt.stdout or "")[-2000:],
    }
    if proc_bt.returncode != 0:
        result["success"] = False

    result["elapsed"] = time.time() - t0
    return result


# ==================== 主流程 ====================
def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 生成所有临时 config
    print("[orchestrator] 生成临时配置文件...")
    base_train = _load_base_train_config()
    base_bt = _load_base_bt_config()

    configs = {}
    for exp in EXPERIMENTS:
        tcfg = gen_train_config(exp, base_train)
        bcfg = gen_bt_config(exp, base_bt)
        configs[exp["factor"]] = (str(tcfg), str(bcfg))
        print(f"  {exp['factor']:12s} train={tcfg.name}  bt={bcfg.name}")

    # 2. 并行跑4个实验
    print(f"\n[orchestrator] 并行启动 {len(EXPERIMENTS)} 个实验（每个：训练→回测）...")
    t_start = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=len(EXPERIMENTS)) as ex:
        futs = {}
        for exp in EXPERIMENTS:
            tcfg, bcfg = configs[exp["factor"]]
            f = ex.submit(run_one_experiment, exp, tcfg, bcfg)
            futs[f] = exp

        for fut in as_completed(futs):
            exp = futs[fut]
            res = fut.result()
            results.append(res)
            status = "✓" if res["success"] else "✗"
            elapsed = res.get("elapsed", 0)
            print(f"  [{status}] {exp['factor']:12s}  耗时 {elapsed:.0f}s")
            if not res["success"]:
                for stage, info in res["stages"].items():
                    if info["returncode"] != 0:
                        print(f"      {stage} 失败 (rc={info['returncode']})")
                        print(f"      输出尾部:\n{info['tail'][-800:]}")

    total_elapsed = time.time() - t_start
    print(f"\n[orchestrator] 全部完成，总耗时 {total_elapsed:.0f}s")

    # 3. 写结果摘要
    summary_path = TMP_DIR / "orchestrator_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_elapsed": total_elapsed,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[orchestrator] 结果摘要 → {summary_path}")

    # 4. 打印最终状态
    ok = sum(1 for r in results if r["success"])
    print(f"\n[orchestrator] 成功 {ok}/{len(results)}")
    if ok < len(results):
        print(f"[orchestrator] 失败的实验: {[r['factor'] for r in results if not r['success']]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
