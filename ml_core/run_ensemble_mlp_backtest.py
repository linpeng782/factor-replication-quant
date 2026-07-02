"""批量回测 MLP+LGBM 集成方案"""
import subprocess, os, yaml
from pathlib import Path

BT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/daily-realtime-backtest-pipeline")
PYTHON = "/nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/python"

SCHEMES = [
    "ensemble_mlp_a_50_50",
    "ensemble_mlp_a_60_40",
    "ensemble_mlp_a_70_30",
    "ensemble_mlp_b_50_50",
    "ensemble_mlp_b_60_40",
    "ensemble_mlp_b_70_30",
    "ensemble_mlp_a_b_50_25_25",
    "ensemble_mlp_a_b_40_30_30",
]

base = yaml.safe_load((BT_DIR / "config" / "config.yaml").read_text())
base["mysql_config"]["enabled"] = False
base["trade_params"]["rebalance_interval"] = 5

for name in SCHEMES:
    base["basic_params"]["signal_dir"] = f"/nfs/ofs-prediction/peterzhenglinpeng/ml/predictions/{name}/signals/"
    cfg_path = BT_DIR / "config" / f"config_{name}.yaml"
    cfg_path.write_text(yaml.dump(base, allow_unicode=True, default_flow_style=False, sort_keys=False))
    print(f"=== 回测 {name} ===")
    env = {**os.environ, "BACKTEST_CONFIG_PATH": str(cfg_path)}
    ret = subprocess.run([PYTHON, "batch_runner.py"], cwd=str(BT_DIR),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    lines = ret.stdout.strip().split("\n")
    print("\n".join(lines[-15:]))
    print(f"returncode={ret.returncode}\n")

print("---CMD_DONE---")
