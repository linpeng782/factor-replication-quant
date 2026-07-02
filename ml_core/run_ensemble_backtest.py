"""批量回测 4 个集成方案，用与已有回测完全相同的参数（interval5/vwapam/shift1/netting/topk100）"""
import subprocess, os, yaml
from pathlib import Path

BT_DIR = Path("/nfs/volume-1593-1/peterzhenglinpeng/daily-realtime-backtest-pipeline")
TEMPLATE = BT_DIR / "config" / "config.yaml"
PYTHON = "/nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/python"

SCHEMES = [
    "ensemble_quota_50_50",
    "ensemble_quota_60_40",
    "ensemble_quota_70_30",
    "ensemble_quota_40_60",
]

# 基础配置模板（从 config.yaml 读，只改 signal_dir + mysql enabled=false）
base = yaml.safe_load(TEMPLATE.read_text())
base["mysql_config"]["enabled"] = False
# 确保与已有回测一致：interval5
base["trade_params"]["rebalance_interval"] = 5

for name in SCHEMES:
    base["basic_params"]["signal_dir"] = f"/nfs/ofs-prediction/peterzhenglinpeng/ml/predictions/{name}/signals/"
    cfg_path = BT_DIR / "config" / f"config_{name}.yaml"
    cfg_path.write_text(yaml.dump(base, allow_unicode=True, default_flow_style=False, sort_keys=False))
    print(f"=== 回测 {name} ===")
    env = {**os.environ, "BACKTEST_CONFIG_PATH": str(cfg_path)}
    ret = subprocess.run([PYTHON, "batch_runner.py"], cwd=str(BT_DIR),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    # 只打印最后 20 行
    lines = ret.stdout.strip().split("\n")
    print("\n".join(lines[-20:]))
    print(f"returncode={ret.returncode}\n")

print("---CMD_DONE---")
