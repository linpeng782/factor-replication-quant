# 服务器因子日更生产手册

> 给 SSH 上的 agent / 操作者。目标：与本地 Mac 保持对齐，每个交易日收盘后跑一次日更，产出所有因子面板。
> 关联文档：`daily_update.md`（流程设计）/ `hf_factor_factory_design.md`（工厂架构）。

---

## 0. 服务器环境（固定，不用改）

```bash
# 激活 venv
source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate

# 项目根（以下所有命令在此执行）
cd /nfs/ofs-prediction/peterzhenglinpeng/factor-replication

# 数据根（代码默认值，无需设环境变量）
# FACTOR_REPL_DATA_ROOT=/nfs/ofs-prediction/peterzhenglinpeng
```

| 路径 | 内容 |
|---|---|
| `/nfs/volume-1593-1/peterzhenglinpeng/peterdidi` | Python 3.11 venv |
| `/nfs/ofs-prediction/peterzhenglinpeng/factor-replication` | 本仓代码 |
| `/nfs/ofs-prediction/peterzhenglinpeng/market-data/minute/raw/` | 分钟原始数据（已 scp） |
| `/nfs/ofs-prediction/peterzhenglinpeng/intermediate-cache/` | L2 superset 缓存（prv_v3 + tide_v1 已 scp） |

---

## 1. 首次上线检查清单（只做一次）

### 1a. 确认数据到位

```bash
# 分钟数据末日（应 = 本地 Mac 最新交易日）
ls /nfs/ofs-prediction/peterzhenglinpeng/market-data/minute/raw/ | tail -3

# superset 缓存状态（应见 ✅已建 prv_v3 + tide_v1，❌未建 sm_v1）
PYTHONPATH=. python pipeline/refresh_supersets.py --dry-run
```

### 1b. 安装依赖（如 alpha-shared 未装）

```bash
pip install -e /nfs/volume-1593-1/peterzhenglinpeng/alpha-shared
```

### 1c. 建 sm_v1（首次必做，本地 Mac 内存不够没建）

```bash
# 服务器内存大，调高并发；首次全史构建约 30~60 分钟
MINUTE_CHUNK_DAYS=250 MINUTE_WORKERS=64 \
  PYTHONPATH=. python pipeline/refresh_supersets.py --cache-key sm_v1
```

完成后再跑一次 `--dry-run`，三个 superset 全部显示 ✅已建 即可。

---

## 2. 日常日更（每个交易日收盘后）

```bash
source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate
cd /nfs/ofs-prediction/peterzhenglinpeng/factor-replication

# 服务器参数（内存大，chunk 调大加速）
export MINUTE_WORKERS=64
export MINUTE_CHUNK_DAYS=250

bash pipeline/daily_update.sh
```

`daily_update.sh` 自动完成以下 8 步（任一失败即停）：

| 步骤 | 内容 |
|---|---|
| 1–6 | 数据线：ex_factors / raw_ohlcv / minute_ohlcv / industry / market_cap |
| 7 | `refresh_supersets.py`：增量刷新 prv_v3 + tide_v1 + sm_v1 到最新交易日 |
| 8 | 因子循环：`run.py` 逐个重算所有 spec 因子，失败仅告警不中断 |
| 9 | `ml/labels.py`：labels 回填（失败不阻塞） |

---

## 3. 日更完成后的验证

```bash
# a. 分钟数据末日 = 最新交易日
ls market-data/minute/raw/ | tail -1

# b. prv_v3 superset 已追到最新日（抽查一只股）
python - <<'PY'
import pandas as pd, glob
f = sorted(glob.glob("/nfs/ofs-prediction/peterzhenglinpeng/intermediate-cache/prv_v3__h*/*.parquet"))[0]
print("prv_v3 cache_last:", pd.read_parquet(f, columns=["date"])["date"].max())
PY

# c. 因子面板末日（抽查 tide_full）
python - <<'PY'
import pandas as pd
f = "/nfs/ofs-prediction/peterzhenglinpeng/factors/raw/founder/paper_02_tide/tide_full.parquet"
print("tide_full 末日:", pd.read_parquet(f).index.max())
PY
```

三项均等于最新交易日 → 日更成功。

---

## 4. 与本地 Mac 保持对齐

| 操作 | 命令 |
|---|---|
| 拉最新代码 | `git pull origin feature/operators-upgrade` |
| 新 spec 上线后重跑对应因子 | `PYTHONPATH=. python run.py <factor>` |
| 强制重建某个 superset | `PYTHONPATH=. python pipeline/refresh_supersets.py --cache-key <key> --rebuild` |
| 只刷新 superset 不算因子 | `PYTHONPATH=. python pipeline/refresh_supersets.py` |

---

## 5. 常见问题

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: core` | 确认在仓库根目录执行（`cd .../factor-replication`） |
| `minute/raw 为空` | 检查 scp 路径是否正确（应在 `market-data/minute/raw/`，非 `minute/`） |
| sm_v1 刷新 OOM | 降低 `MINUTE_CHUNK_DAYS=100`，或分批建（待实现 `--rebuild` 续建） |
| `daily_update.sh` 在步骤7卡住 | sm_v1 未建，先单独执行 1c 步骤 |
| 因子末日停在旧日期 | 检查 `END_DATE` 是否正确（脚本从 `minute/raw/` 末日文件名动态推导） |
