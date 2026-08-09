# 日更操作手册（dquant）

> Agent 日更唯一入口。严格 **A 数据线 → B 因子线 → P 推理线**；每步看日志末日；失败即停。
> 数据根：`/nfs/ofs-prediction/peterzhenglinpeng`

---

## 0. 前置（必做）

```bash
source /nfs/ofs-prediction/peterzhenglinpeng-code/peterdidi/bin/activate
cd /nfs/ofs-prediction/peterzhenglinpeng-code/factor-replication-quant-new
# 全程停在项目根；统一用 PYTHONPATH=. ，不要 cd 进子目录
```

```bash
# 0.1 目标交易日（17:00 前=T-1，之后=T；周末=最近交易日）
PYTHONPATH=. python -c "from data_fetching.dquant_source import latest_trading_date as L; print(L())"
```

```bash
# 0.2 A8 前置：combo_mask 末日须 ≥ 目标日（上游 cron 维护，本仓不拉）
python -c "import pandas as pd; p='/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir_dquant/combo_mask_long.parquet'; d=pd.read_parquet(p, columns=['datetime'])['datetime'].max(); print(d.date())"
```

规则：目标日=`latest_trading_date()`；建议 ≥17:00 再跑。combo_mask 落后则**跳过 A8**。默认不更 `hlv_v1`/`000985`。命令幂等。
划分：**A**=原料；**B**=因子；**P**=推理（消费 B，不训模型）

---

## A. 数据线（顺序固定，fail-fast）

| # | 命令 | 输出 | 依赖/注意 |
|---|------|------|-----------|
| A1 | `PYTHONPATH=. python data_fetching/ex_factors_jy.py` | `market-data/daily-dquant/stock-ex-factors-jy/<股>.parquet` | 稀疏复权 |
| A2 | `PYTHONPATH=. python data_fetching/raw_ohlcv_dquant.py` | `market-data/daily-dquant/stock-ohlcv-dquant/<股>.parquet` | 日频 OHLCV |
| A3 | `PYTHONPATH=. python data_fetching/minute_ohlcv_dquant.py` | `market-data/minute-dquant/raw/<YYYY-MM-DD>.parquet` | **禁止 `--full`** |
| A4 | `PYTHONPATH=. python data_fetching/industry_dquant.py` | `market-data/industry-dquant/industry_panel_zx_dquant.parquet` | 中信一级 |
| A5 | `PYTHONPATH=. python data_fetching/market_cap_dquant.py` | `market-data/market_cap/market_cap_panel.parquet` | 总市值 |
| A6 | `PYTHONPATH=. python data_fetching/fundamentals_dquant.py` | `market-data/fundamentals-dquant/<field>.parquet` | cxl 原料 |
| A7 | `PYTHONPATH=. python data_fetching/industry_index_dquant.py` | `market-data/industry/industry_index_return_dquant.parquet` | 勿动旧 rq 文件 |
| A8 | `PYTHONPATH=. python data_fetching/new_stock_mask_dquant.py` | `backtest_engine/cache_dir_dquant/new_stock_mask_long.parquet` | 需 combo_mask≥目标日 |
| A9 | `PYTHONPATH=. python data_fetching/update_labels.py` | `market-data/labels/{vwap_panel,forward_return_*d}.parquet` | **依赖 A1+A2** |
| A10 | `PYTHONPATH=. python scripts/build_ret20_panel.py` | `factors/helpers/ret20_panel.parquet` | **依赖 A1+A2** |
| A11 | `PYTHONPATH=. python scripts/build_ret20_panel.py --window 1` | `factors/helpers/ret1_panel.parquet` | **依赖 A1+A2**；htsec 动量原料 |
| A12 | `PYTHONPATH=. python data_fetching/turnover_rate_dquant.py` | `market-data/turnover-dquant/turnover_rate_panel.parquet` | 日换手率；htsec 动量原料 |
| A13 | `PYTHONPATH=. python scripts/build_amplitude_panels.py` | `factors/helpers/{amp,amp_hl}_panel.parquet` | **依赖 A1+A2**；长端动量原料 |
| A14 | `PYTHONPATH=. python data_fetching/normal_day_dquant.py` | `market-data/limit-dquant/normal_day_panel.parquet` | 非停牌且未触涨跌停；**依赖 A2+combo_mask** |

```
A1+A2 → A9/A10/A11/A13；A11+A12 → B6；A10+A11+A13+A14 → B7
A5 → B1；A3 → B2；combo_mask → A8/A14；B1+B3+B5+mask → P
```

A 段硬门槛：跑 D 节验证脚本中 raw/mcap/ind/idx/vwap/ret20/minute 七项，全 OK 再进 B。

---

## B. 因子线（A 门槛通过后再跑）

| # | 命令 | 输出 | 注意 |
|---|------|------|------|
| B1 | `PYTHONPATH=. python data_fetching/style_ln_market_cap.py` | `factors/raw-dquant/style-dquant/size/ln_market_cap.parquet` | 依赖 A5 |
| B2 | `PYTHONPATH=. python pipeline/refresh_supersets.py --cache-key prv_v3` | `intermediate-cache-dquant/prv_v3__*/` | 只刷 prv_v3；依赖 A3 |
| B3 | `PYTHONPATH=. python pipeline/refresh_factors_batch.py --factor-glob 'sources/kysec/paper_27_microstructure/specs/*' --end-date $END_DATE` | `factors/raw-dquant/kysec-dquant/paper_27_microstructure/*.parquet` | **kysec-dquant/**；依赖 B2 |
| B4 | cxl 并行（**可选**） | `factors/raw-dquant/cxl-dquant/...` | 依赖 A6；默认模型不用可不跑 |
| B5 | `PYTHONPATH=. python alpha158/daily_update.py` | `factors/raw-dquant/alpha158-dquant/{kline,price,rolling,volume}/` | 必跑；依赖 A2 |
| B6 | 见下方面板型因子循环（htsec 改进动量 8 因子） | `factors/raw-dquant/htsec/paper_04_momentum/*.parquet` | 依赖 A11+A12；当前模型未消费，可选 |
| B7 | 见下方面板型因子循环（kysec 长端动量 1.0/2.0） | `factors/raw-dquant/kysec-dquant/paper_67_long_momentum/*.parquet` | 依赖 A10+A11+A13+A14；当前模型未消费，可选 |

⚠️ B6/B7 是**面板型**因子（load_panel 起手，无 minute superset），`refresh_factors_batch.py`
只服务分钟聚合类因子，对它们会报"无匹配的可批量因子" → 走 `run.py --yolo-only` 循环
（spec 均 incremental_safe，自动走尾窗增量）：

```bash
for d in sources/htsec/paper_04_momentum/specs/* sources/kysec/paper_67_long_momentum/specs/*; do
  qp=$(echo "$d" | sed -E 's#^sources/([^/]+)/([^/]+)/specs/([^/]+)/?$#\1/\2/\3#')
  PYTHONPATH=. python run.py "$qp" --yolo-only --end-date "$END_DATE"
done
```

```bash
END_DATE=$(ls /nfs/ofs-prediction/peterzhenglinpeng/market-data/minute-dquant/raw/2*.parquet | tail -1 | xargs -n1 basename | sed 's/\.parquet//; s/-//g')
echo END_DATE=$END_DATE
```

B4 可选：

```bash
N_FACTOR_JOBS=12; specs=""
for d in sources/cxl/*/specs/*; do
  [ -f "$d/spec.yaml" ] || continue
  specs="$specs $(echo "$d" | sed -E 's#^sources/([^/]+)/([^/]+)/specs/([^/]+)/?$#\1/\2/\3#')"
done
printf '%s\n' $specs | xargs -P "$N_FACTOR_JOBS" -I{} \
  sh -c "PYTHONPATH=. python run.py '{}' --yolo-only --end-date $END_DATE >/dev/null 2>&1 && echo OK || echo FAIL {}"
```

---

## P. 推理线（B 完成后）

入口：`PYTHONPATH=. python -m ml_core.predict`  
配置：`ml_core/predict_config.yaml`（只改这个；口径读 `run_meta.json`，不重训）  
默认 `model_run_id=lgbm_a158_p27_style_shap_all_dq`（a158+p27+style，**无 cxl**）  
`latest_n=1` / `end=null` / `rebuild=false` / `top_n=500`  
产物：`ml/predictions/<run_id>/{pred_panel_live.parquet,signals/YYYY-MM-DD.txt}`

| # | 步骤 | 注意 |
|---|------|------|
| P1 | 三源+mask 末日=目标日（a158/p27/ln_mcap/combo+new_stock） | 不齐则停 |
| P2 | `PYTHONPATH=. python -m ml_core.predict` | `latest_n=1`，`rebuild=false` |
| P3 | pred 末日=目标日；`signals/<目标日>.txt` 行数=500 | append-only |

漏日：临时 `latest_n=N`，跑完改回 1。`rebuild=true` 日常禁止。

---

## C. 收盘后标准流程（复制，fail-fast）

```bash
source /nfs/ofs-prediction/peterzhenglinpeng-code/peterdidi/bin/activate
cd /nfs/ofs-prediction/peterzhenglinpeng-code/factor-replication-quant-new
set -e
# A
PYTHONPATH=. python data_fetching/ex_factors_jy.py
PYTHONPATH=. python data_fetching/raw_ohlcv_dquant.py
PYTHONPATH=. python data_fetching/minute_ohlcv_dquant.py
PYTHONPATH=. python data_fetching/industry_dquant.py
PYTHONPATH=. python data_fetching/market_cap_dquant.py
PYTHONPATH=. python data_fetching/fundamentals_dquant.py
PYTHONPATH=. python data_fetching/industry_index_dquant.py
PYTHONPATH=. python data_fetching/new_stock_mask_dquant.py   # combo_mask 未齐则跳过
PYTHONPATH=. python data_fetching/update_labels.py
PYTHONPATH=. python scripts/build_ret20_panel.py
PYTHONPATH=. python scripts/build_ret20_panel.py --window 1
PYTHONPATH=. python data_fetching/turnover_rate_dquant.py
PYTHONPATH=. python scripts/build_amplitude_panels.py
PYTHONPATH=. python data_fetching/normal_day_dquant.py
# B
END_DATE=$(ls /nfs/ofs-prediction/peterzhenglinpeng/market-data/minute-dquant/raw/2*.parquet | tail -1 | xargs -n1 basename | sed 's/\.parquet//; s/-//g')
PYTHONPATH=. python data_fetching/style_ln_market_cap.py
PYTHONPATH=. python pipeline/refresh_supersets.py --cache-key prv_v3
PYTHONPATH=. python pipeline/refresh_factors_batch.py --factor-glob 'sources/kysec/paper_27_microstructure/specs/*' --end-date "$END_DATE"
PYTHONPATH=. python alpha158/daily_update.py
# 可选 B4 cxl：见上节
# P
PYTHONPATH=. python -m ml_core.predict
```

---

## D. 最终验证

```bash
PYTHONPATH=. python - <<'PY'
import pandas as pd, glob, os
from pathlib import Path
R='/nfs/ofs-prediction/peterzhenglinpeng'
from data_fetching.dquant_source import latest_trading_date
T=latest_trading_date(); print('target', T)
def show(name, val):
    print(('OK ' if str(val)==T else 'BAD'), f'{name:18s}', val)
show('raw_ohlcv', pd.read_parquet(sorted(glob.glob(f'{R}/market-data/daily-dquant/stock-ohlcv-dquant/*.parquet'))[0]).index.max().date())
show('minute', os.path.basename(sorted(glob.glob(f'{R}/market-data/minute-dquant/raw/2*.parquet'))[-1])[:-8])
for n,p in {
 'market_cap':f'{R}/market-data/market_cap/market_cap_panel.parquet',
 'industry_zx':f'{R}/market-data/industry-dquant/industry_panel_zx_dquant.parquet',
 'industry_idx':f'{R}/market-data/industry/industry_index_return_dquant.parquet',
 'vwap':f'{R}/market-data/labels/vwap_panel.parquet',
 'ret5':f'{R}/market-data/labels/forward_return_5d.parquet',
 'ret20':f'{R}/factors/helpers/ret20_panel.parquet',
 'ln_mcap':f'{R}/factors/raw-dquant/style-dquant/size/ln_market_cap.parquet',
}.items():
    show(n, pd.to_datetime(pd.read_parquet(p).index).max().date())
for m in ['combo_mask_long','new_stock_mask_long']:
    show(m, pd.read_parquet(f'{R}/backtest_engine/cache_dir_dquant/{m}.parquet', columns=['datetime'])['datetime'].max().date())
fs=sorted(glob.glob(f'{R}/factors/raw-dquant/kysec-dquant/paper_27_microstructure/*.parquet'))
show('paper27_dq', pd.read_parquet(fs[0]).index.max().date() if fs else 'MISS')
fs=sorted(glob.glob(f'{R}/intermediate-cache-dquant/prv_v3*/*.parquet'))
show('prv_v3', pd.read_parquet(fs[0], columns=['date'])['date'].max().date() if fs else 'MISS')
fs=sorted(glob.glob(f'{R}/factors/raw-dquant/alpha158-dquant/kline/*.parquet'))
show('alpha158', pd.read_parquet(fs[0]).index.max().date() if fs else 'MISS')
fund=sorted(os.listdir(f'{R}/market-data/fundamentals-dquant'))
show('fundamentals', pd.read_parquet(f'{R}/market-data/fundamentals-dquant/{fund[0]}').index.max().date() if fund else 'MISS')
run='lgbm_a158_p27_style_shap_all_dq'
pred=Path(R)/f'ml/predictions/{run}/pred_panel_live.parquet'
show('pred_live', pd.read_parquet(pred).index.max().date() if pred.exists() else 'MISS')
show('signal_txt', T if (Path(R)/f'ml/predictions/{run}/signals/{T}.txt').exists() else 'MISS')
PY
```

全部 `OK`=target 才算完成。`forward_return_*` 近 N 日可为 NaN（前瞻标签，正常）。

---

## E. 禁忌

1. 不要 `cd` 进子目录；一律项目根 + `PYTHONPATH=.`
2. 禁止 minute `--full`
3. paper_27 只认 `kysec-dquant/`；industry_index 只认 `*_dquant.parquet`
4. B3 必须 `--end-date $END_DATE`；A8 前查 combo_mask
5. 默认跳过 hlv_v1 / 000985
6. 任一步失败：修好重跑该步，勿跳过下游
7. P 前 a158/p27/ln_mcap+mask 须齐；日常 `rebuild=false`；不重训、不改模型文件
