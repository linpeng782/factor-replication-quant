# 服务器日更生产手册（给执行 agent）

> **读者**：每个交易日收盘后在服务器上跑增量更新的 agent / 操作者。
> **目标**：把「因子面板 → 掩码 → 信号 → 回测」整条链路推到最新交易日。
> **设计准则**：所有层都是 **append-only 增量、历史冻结、可复现**（见 `minute_incremental_design.md` /
> `alpha158_incremental_design.md` / `cxl_fundamental_incremental_design.md` 三胞胎设计文档）。
> ⚠️ 全部在**服务器**执行；不要在 Mac 上跑。命令默认值已是服务器路径。

---

## 0. 环境 & 固定路径

```bash
source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate   # Python 3.11（勿用裸 python）
cd     /nfs/volume-1593-1/peterzhenglinpeng/factor-replication-quant-new   # 本仓（因子线 + 数据线）
```

| 名称 | 路径 |
|---|---|
| venv (py3.11) | `/nfs/volume-1593-1/peterzhenglinpeng/peterdidi` |
| 因子仓（本仓） | `/nfs/volume-1593-1/peterzhenglinpeng/factor-replication-quant-new` |
| 数据根 `FACTOR_REPL_DATA_ROOT` | `/nfs/ofs-prediction/peterzhenglinpeng`（**代码默认值，单独跑 python 时无需设**） |
| 因子面板 | `<数据根>/factors/raw/<source>/<group>/<factor>.parquet` |
| 基本面 PIT 库 | `<数据根>/market-data/fundamentals/<field>.parquet` |
| 掩码 | `<数据根>/market-data/masks/{combo_mask_long,new_stock_mask_long}.parquet` |
| ML 模型 / 信号 | `<数据根>/ml/models/<run_id>/` ・ `<数据根>/ml/signals/<run_id>/` |
| 掩码来源（**另一项目产出**） | `<数据根>/backtest_engine/cache_dir/` |
| 回测仓 | `/nfs/volume-1593-1/peterzhenglinpeng/daily-realtime-backtest-pipeline` |
| 当前生产模型 run_id | `cxl_a158_p27_raw_shap_v2`（如有多个模型，对每个重复 C/D 段） |

> ⚠️ **`daily_update.sh` 会 `export FACTOR_REPL_DATA_ROOT`**（已设为服务器默认）。**单独跑某个 python 命令**时不要乱设这个变量——代码默认值就是对的。

---

## 1. 全流程总览（4 段，按序执行）

| 段 | 做什么 | 自动化程度 |
|---|---|---|
| **A. 因子生产** | 数据线 + 因子面板 + labels | ✅ 一条命令 `daily_update.sh` |
| **B. 掩码更新** | 从 backtest_engine 拷 combo/new_stock 掩码 | ⚠️ 手动 + **必须先验证对齐** |
| **C. 信号生产** | 用模型推理 → 导出每日 top-N 选股 | 手动 2 条命令 |
| **D. 回测** | 跑回测看收益/IC | 手动 1 条命令 |

> **依赖关系**：C（信号）同时依赖 A（因子到最新）**和** B（掩码到最新）——掩码常是「信号能推到哪天」的瓶颈，所以 **B 一定要在 C 之前做**。

---

## 2. A. 因子生产（一条命令）

```bash
cd /nfs/volume-1593-1/peterzhenglinpeng/factor-replication-quant-new
bash pipeline/daily_update.sh
```

`daily_update.sh` 自动按序跑（数据线 fail-fast；因子线失败仅告警）：

| 步 | 内容 | 说明 |
|---|---|---|
| 1–5 | ex_factors / raw_ohlcv / minute_ohlcv(+--full) / industry / market_cap | 数据线，拉当日数据（需 rqdatac）|
| 5b | **fundamentals** | 基本面 PIT 快照 append 当日（cxl 因子线读它）+ `--audit` 重述审计（只告警）|
| 6 | **refresh_supersets `--cache-key prv_v3`** | 只刷 paper_27 用的 prv_v3 superset（含 pass2 新股回填）。**默认不刷** apm/sm/tide/dazzle（那些是别的 paper 用的，本生产不更新）|
| 7a | **批量 L3** `refresh_factors_batch.py`（paper_27/superset 因子）| **读 prv_v3 superset 一次、算 23 个因子**（~14× 快，~3-4 min）。产出与逐个 run.py **bit 一致**。增量 append；不安全/非 superset 因子自动跳过 |
| 7b | **run.py 并行**（cxl，22 个，读本地基本面）| L3 增量。**自动判定**：面板已存在→增量（尾窗只算新日）；cxl 5 个因子（`reg_pb_gshe`/`reg_pe_hist` filter→rolling、`roic_ttm_*8` change_on）→全量重算（确定性，**正常非 bug**，见 §6）|
| 7c | **alpha158**（`alpha158/daily_update.py`）| ⚠️ alpha158 **无 spec、glob 扫不到**，必须独立这步；否则下游信号被 alpha158 旧日期卡死。读本地 raw_ohlcv、零 API |
| 8 | ml/labels.py | 标签回填（失败不阻塞）|

> **范围 = 生产模型 `cxl_a158_p27_raw_shap_v2` 用到的源**：alpha158 + cxl(22) + kysec/paper_27(23)。
> **不更新** founder / guosen / 其它 kysec paper。要更全：把 `MINUTE_FACTOR_GLOB`（superset 因子，走 7a 批量）
> 与 `RUNPY_FACTOR_GLOB`（非 superset 因子，走 7b run.py）改宽，并置 `SUPERSET_KEY=`（刷全部 superset）。例：
> `MINUTE_FACTOR_GLOB='sources/*/*/specs/*' RUNPY_FACTOR_GLOB='sources/cxl/*/specs/*' SUPERSET_KEY= bash pipeline/daily_update.sh`

**耗时参考**（128核/800G，单个新交易日）：数据线 ~5–10min；refresh prv_v3 ~5min；
7a 批量 paper_27 ~3–4min；7b cxl 并行 ~3–5min；7c alpha158（158 面板写盘）~3–8min；labels ~1–3min。
→ **A 段合计 ≈ 20–35min**（B 掩码 + C 信号 + D 回测 另 ~10min）。落后多天则数据线/superset 按天数增加。

**验收 A**：
```bash
PYTHONPATH=. python - <<'PY'
import pandas as pd, glob
from core.config import RAW_FACTOR_BASE, FUNDAMENTALS_DIR, INTERMEDIATE_CACHE_DIR
def mx(p): return pd.to_datetime(pd.read_parquet(p, columns=[]).index).max().date()
print("fundamentals:", mx(FUNDAMENTALS_DIR/"net_profit_mrq_0.parquet"))
print("cxl 抽查:", mx(RAW_FACTOR_BASE/"cxl/roe_series/roe_apoq_mrq.parquet"))
print("paper_27 抽查:", mx(RAW_FACTOR_BASE/"kysec/paper_27_microstructure/peak_minute_count.parquet"))
print("alpha158 抽查:", mx(RAW_FACTOR_BASE/"alpha158/rolling/MAX5.parquet"))
PY
```
四项末日都 = 最新交易日 → A 成功。

---

## 3. B. 掩码更新（依赖另一项目 + 必须验证对齐）

掩码（ST/停牌/涨停/新股）由**另一个项目**产出到 `backtest_engine/cache_dir/`，本仓不生成。日更时把最新版拷到 `market-data/masks/`。

> 🔑 **铁律：拷之前必须验证「重叠段逐格一致」**——历史掩码必须冻结，错拷会污染历史信号。

```bash
PYTHONPATH=. python - <<'PY'
import pandas as pd
CACHE="/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir"
MASKS="/nfs/ofs-prediction/peterzhenglinpeng/market-data/masks"
for fn, cols in [("combo_mask_long.parquet", ["is_st","is_suspended","is_limit_up"]),
                 ("new_stock_mask_long.parquet", ["is_new_stock"])]:
    src=pd.read_parquet(f"{CACHE}/{fn}"); cur=pd.read_parquet(f"{MASKS}/{fn}")
    for d in (src,cur): d["datetime"]=pd.to_datetime(d["datetime"])
    cut=cur["datetime"].max()
    s=src[src.datetime<=cut].set_index(["order_book_id","datetime"])[cols].sort_index()
    c=cur[cur.datetime<=cut].set_index(["order_book_id","datetime"])[cols].sort_index()
    common=s.index.intersection(c.index)
    mism=sum(int((s.loc[common,k].values!=c.loc[common,k].values).sum()) for k in cols)
    extra=len(c.index.difference(s.index))
    print(f"{fn}: 源末日={src.datetime.max().date()} 当前末日={cur.datetime.max().date()} "
          f"重叠不一致={mism} 当前独有={extra} -> {'✅可拷' if mism==0 and extra==0 else '❌不对齐,勿拷,先查'}")
PY
```

**两个文件都显示 `✅可拷` 才执行拷贝**（先备份当前版，再覆盖）：
```bash
CACHE=/nfs/ofs-prediction/peterzhenglinpeng/backtest_engine/cache_dir
MASKS=/nfs/ofs-prediction/peterzhenglinpeng/market-data/masks
TS=$(date +%Y%m%d)
cp "$MASKS/combo_mask_long.parquet"     "$MASKS/combo_mask_long.parquet.bak.$TS"
cp "$MASKS/new_stock_mask_long.parquet" "$MASKS/new_stock_mask_long.parquet.bak.$TS"
cp "$CACHE/combo_mask_long.parquet"     "$MASKS/combo_mask_long.parquet"
cp "$CACHE/new_stock_mask_long.parquet" "$MASKS/new_stock_mask_long.parquet"
```
（若某文件源末日 ≤ 当前末日，说明上游还没更新，跳过该文件、信号只能推到旧的那天。）

---

## 4. C. 信号生产（每个生产模型一遍）

```bash
RUN=cxl_a158_p27_raw_shap_v2          # 生产模型 run_id
PYTHONPATH=. python -m ml.predict_live  --run-id "$RUN" --start 2022-01-01
PYTHONPATH=. python -m ml.export_signal --run-id "$RUN" --source live --layout daily --top-n 500
```

- `predict_live` 用 `models/$RUN/{model.txt,scaler_x.parquet,selected_features.json}` 推理，自动末日 =
  **min(入选因子覆盖末日, 掩码末日)**。所以**掩码（B）不更新，信号就推不到最新**——日志里看
  `[filters] ... 区间 ~YYYY-MM-DD` 和 `区间 ...~YYYY-MM-DD` 确认推到哪天。
- `export_signal` **默认 append-only**：只新增信号目录里尚不存在的交易日 `.txt`，**绝不覆盖历史信号**
  （= 交易流水，写一次冻结；无前视、可复现）。日志会报「新增 N 份，跳过已存在 M 份」。
  日更照上面跑即可，**不需要也不应该**手动算 `--start`。
- 产物：`ml/predictions/$RUN/pred_panel_live.parquet`、`ml/signals/$RUN/YYYY-MM-DD.txt`（每日 top-500 排序名单）。

> ⚠️ **不要**为了"刷新历史"去加 `--rebuild`：那会按当前因子口径重写历史信号、引入前视、破坏可复现。
> `--rebuild` 仅用于**显式、罕见的 reconcile**（如修了某因子 bug 后要重定基历史信号），由人决策。
> 模型不需重训：日更只是「同模型 + 新因子 → 新打分」。重训见 `ml-pipeline-guide.md`。

---

## 5. D. 回测

```bash
cd /nfs/volume-1593-1/peterzhenglinpeng/daily-realtime-backtest-pipeline
source /nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate
python compare_signals.py --signals cxl_a158_p27_raw_shap_v2      # 单信号
# 多信号对比： python compare_signals.py --signals <run_a> <run_b>
```
- 基准 = 中证全指 000985。产物在 `results/compare/`。
- ⚠️ **已知 bug**：并行回测多个信号（workers≥2）会抢 `results/latest` 软链接报 `FileNotFoundError`。
  **规避**：一次只 `--signals` 一个；要对比多个就分多次单跑。

---

## 6. 关键不变量 / 须知（理解这些才不会误判）

1. **历史冻结、append-only**：每层都只追加新交易日，历史不重写。重跑某天得到相同结果（幂等）。
2. **cxl 5 个因子全量重算是正常的**：`reg_pb_gshe`/`reg_pe_hist`(filter→rolling)、`roic_ttm_{all,dev,ind}_*8`(change_on) 日历回看无界，run.py 自动对它们全量重算（从**冻结的本地 PIT 库**确定性重算、历史不漂移）。看到它们每天"全量"是设计，**不是 bug**。其余因子走有界尾窗增量。
3. **基本面是冻结 PIT 快照**：`fundamentals.py` 只 append 当天、**绝不回写历史**。`--audit` 检测厂商财报重述，**只告警绝不覆盖**。
4. **掩码对齐铁律**：见 §3，错拷会污染历史。
5. **标签 provisional**：最近 ~N+1 天的 forward-return 标签会随新交易日变（天然非冻结）；IC 仅 > N+1 天可复现。

---

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: core` | 必须在仓库根执行，且带 `PYTHONPATH=.` |
| 写到了 `/Users/didi/...` | 误设了 `FACTOR_REPL_DATA_ROOT`；unset 它用代码默认 `/nfs/ofs-prediction/peterzhenglinpeng` |
| 信号推不到最新交易日 | 掩码没更新（B 段）；查 predict_live 日志的 filters 末日 |
| 回测 `results/latest` 报错 | 并行竞态；一次只跑一个 `--signals` |
| 某 cxl 因子每天"全量重算" | 正常（§6.2）|
| `minute/raw 为空` / superset 报错 | 数据线（A 段 step 3）未跑或失败，先补数据线 |
| 重述审计告警「本地 vs API 不一致」 | 记录即可，**不要** `--full` 覆盖（除非人工决策重定基）|

---

## 8. 代码同步 / push

- 代码仓 `git pull` / `git push` 走**内网 GitLab**（`git.xiaojukeji.com` 可达；github.com 在本服务器**不通**）。
- 若 push github 超时，等网络或改用内网 remote；本地 commit 不会丢。
- 因子/掩码/信号等**数据产物不入 git**（在 `FACTOR_REPL_DATA_ROOT` 下）。
