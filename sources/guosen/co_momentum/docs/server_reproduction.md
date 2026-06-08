# 联合动量因子族 服务器复现 / 对齐手册

> 给 SSH 端 agent：按本手册顺序自产 5 因子，并用本地实测的【三层基准】严谨对齐。
> 本地环境：macOS，2026-06-08 实测。代码经 git pull 取得（commit 含 industry_co_momentum 算子 + 5 spec）。

---

## 0. 前置检查
```bash
cd factor-repilcation-quant && git pull        # 取算子 + 5 spec + 本手册
# 行业指数收益（唯一新上游；本地 scp 来 或 服务器自建）
ls $DATA_ROOT/market-data/industry/industry_index_return.parquet \
  || python data_fetching/industry_index.py --full     # 自建需 rqdatac，~3s，30个中信一级指数
# 其余上游（daily OHLCV / ex-factors / industry_panel_zx / masks / labels / market_cap）应已就位
```

## 1. 生产步骤（严格按序）
```bash
# ① 算法自检（秒级，无数据依赖，最硬基准）
PYTHONPATH=. python scripts/comomentum_cmc.py --example-only

# ② 生产 5 因子（各自 raw→cleaned→neu + 评估PNG）
for f in ICM VICM VICR CMC MCMC; do
  PYTHONPATH=. python run.py guosen/co_momentum/$f
done
#   产出 factors/{raw,cleaned,neu}/guosen/co_momentum/<f>.parquet

# ③（推荐）论文对齐月频验证 —— 最权威的对齐
PYTHONPATH=. python scripts/comomentum_family_verify.py
```

---

## 2. 三层对齐基准（本地实测，2010-2023，vwap 收益）

### 第①层 — 算例金标准【必须 bit 一致，无数据依赖】
`comomentum_cmc.py --example-only` 输出：
```
ICM = 0.7231%   (论文标注 0.719%，差异仅论文权重四舍五入)
```
→ **服务器必须一字不差得到 0.7231%**。不一致 = 算子代码被改坏，停止排查。

### 第②层 — 论文对齐月频 RankIC / 年化ICIR【主对齐，对论文+本地】
`comomentum_family_verify.py` 输出（应逐行对上）：

| 因子 | raw (RankIC/ICIR) | **neu (RankIC/ICIR)** | 论文 |
|------|-------------------|----------------------|------|
| ICM  | +4.67% / 1.70  | **+5.83% / 3.69** | 5.59 / 3.85 |
| VICM | +5.07% / 1.88  | **+6.17% / 3.85** | 5.90 / 3.90 |
| VICR | −4.24% / −1.31 | **−6.21% / −3.47** | −5.93 / −3.55 |
| CMC  | +5.12% / 1.85  | **+6.35% / 3.71** | 6.02 / 3.86 |
| MCMC | +7.32% / 2.89  | **+6.91% / 3.92** | 6.40 / 4.00 |

→ **neu 列是对齐核心**：5 因子 neu RankIC 应在 ±0.5%、ICIR 在 ±0.3 内对上本地与论文。
   raw 列 ICIR 系统性偏低是正常的（行业暴露 β，见 docs/co_momentum.md 的 between/within 洞察）。

### 第③层 — 入库日频 IC【run.py 直接输出，量级+方向+单调】
`run.py guosen/co_momentum/<f>` 每个的 neu 版（direction 已自动判定，VICR 为 −1）：

| 因子 | neu IC20d | neu ICIR(日频,未年化) | 单调性 mono |
|------|-----------|----------------------|-------------|
| ICM  | +5.39% | 0.99 | 0.96 |
| VICM | +5.70% | 1.07 | 0.96 |
| VICR | +5.78% | 0.97 | 0.96 |
| CMC  | +5.88% | 1.03 | 0.96 |
| MCMC | +6.46% | 1.03 | 0.95 |

→ 服务器对照：**方向全为正（VICR dir=−1 翻正后）、单调性 > 0.93、IC20d 量级 5~6.5%**。

### bit 级旁证（若服务器上游数据与本地一致）
raw CMC 共 13,932,611 个非空值，本地 spec 路径 vs standalone 最大绝对差 = 0.00e+00。

---

## 3. 对齐判定 + 容差
| 层 | 判定 | 容差 |
|----|------|------|
| ① 算例 | ICM=0.7231% | **0（必须 bit 同）** |
| ② 月频 neu | 5因子 RankIC/ICIR 对上表 | RankIC ±0.5%，ICIR ±0.3 |
| ③ 日频 IC | 方向全正、mono>0.93、IC20d 5~6.5% | 量级级，不强求 bit |

**为何 ②③ 不强求 bit**：服务器上游数据范围（OHLCV/mask/labels 末日）可能与本地不同 →
因子结尾日期、绝对 IC 会有差异。但**方向、单调性、月频 neu 对论文的吻合度**应稳定。
日期范围参考：本地 raw 2010-02-01~2026-05-29；cleaned/neu 截至 mask/labels 末日（本地 2026-05-27）。

## 4. 故障排查
- **① 算例不对** → 算子 `core/operators/industry_co_momentum.py` 被改坏（不应发生）。
- **② neu RankIC 远低于表（如 ICIR<2）** → 多半是 `industry_index_return.parquet` 缺失/错位：
  检查行业指数收益面板是否 33 列、是否覆盖回测期、行业名是否能对上 industry_panel_zx（含3旧名别名）。
- **③ 某因子方向反** → evaluate 自动判 direction；看日志 dir=±1。VICR 本就是 −1。
- **结尾日期偏早** → 服务器上游 OHLCV/mask/labels 比本地旧，属正常（日更范畴）。
- **MCMC 报错/全 NaN** → MCMC 需 rqdatac 取中证全指 000985，确认 rqdatac.init() 可用。

## 5. 日更接入（重要，否则因子停滞）
`data_fetching/industry_index.py` 需加入日更链（在 industry.py 后）：
```bash
python industry_index.py        # 增量日更行业指数收益
```
否则行业指数不更新 → CMC 族结尾日期停在当前不前进。
