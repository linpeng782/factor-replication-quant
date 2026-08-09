# 市场层广播特征 —— 评审、全景梳理与第一批实施方案

> 日期：2026-08-07
> 关联文档：`/nfs/ofs-prediction/peterzhenglinpeng-code/市场情绪因子口径定义文档.md`（v1.0，同事编写，本文不修改它）
> 目标模型：`lgbm_shap128_a158_size_kymom_htmom_csrank5_dq`
> 本文定位：**动手前的设计与决策文档**。第 4/5 节是可直接照着写代码的实现规格。

---

## 0. 摘要

- 《市场情绪因子口径定义文档》口径扎实，但有 4 个必须修正的问题：`_ext` 哑变量对 GBDT 冗余、双重平滑导致信息过期、iVIX 数据 2019 年后已停发、两融/期货数据仓库里没有。
- 文档 14 项里**波动率维度几乎空白**（唯一一项 iVIX 还拿不到数），这正是同事提到的方向。本文第 3 节把市场层特征扩展为 **8 个维度、约 60 项**的全景清单。
- **工程上是好消息**：`style-dquant/calendar/week_of_year.parquet` 已经是一个"每日单值广播到全市场"的现成先例，跑通过。走同样的模式，**ml_core 一行代码都不用改**。
- 第一批建议做 **10 个特征，全部只依赖现有数据，零新建数据管线**。
- **两条硬约束**（第 6 节）：(a) 广播特征只能用于 LGBM 线，MLP 线的 `DailyCrossSectionMAD` 会把它抹成全 0；(b) 广播特征的 SHAP 重要度会系统性虚高，不能和截面因子放同一张表排序。

---

## 1. 对现有《市场情绪因子口径定义文档》的评审

### 1.1 根本问题：广播特征与截面模型的适配性

模型标签是 `csrank_normcdf_robust`（逐日截面秩 → 正态分位），是**纯截面排序目标**。同一天所有股票同值的特征，在单日截面上对排序的直接贡献严格为 0；它唯一的作用路径是被树用作分裂条件去**调制其他因子的用法**——本质是**因子择时**，不是选股。

两个真实风险：

1. **有效样本量的幻觉。** 2016–2026 约 2400 个交易日，广播特征的独立观测数就是 2400。但在长表里它被复制到约 5000 只股票上变成 1200 万行，LGBM 的直方图分裂器看到的是 1200 万样本，`min_child_samples` 一类正则完全失效。
2. **regime 空间爆炸。** 14 个广播列，即使每列只切 5 个桶，联合状态空间也远超 2400 天能支撑的自由度。

**结论：不同意文档 §4「分项独立入模、不预先合成」。** 这条对第二类截面因子是对的，对第一类广播特征必须反过来——**先降维**，控制在 5–10 个彼此低相关的维度代表，或合成 3–4 个正交主轴。

### 1.2 `xxx_ext` 极端哑变量对 GBDT 是纯冗余 → 建议删除

`ext = 1[pct > 0.9] − 1[pct < 0.1]`，是 `pct` 的确定性单调分段函数。树模型本身就能在 0.9 / 0.1 处分裂，这一列提供不了任何新信息，只是把特征数翻倍、稀释 SHAP、扩大过拟合面。

**建议：去掉 `_ext`，14×2 = 28 列降回 14 列。** 若日后接线性模型或做规则化择时，再单独加回。

### 1.3 双重平滑导致信息过期 → 建议补"变化"视角

"20 日均 → 滚动 2 年分位"两层平滑叠加，实际刻画的是约一个月前的市场状态，而我们是 5/10 日调仓。

经验上情绪类指标的**变化率（加速度）比水平值预测力更强**——顶部往往不是"情绪最高"，而是"情绪开始掉头"。建议每个指标至少给出水平 + 变化两个视角（`pct` 的 5 日差分，或 `当前值 / 20日均 − 1`）。

### 1.4 数据可得性硬伤（已实测确认）

`<DATA_ROOT> = /nfs/ofs-prediction/peterzhenglinpeng`，`market-data/` 下实有：`daily-dquant/{stock-ohlcv-dquant, per-day, stock-ex-factors-jy}`、`minute-dquant/raw`、`turnover-dquant`、`market_cap`、`industry{,-dquant}`、`index/000985_segments`、`limit-dquant`、`fundamentals-dquant`、`labels`。

| 文档项 | 状态 |
|---|---|
| #5 融资余额、#6 融资买入额占比、§2.3 融资截面因子 | ❌ **无两融数据**，需新建 `get_securities_margin` 管线 |
| #13 股指期货升贴水 | ❌ **无期货数据** |
| #14 iVIX | ⚠️ **上交所官方 iVIX 已于 2018-12-24 起停止发布**。按"固定单一官方源"口径，该特征只有 2015-02～2018-12 约 4 年数据，其余全 NaN，**训练段（≤2017-11）有值、测试段（2020+）全空**，是最坏的情况。此口径必须重写。**请同事核实。** |
| #12 破净占比 | ⚠️ 需确认 `fundamentals-dquant` 内有无 `pb_ratio_lf` |
| #7 自然涨跌停 | ⚠️ `combo_mask_long` 只有 `is_limit_up`，**没有 `is_limit_down`**；跌停需从 OHLCV 自算（`close == round(prev_close × (1−limit), 2)`），且一字板判定需 `low == 涨停价` |
| #1~#4, #8~#11 | ✅ 现有日频 OHLCV + 换手率 + 市值 + mask 足够 |

### 1.5 第二类截面因子：东吴换手率系列已实测，结论为负

`turn20 / str20 / pct_turn20 / gtr20` 四个已复现落库（`sources/dongwu/paper_07_stable_turnover/`），并做过完整 A/B：

| 实验 | test IC | ICIR | 回测累计收益 |
|---|---|---|---|
| 基线（无 dongwu） | 0.1390 | 1.119 | **596.41%** |
| E1 raw 四因子 | 0.1395 | 1.118 | 587.18% |
| E2 neu 四因子 | 0.1389 | **1.136** | 526.79%（全部模型中第 13） |

**IC 微涨、回测反而跌**。原因不难理解：Alpha158 的 30 个成交量因子（VMA/VSTD/WVMA/VSUMP/VSUMN/VSUMD × 5 窗口）+ htsec 8 个换手率加权动量，已经把换手率维度榨得很干。

**建议：文档 §2.1 剩余的 SCR / UTD 优先级下调，不再投入。** 反而 §2.2 量价相关性（CPV / RPV / SRV）更值得做——分钟级日内结构，与 p27 同源但角度不同，Alpha158 完全没覆盖，且 `minute-dquant/raw` 数据与管线现成。**建议 §2.2 优先级提到 §2.1 之上。**

### 1.6 值得肯定的地方

- §验证协议第 3 步已经意识到"广播特征单日截面同值，单因子 IC 无意义"，要用 SHAP + 整体 RankIC 评估——这个认知是对的，很多人会在这里翻车。
- 两融 T+1 披露、信号放次日 9:00 后——前视控制意识到位。
- U 型方向的标注（而非强行标正负）说明作者理解这批指标的非单调性。

---

## 2. 命名与目录约定（先定下来，后面都按这个走）

```
factors/raw-dquant/market-dquant/<group>/<feature>.parquet
```

- `<source>` = `market-dquant`（与 `style-dquant`、`alpha158-dquant` 平级）
- `<group>` 按维度分，便于在 `train_config.yaml` 里**按维度独立开关做消融**：
  - `vol`      —— 波动与风险
  - `breadth`  —— 广度与赚钱效应
  - `regime`   —— 趋势与量能状态
  - `alphaenv` —— alpha 环境（因子动量）
- 特征名统一前缀 `mkt_` / `ind_` / `fac_`，与截面因子在 SHAP 表里一眼可分。

> 依据：`ml_core/features.py::discover_features` 的 glob 是 `base.glob("*/*/*.parquet")`，即 `<source>/<group>/<factor>.parquet` 三层；source 匹配支持 `rel == s` 或 `rel.startswith(s + "/")`，所以 `sources` 里既能写 `market-dquant`（全要）也能写 `market-dquant/vol`（只要一个维度）。

---

## 3. 市场层特征全景梳理（8 维度）

可得性：✅ 现有数据可算 ／ ⚠️ 需新拉数 ／ 🔬 需分钟数据
优先级：★★★ 第一批 ／ ★★ 第二批 ／ ★ 备选

### A. 波动与风险（原文档空白，最该补 —— 同事提的方向）

| # | 特征 | 口径 | 数据 | 优先级 |
|---|---|---|---|---|
| A1 | **全市场截面离散度** | 每日全A个股收益的横截面 std，20 日均 → 2 年分位 | ✅ | **★★★** |
| A2 | **平均两两相关性** | 由等权组合方差与成分股方差反解的 implied avg corr，20 日窗 | ✅ | **★★★** |
| A3 | 市场已实现波动 RV | 等权市场日收益 20/60 日 std 年化 → 2 年分位 | ✅ | **★★★** |
| A4 | 波动期限结构 | RV20 / RV60 − 1（波动加速度） | ✅ | **★★★** |
| A5 | 半方差比 | 下行波动 / 上行波动（20 日） | ✅ | ★★ |
| A6 | 市场日内振幅 | 个股 (H−L)/前收 的截面中位数，20 日均 → 分位 | ✅（`amp_panel`） | ★★ |
| A7 | 隔夜/日内波动比 | 隔夜跳空 std ÷ 日内 std | ✅ | ★★ |
| A8 | **行业收益离散度** | 33 个中信行业指数日收益的截面 std，20 日均 → 分位 | ✅ | **★★★** |
| A9 | 跳跃频率 | 市场 \|日收益\| > 2σ 的 20 日占比 | ✅ | ★ |
| A10 | 收益偏度/峰度 | 市场 60 日收益 skew / kurt | ✅ | ★ |
| A11 | GARCH 条件波动 | GARCH(1,1) 条件方差，作 iVIX 的可行代理 | ✅ | ★★ |
| A12 | 官方 iVIX | 见 §1.4，2019+ 停发 | ❌ | — |

> A1、A2 是我最想加的两个。**截面离散度直接度量"选股环境好不好"**：离散度低时（如 2024 年初极端抱团）任何 alpha 都赚不到钱，离散度高时因子收益放大。这比"情绪是否过热"与我们模型的相关性直接得多。A2 平均相关性是抱团/羊群的量化度量，与 2021 核心资产、2024 微盘踩踏这类事件高度对应。

### B. 趋势与 regime 状态

| # | 特征 | 口径 | 数据 | 优先级 |
|---|---|---|---|---|
| B1 | 市场动量 | 等权市场 5/20/60/120 日收益 | ✅ | ★★ |
| B2 | **距 250 日高点回撤** | NAV / rolling_max(NAV, 250) − 1 | ✅ | **★★★** |
| B3 | 均线状态 | close/MA20、MA20/MA60、MA60/MA250 | ✅ | ★★ |
| B4 | **量能扩张** | 全A成交额 / 其 60 日均 − 1 | ✅ | **★★★** |
| B5 | 换手率变化率 | 原文档 #1 的差分版（补充而非替代） | ✅ | ★★ |
| B6 | 牛熊/震荡状态 | 由 B2 + A3 规则划分，或 HMM 后验概率 | ✅ | ★ |

### C. 风格与结构（原文档完全没有，但对选股模型最直接）

| # | 特征 | 口径 | 数据 | 优先级 |
|---|---|---|---|---|
| C1 | **大小盘相对强弱** | 用自有市值面板构造小盘 decile − 大盘 decile 的 20/60 日收益差（**无需外部指数**） | ✅ | **★★** |
| C2 | 价值成长相对强弱 | 国证价值 vs 成长指数 | ⚠️ | ★★ |
| C3 | **alpha 环境 / 因子动量** | 一组固定代表因子近 20 日 rank-IC 的绝对值均值 | ✅ | **★★★** |
| C4 | 因子 IC 稳定性 | 上述 IC 的近期 std / ICIR | ✅ | ★★ |
| C5 | 行业轮动强度 | 行业收益排名的 20 日 Spearman 自相关（低 = 轮动快） | ✅ | ★★ |
| C6 | 成交额行业集中度 | 行业成交额 HHI；涨幅前 5 行业成交占比 | ✅ | ★ |
| C7 | 北向资金净流入 | ⚠️ **2024-08-19 起交易所已停止日频披露**，前后样本不一致，**不建议用** | ❌ | — |
| C8 | 主动买卖资金净额 | 分钟级 tick rule 近似大单净流入占比 | 🔬 | ★★ |

> C3 是这批里性价比最高、且没有任何研报会告诉你的一个。**与其用宏观情绪指标间接猜"现在哪类因子有效"，不如直接把因子近期 IC 喂进去。** 成本极低（IC 时序在 `factor-inventory` 里已有），风险是 factor momentum 在 A 股是否稳定需实测（美股上显著）。

### D. 广度与赚钱效应（原文档已覆盖 8 成，补 5 个）

| # | 特征 | 口径 | 数据 | 优先级 |
|---|---|---|---|---|
| D1 | 涨跌家数比 A/D | 上涨家数 / 下跌家数，20 日均 | ✅ | ★★ |
| D2 | **ADL 与指数背离** | 累计(涨−跌)家数线的 20 日斜率 z 值 − 市场 20 日收益 z 值 | ✅ | **★★★** |
| D3 | McClellan 振荡器 | EMA19(A−D) − EMA39(A−D) | ✅ | ★★ |
| D4 | TRIN / ARMS | (涨家数/跌家数) ÷ (涨家成交额/跌家成交额) | ✅ | ★★ |
| D5 | **赚钱效应偏离** | 个股涨幅中位数 − 市值加权市场涨幅 | ✅ | **★★★** |
| D6 | 原文档 #2/#4/#8~#11 | 强势股占比、创新高低、封板率、连板率、多空排列 | ✅ | ★★ |
| D7 | 原文档 #12 破净占比 | 需 `pb_ratio_lf` | ⚠️ | ★★ |

> D2、D5 抓的是"指数涨但个股不涨"这类典型顶部结构，比单看强势股占比更锐利。**D5 对我们尤其重要——我们是全市场近似等权选股，市值加权的指数涨幅根本代表不了持仓环境。**

### E. 流动性与交易成本

| # | 特征 | 口径 | 数据 | 优先级 |
|---|---|---|---|---|
| E1 | 全市场 Amihud 非流动性 | median(\|ret\| / 成交额)，20 日均 | ✅ | ★★ |
| E2 | **Corwin-Schultz 价差估计** | 仅用日频 High/Low 估买卖价差，取全市场中位数（**不需要 tick，很划算**） | ✅ | ★★ |
| E3 | 龙头成交集中度 | 原文档 #3；补 Top10% 个股成交占比 | ✅ | ★★ |
| E4 | 平均单笔成交额 | 成交额 / 成交笔数（散户化程度） | 🔬 | ★ |
| E5 | 停牌股占比 | 停牌家数 / 全市场 | ✅ | ★ |

### F. 日历与供给事件

| # | 特征 | 数据 | 优先级 |
|---|---|---|---|
| F1 | 距春节交易日数、月末/季末/年末哑变量 | ✅ 零成本（`style-dquant/calendar` 已有 `week_of_year`） | ★★ |
| F2 | 财报密集披露期（4 月 / 8 月下旬 / 10 月） | ✅ 零成本 | ★★ |
| F3 | 未来 20 日限售解禁市值 / 流通市值 | ⚠️ | ★★ |
| F4 | IPO 家数 / 募资规模 20 日累计 | ⚠️ | ★ |
| F5 | 指数成分调整生效日 | ⚠️ | ★ |

### G. 宏观与资金价格

| # | 特征 | 数据 | 优先级 |
|---|---|---|---|
| G1 | 10Y 国债收益率及 20 日变化 | ⚠️ | ★★ |
| G2 | 期限利差 10Y−1Y、信用利差 AA−国开 | ⚠️ | ★★ |
| G3 | 股债性价比 ERP = 全A EP(TTM) − 10Y 国债 | ⚠️ | ★★ |
| G4 | DR007 / SHIBOR（资金面松紧） | ⚠️ | ★★ |
| G5 | USDCNH 汇率及波动 | ⚠️ | ★ |
| G6 | 南华工业品 / 原油 / 黄金（避险） | ⚠️ | ★ |
| G7 | 美元指数、CBOE VIX、中美利差 | ⚠️ | ★ |

### H. 杠杆与一致预期（二期）

| # | 特征 | 数据 | 优先级 |
|---|---|---|---|
| H1 | 原文档 #5/#6 两融余额、融资买入占比 | ⚠️ 需建管线 | ★★ |
| H2 | 分析师盈利预测上调家数占比 | ⚠️ | ★ |
| H3 | 评级上下调比、研报覆盖数变化 | ⚠️ | ★ |

---

## 4. 第一批 10 个特征：详细实现规格

**选择原则**：覆盖波动 / 广度 / 状态 / alpha 环境 四个正交轴，**全部只用现有数据，零新建数据管线**，一周内可出 A/B 结论。若这 10 个都没有增量，原文档那 14 个（还要新建两融/期货/期权管线）基本也不必做了。

### 4.1 公共输入

| 别名 | 路径 | 形状 | 说明 |
|---|---|---|---|
| `RET1` | `factors/helpers/ret1_panel.parquet` | (5240, 5516) | 日频后复权收益面板 |
| `MCAP` | `market-data/market_cap/market_cap_panel.parquet` | (5240, 5516) | 总市值（亿元） |
| `AMP` | `factors/helpers/amp_panel.parquet` | — | 日振幅 (H−L)/前收 |
| `INDRET` | `market-data/industry/industry_index_return.parquet` | (5229, 33) | 中信一级行业指数日收益 |
| `PERDAY` | `market-data/daily-dquant/per-day/<YYYY-MM-DD>.parquet` | 逐日 | 含 `total_turnover`，截面聚合首选 |
| `COMBO` | `backtest_engine/cache_dir_dquant/combo_mask_long.parquet` | 17.5M 行 | 字段 `tradable / is_st / is_suspended / is_limit_up` |
| `NEWSTK` | `backtest_engine/cache_dir_dquant/new_stock_mask_long.parquet` | 17.5M 行 | 字段 `is_new_stock` |

### 4.2 统计口径池 `POOL`（全局唯一定义，10 个特征共用）

```
POOL[t, s] = (~is_suspended[t, s]) & (~is_new_stock[t, s])
```

- **不剔 ST**：ST 股本身是市场情绪的一部分，剔掉会损失信息（与原文档口径的分歧点，需和同事确认）。
- **不剔涨跌停**：涨跌停是情绪信号本身。⚠️ 注意**不要用** `limit-dquant/normal_day_panel.parquet`——它把涨跌停也剔掉了，那是给动量因子用的口径，用在这里会系统性削掉极端行情样本。
- 全部统计只在 `POOL` 内做，`nanstd / nanmean / nanmedian` 一律加 `min_count`，当日有效股票数 < 100 的日期直接置 NaN。

### 4.3 输出统一规格

| 项 | 值 |
|---|---|
| 格式 | 宽表 parquet，`index=DatetimeIndex(name='date')`，`columns=股票代码(name='stock')`，`dtype=float32` |
| 值 | **每行所有列同值**（广播） |
| 网格 | 与 `ret1_panel` 完全对齐（同 index、同 columns），未覆盖处 NaN |
| 落盘 | `factors/raw-dquant/market-dquant/<group>/<name>.parquet` |
| 体积 | 实测 `week_of_year.parquet` (5217×5511) 仅 **11 MB**（常数行的 parquet 压缩率极高），10 个 ≈ 110 MB，可接受 |
| 阶段 | **只出 raw，不走 cleaned / neu**（与 `style-dquant/size` 同理：对市场层特征做截面 MAD/中性化是自我抵消，会直接抹成 0） |

**滚动分位统一定义**（记作 `pct500(x)`）：

```
pct500(x)[t] = (# { x[t-499 .. t] <= x[t] } - 1) / (窗口内有效样本数 - 1)
```
纯历史窗口、含当日、不含未来，`min_periods = 250`（不足则 NaN）。

### 4.4 逐特征规格

> 记号：`mkt_ew[t] = nanmean(RET1[t, POOL[t]])`（等权市场收益）；`mkt_vw[t] = Σ(MCAP·RET1) / Σ MCAP`（市值加权）；`NAV = cumprod(1 + mkt_ew)`。

| # | 特征名 | group | 计算式 | 值域/说明 |
|---|---|---|---|---|
| M1 | `mkt_disp20_pct` | `vol` | `d[t] = nanstd(RET1[t, POOL])` → `MA20(d)` → `pct500` | [0,1]，截面离散度 |
| M2 | `mkt_avgcorr20_pct` | `vol` | 见下方公式 → `pct500` | [0,1]，平均相关性 |
| M3 | `mkt_rv20_pct` | `vol` | `RV20 = std(mkt_ew, 20) × √252` → `pct500` | [0,1] |
| M4 | `mkt_rv_ts` | `vol` | `RV20 / RV60 − 1` | 无量纲，>0 = 波动加速 |
| M5 | `ind_disp20_pct` | `vol` | `nanstd(INDRET[t, :])`（33 行业截面）→ `MA20` → `pct500` | [0,1]，行业分化度 |
| M6 | `mkt_dd250` | `regime` | `NAV / rolling_max(NAV, 250) − 1` | (-1, 0] |
| M7 | `mkt_amt_exp` | `regime` | `AMT[t] = Σ_{POOL} total_turnover`；`AMT / MA60(AMT) − 1` | 无量纲 |
| M8 | `mkt_adl_div` | `breadth` | 见下方公式 | z 差值，负 = 广度背离 |
| M9 | `mkt_median_gap20` | `breadth` | `g[t] = median(RET1[t,POOL]) − mkt_vw[t]` → `MA20(g)` | 赚钱效应偏离 |
| M10 | `fac_absic20` | `alphaenv` | 见下方公式 | ≥0，alpha 环境强度 |

**M2 平均相关性**（由等权组合方差反解 implied average correlation）：

```
σ_p  = std(mkt_ew, 20)                      # 等权组合 20 日波动
σ_i  = std(RET1[:, i], 20)  for i in POOL   # 个股 20 日波动
w    = 1/N                                  # 等权
avgcorr[t] = (σ_p² − Σ w²σ_i²) / ((Σ w σ_i)² − Σ w²σ_i²)
```
分母 ≤ 0 或结果落在 [-1, 1] 外时置 NaN。

**M8 ADL 背离**：

```
ad[t]    = (#{RET1[t,POOL] > 0} − #{RET1[t,POOL] < 0}) / #POOL[t]
ADL      = cumsum(ad)
slope[t] = ADL[t] − ADL[t-20]                       # 20 日广度斜率
mom[t]   = NAV[t] / NAV[t-20] − 1                   # 20 日市场收益
div[t]   = zscore500(slope)[t] − zscore500(mom)[t]  # 500 日滚动 z
```
`div < 0` 表示"指数在涨但广度没跟上" = 顶部背离。

**M10 alpha 环境强度**：

```
代表因子集 F（12 个，固定不变，不依赖模型入选结果，保证 A/B 干净）：
  Alpha158: ROC20, STD20, MA20, RSQR20, MAX20, CORR20, VMA20, VSTD20, WVMA20, RANK20
  其他:     ln_market_cap, wgt_return_1m

ic[f, t] = 截面 Spearman( factor_f[t-20, POOL], forward_return_20d[t-20, POOL] )
           # 注意：这是 t-20 日建仓、到 t 日才实现的 IC，故 t 日盘后可知
fac_absic20[t] = mean_f( mean_{最近20个可得交易日}( |ic[f, ·]| ) )   然后再 shift(1)
```

> ⚠️ **M10 是 10 个里唯一有前视风险的**，必须两道保险：(1) `ic[·, t]` 只用 t 日已实现的远期收益；(2) 最后再 `shift(1)` 一天。实现完必须写 `scripts/verify_market_features_pit.py` 做前视专项验证（第 7 节步骤 4）。若验证麻烦，**M10 可以从第一批拿掉，先做 M1–M9**。

### 4.5 生产脚本组织

新建 `data_fetching/market_broadcast.py`，仿 `data_fetching/style_ln_market_cap.py` 的结构（`build(output, full=False)` + `--full` / 增量日更 + `_to_wide` 统一 schema）：

```
data_fetching/market_broadcast.py
  ├── _load_pool()                 -> (T,N) bool，POOL 掩码
  ├── _market_series()             -> DataFrame，一次性算出全部 10 条日频标量序列（T,10）
  ├── _broadcast(s, dates, stocks) -> (T,N) 宽表，s 沿列方向 tile
  └── main()                       -> 逐个落盘到 market-dquant/<group>/<name>.parquet
```

设计要点：
- **10 个特征共用一次 POOL 与截面聚合**，聚合是全部开销所在（约 5200 天 × 5500 股），只扫一遍。预计全量 < 5 分钟。
- 中间的 10 条标量序列另存一份 `factors/helpers/market_broadcast_series.parquet`（T×10 小文件），便于画图、算相关性、做 PIT 审计，不进模型。
- 遵守项目规范：**不用命令行传业务参数**，特征清单/窗口等常量写在脚本顶部常量区，用户手改。仅保留 `--full` / `--output` 两个开关（与 `style_ln_market_cap.py` 一致）。

---

## 5. 模型接入方案

### 5.1 结论：ml_core 零代码改动

`ml_core/features.py::load_factor_grid` 的实现是：

```python
df = pd.read_parquet(path)
arr = df.reindex(index=dates, columns=stocks).to_numpy(dtype=np.float32)
```

只要落盘的就是 T×N 宽表，广播特征与普通因子走完全相同的路径。**已有现成先例佐证**：

```
factors/raw-dquant/style-dquant/calendar/week_of_year.parquet
  shape (5217, 5511) | 单日唯一值数 = 1 | 11 MB
```

这就是一个每日单值广播到全市场的特征，格式已跑通。因此**不需要改 `features.py`、`pipeline.py` 或任何代码**。

> 备选方案（存 T×1 再在 `load_factor_grid` 里广播）可省 90% 磁盘，但要改核心读取函数、且 train/live 两条路径都得同步改，**风险收益比不划算，不采用**。110 MB 磁盘不是问题。

### 5.2 `train_config.yaml` 怎么改

对 `lgbm_shap128_a158_size_kymom_htmom_csrank5_dq`，**推荐用 `feature_set.inherit_from` 做干净的增量 A/B**，而不是重新跑 SHAP：

```yaml
run_id: lgbm_shap128_a158_size_kymom_htmom_mkt10_csrank5_dq   # 基线 run_id + _mkt10

sources:
  - alpha158-dquant
  - style-dquant/size
  - kysec-dquant/paper_67_long_momentum
  - htsec/paper_04_momentum
  - market-dquant            # ← 新增；也可细到 market-dquant/vol 做单维度消融

select_method: null          # ← 关键：关掉 SHAP 重选，否则新旧模型差异不只来自新因子
# top_k: 128                 # select_method=null 时忽略

feature_set:
  inherit_from: lgbm_shap128_a158_size_kymom_htmom_csrank5_dq   # 继承基线入选的 128 个
  append:
    - mkt_disp20_pct
    - mkt_avgcorr20_pct
    - mkt_rv20_pct
    - mkt_rv_ts
    - ind_disp20_pct
    - mkt_dd250
    - mkt_amt_exp
    - mkt_adl_div
    - mkt_median_gap20
    - fac_absic20
```

这样得到 128 + 10 = 138 个特征，**旧的 128 个一字不动**，新旧模型的全部差异只来自这 10 个广播特征——这是最干净的 A/B。（此机制在 `train_config.yaml` 第 35–48 行有说明，之前跑 dongwu A/B 时已用过。）

其余（`label`、`split`、`horizon`、`predict`、`lgbm` 超参）**全部保持不变**。

### 5.3 建议的实验矩阵

| 实验 | sources 追加 | append 特征 | 目的 |
|---|---|---|---|
| **B0** | — | — | 基线（已有，直接复用结果） |
| **B1** | `market-dquant/vol` | M1–M5 | 波动维度单独增量（同事关心的方向） |
| **B2** | `market-dquant/breadth` + `/regime` | M6–M9 | 广度+状态维度单独增量 |
| **B3** | `market-dquant/alphaenv` | M10 | 因子动量单独增量 |
| **B4** | `market-dquant` | M1–M10 | 全量 |
| **B5** | `market-dquant` | M1–M10，但 `select_method: shap`, `top_k: 128` | 看 SHAP 会不会主动选广播特征、挤掉谁 |

先跑 **B4 vs B0**：全量都没增量就直接停，不用跑 B1–B3。有增量再用 B1–B3 定位贡献来源，B5 看它在自由竞争下的地位。

### 5.4 落盘产物

沿用现有惯例，无需新增目录：

```
ml/models/<run_id>/       模型 + scaler_x.parquet + selected_features.json + run_meta.json
ml/predictions/<run_id>/  pred_panel_live.parquet + signals/
ml_core/logs/<run_id>_<时间戳>.log
```

回测走 `daily-realtime-backtest-pipeline`，改 `config/config.yaml` 指到新 `run_id`。

---

## 6. 关键工程约束与坑（动手前必读）

### 6.1 【硬约束】广播特征只能用于 LGBM 线，MLP 线会被静默抹成 0

`ml_core/scaling.py` 有两个标准化器：

| 标准化器 | 用于 | 对广播特征的影响 |
|---|---|---|
| `WholeSetRobustZ` | **LGBM** | ✅ 安全。全集 per-feature 中位数/MAD，广播特征随时间变化 → MAD > 0 → 正常缩放 |
| `DailyCrossSectionMAD` | **MLP** | ❌ **致命**。逐日截面 zscore，广播特征当日 σ=0 → `sigma = where(<1e-8, 1.0)` 且 `day − mu = 0` → **整列恒为 0，且不报错** |

> `WholeSetRobustZ.fit` 第 63 行 `scale_ = np.where(mad > 0, mad, np.nan)`：只有**全时段恒定**的列才会变 NaN。我们的广播特征随时间变化，安全。但**若某特征在训练段（≤2017-11）内恒定或全 NaN，该列会全程失效**——这正是 iVIX 那类"训练段有值/测试段没值"或反之的特征会踩的坑，落盘后必须检查训练段方差。

**行动项**：在 `market_broadcast.py` 落盘时加断言——训练段（≤2017-11-30）内 `nunique() > 50` 且 NaN 占比 < 20%，否则 raise。同时在文档和 `train_config.yaml` 注释里写明"`model: mlp` 时不得启用 `market-dquant` 源"。

### 6.2 广播特征的 SHAP 重要度会系统性虚高

每个广播列在长表里被复制约 5000 倍，LGBM 看到的是 1200 万样本而非 2400 个独立观测。**不要把广播特征和截面因子放在同一张 SHAP 表里排序**，那是不可比的。

评估广播特征只看三件事：**(1) 测试段回测的累计收益 / Calmar / 最大回撤；(2) 分年度 IC 符号与稳定性；(3) 剔除该组后的性能下降幅度（消融）。**

### 6.3 IC 涨不等于回测涨 —— 以回测为准

dongwu 那轮的教训（§1.5）：E1 的 IC 从 0.1390 涨到 0.1395，回测反而从 596% 掉到 587%；E2 的 ICIR 从 1.119 涨到 1.136（三者最高），回测掉到 527%（全部模型第 13）。

**2400 个独立观测下，IC 的万分之几差异完全在噪声内。** 验收标准必须是回测指标，且要看分年度稳定性，不能只看一个总数。

### 6.4 有效样本量问题 → 验证协议要升级

原文档 §验证协议第 3 步（同一切分跑 A/B/C）对广播特征不够。建议追加：

- **分年度 IC 符号表**（2020–2026 逐年），看新增特征是否只在某一两年生效
- **block bootstrap**：以月为块重采样，给回测收益差一个置信区间，判断 B4 − B0 的差异是否显著
- **重点检视段**：2024/01–02（微盘踩踏）、2024/09–12（政策急转）、2026 YTD。这几段恰是上一轮分析中 p27 微观结构因子体现防御价值的时期，广播特征若有用，应该在这些段最明显。

### 6.5 与已有特征的相关性

落盘后先算这 10 个与 Alpha158 中时序类因子（STD20/60、CORR20、RSQR20 等）的**时序相关性**（不是截面）。Alpha158 的个股波动因子在截面平均后其实携带了部分市场波动信息，可能与 M1/M3 高度相关。`|ρ| > 0.8` 的对子考虑二选一。

### 6.6 数据末日不齐

各源末日不同：`ret1_panel` 到 2026-07-31、`turnover` 到 2026-07-31、`industry_index_return` 到 2026-07-17、`000985_segments` 只到 2026-06-05 且 2011 年才开始。M5 用的 `INDRET` 会成为**共同覆盖末日的瓶颈**（2026-07-17）。`ml_core` 的 `predict.end: null` 会自动取"入选因子共同覆盖末日"，加入 M5 后推理区间会缩短约两周。**接受，但要知道**；或把 `industry_index_dquant.py` 先日更到最新。

---

## 7. 实施步骤（按序）

| 步骤 | 内容 | 产物 | 预估 |
|---|---|---|---|
| 0 | 与同事确认：POOL 是否剔 ST、iVIX 口径、M10 是否纳入第一批 | 口径共识 | — |
| 1 | 写 `data_fetching/market_broadcast.py`，产出 10 条标量序列 | `factors/helpers/market_broadcast_series.parquet` (T×10) | 半天 |
| 2 | 序列体检：画图 + 描述统计 + 两两相关 + 与 Alpha158 时序相关（§6.5） | 一张图 + 相关矩阵 | 1 小时 |
| 3 | 广播落盘 + 训练段方差断言（§6.1） | `market-dquant/{vol,breadth,regime,alphaenv}/*.parquet` | 10 分钟 |
| 4 | **PIT 前视专项验证** `scripts/verify_market_features_pit.py`：逐特征断言 `t` 日值只依赖 `≤t` 的数据（重算截断序列比对），M10 重点查 | 验证脚本 + 通过日志 | 半天 |
| 5 | 跑 **B4**（全量 10 个，`inherit_from` 增量）| `ml/models/..._mkt10_...` | 视训练时长 |
| 6 | 回测 B4 vs B0：累计收益 / Calmar / 最大回撤 / 分年度 | 对比表 | 1 小时 |
| 7 | 有增量 → 跑 B1/B2/B3 定位来源 + B5；无增量 → 停，结论写回本文档 | 结论 | — |

**Go / No-Go 判据（步骤 6）**：B4 相对 B0，测试段累计收益提升 **> 3 个百分点** 且最大回撤不恶化，才继续第二批；否则判定"广播特征进截面模型"这条路不通，转 §8 的替代架构。

---

## 8. 替代架构：广播特征不进截面模型

如果第 7 节 Go/No-Go 不通过，**不代表这些特征没价值，而是入口选错了**。

更稳妥的架构是把市场层信息放在**组合层**做仓位/暴露调节，而不是混进 alpha 模型：

- 截面模型只负责**排序**（它擅长的事），市场状态负责**仓位与换手**（择时）
- 例如：`mkt_rv20_pct > 0.9 且 mkt_disp20_pct < 0.2` 时降仓 / 降换手 / 收紧持仓集中度
- 优势：不消耗截面模型的自由度、规则可解释、失败可即时回滚、不污染已验证的 alpha 信号

**如果时间有限、只能选一条路，我倾向先试组合层。** 它的下行风险明显更小，而截面模型这条路的过拟合风险（§1.1、§6.2）是结构性的。

---

## 附：待确认清单

| # | 事项 | 找谁 |
|---|---|---|
| 1 | iVIX 2019 年后已停发，原文档 #14 口径需重写（自算 VIX / GARCH 代理 / 放弃） | 同事 |
| 2 | POOL 是否剔除 ST（本文建议不剔，与原文档"正常交易股票"口径有分歧） | 同事 |
| 3 | 原文档 #12 破净占比：`fundamentals-dquant` 内有无 `pb_ratio_lf` | 自查 |
| 4 | 原文档 #7 自然涨跌停：`combo_mask_long` 无 `is_limit_down`，跌停与一字板判定需自算 | 自查 |
| 5 | 两融 / 期货 / 期权数据是否值得新建管线（建议等第一批 A/B 结论出来再决定） | 同事 |
| 6 | 原文档 #12 光大策略报告全名（原文档自己标了 ⚠️ 待补全） | 同事 |
| 7 | M10 `fac_absic20` 是否纳入第一批（前视验证成本最高） | 决策 |
