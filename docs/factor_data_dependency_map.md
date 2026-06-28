# 因子 ↔ 数据 依赖反查表

> 配套文档：数据集编号 ①~⑩ 与字段口径见 `docs/ricequant_data_inventory.md`。
> 本表用于判断**关键路径 / 分批上线优先级**，以及核对"漏了哪个数据会断哪些因子"。
>
> - **关键路径** = 当前生产模型 `cxl_a158_p27_raw_shap_v2` 实际用到的源（= **alpha158 + cxl + 微结构 paper_27**）。
> - **候选** = 已复刻、在仓库里、模型可选但当前生产模型未选的因子族（可后批上）。

数据集编号速查：① 股票池 · ② 日频OHLCV · ③ 分钟OHLCV · ④ 复权因子 · ⑤ 总市值 · ⑥ 中信一级行业 · ⑦ 中信行业指数 · ⑧ 000985分钟 · ⑨ 基本面PIT财务 · ⑩ 交易日历

---

## 1. 数据集 → 消费它的因子族

| 数据集 | 消费的因子族（因子数）| 是否关键路径 |
|---|---|---|
| ① 股票池 | **所有**因子 + 所有数据线 | ✅ 关键（地基）|
| ② 日频OHLCV | **alpha158**(158) · **收益标签** · ret20面板(→apm) | ✅ 关键 |
| ③ 分钟OHLCV | paper_27(23✅) · pricejump(18) · apm(5) · smartmoney(2) · tide(5) · dazzle(7) | ✅ 关键（生产仅 paper_27）|
| ④ 复权因子 | alpha158 + **全部**分钟因子（读时复权）| ✅ 关键 |
| ⑤ 总市值 market_cap_3 | cxl/pe_series · cxl/cross_section_regress · **因子中性化(neu)** | ✅ 关键 |
| ⑥ 中信一级行业 | **因子中性化(neu)** · co_momentum(候选) | ✅ 关键 |
| ⑦ 中信行业指数 | guosen/co_momentum(5) | ⬜ 候选 |
| ⑧ 中证全指000985分钟 | kysec/paper_05_apm(5) | ⬜ 候选 |
| ⑨ 基本面PIT财务 | **cxl 全部 22 个**（字段级见第 3 节）| ✅ 关键 |
| ⑩ 交易日历 | 所有增量调度 | ✅ 关键 |

---

## 2. 因子族 → 数据依赖（按 source/group）

| 因子族 (source/group) | 因子数 | 数据依赖 | 生产模型 |
|---|---|---|---|
| `alpha158`（无 spec，独立脚本）| 158 | ②日频OHLCV + ④复权 | ✅ |
| `kysec/paper_27_microstructure` | 23 | ③分钟(superset=prv_v3) + ④复权 | ✅ |
| `cxl/npf_series` | 8 | ⑨ net_profit_mrq_0~8 | ✅ |
| `cxl/roe_series` | 5 | ⑨ net_profit_mrq_0/1/4 + total_equity_mrq_0/1/4 | ✅ |
| `cxl/pe_series` | 3 | ⑨ market_cap_3 + net_profit_mrq_0 + pe_ratio_ttm | ✅ |
| `cxl/roic_series` | 3 | ⑨ return_on_invested_capital_ttm | ✅ |
| `cxl/cross_section_regress` | 2 | ⑨ market_cap_3 / net_profit_mrq_0/_4 / net_profit_ttm_0 / pb_ratio_lf / pe_ratio_ttm | ✅ |
| `cxl/cashflow_series` | 1 | ⑨ cash_flow_from_operating_activities_ttm_0 | ✅ |
| `kysec/paper_33_pricejump` | 18 | ③分钟(superset=pjr_v1) + ④复权 | ⬜ 候选 |
| `founder/paper_01_moderate_risk` | 7 | ③分钟(superset=dazzle_v1) + ④复权 | ⬜ 候选 |
| `founder/paper_02_tide` | 5 | ③分钟(superset=tide_v1) + ④复权 | ⬜ 候选 |
| `guosen/co_momentum` | 5 | ⑦行业指数 + ⑥行业 + ②日频 | ⬜ 候选 |
| `kysec/paper_05_apm` | 5 | ③分钟(superset=apm_v1) + ⑧000985分钟 + ②日频(ret20) | ⬜ 候选 |
| `kysec/paper_03_smartmoney` | 2 | ③分钟(superset=sm_v1, sm_v1_b05) + ④复权 | ⬜ 候选 |

> 说明：分钟因子都先把分钟数据归约成「superset」中间层（每族一个 cache_key），再算因子。
> 生产模型只用 `prv_v3`(→paper_27)；其余 superset 是候选因子族用的。

---

## 3. 基本面字段 → cxl 因子族（⑨ 的字段级反查，便于分批 / 核对覆盖）

| 字段 | 被哪些 cxl 族用 |
|---|---|
| `net_profit_mrq_0` ~ `net_profit_mrq_8` | npf_series（8 期增速 / SUE）|
| `net_profit_mrq_0/1/4` | roe_series、pe_series、cross_section_regress |
| `net_profit_ttm_0` | cross_section_regress |
| `total_equity_mrq_0/1/4` | roe_series（ROE = 净利 / 净资产）|
| `return_on_invested_capital_ttm` | roic_series |
| `cash_flow_from_operating_activities_ttm_0` | cashflow_series |
| `market_cap_3` | pe_series、cross_section_regress、**中性化** |
| `pe_ratio_ttm` | pe_series、cross_section_regress |
| `pb_ratio_lf` | cross_section_regress |

---

## 4. 上线优先级建议

1. **第一批（关键路径，生产模型直接依赖）**
   ① 股票池 · ⑩ 交易日历 · ② 日频OHLCV · ④ 复权因子 · ③ 分钟OHLCV（至少 prv_v3 这条）· ⑨ 基本面 18 字段 · ⑤ 市值 · ⑥ 行业。
   → 齐了即可重建生产模型 `cxl_a158_p27_raw_shap_v2` 的全部输入。

2. **第二批（候选因子，扩模型时再上）**
   ⑦ 行业指数 · ⑧ 000985 分钟；分钟③扩到其余 superset（apm / sm / tide / dazzle / pricejump）。

> 口径校验来源：以上映射由扫描 `sources/*/*/specs/*/spec.yaml` 的数据依赖（superset cache_key、get_factor 字段、算子）自动得出，
> 截至 2026-06-17 仓库状态。
