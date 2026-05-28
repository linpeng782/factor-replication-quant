net_oper_cash_flow_ttm：经营活动产生的现金流量净额 TTM

直接取米筐 `cash_flow_from_operating_activities_ttm_0`（最近报告期为终点的 4 季度滚动）作为因子值。
属于质量类因子，方向 +1（现金流越大公司质量越好）。

注：原始 TTM 量纲与市值高度相关；如需中性化版本可后续做 `..._to_mc` 衍生因子。
本因子按总表"现金流 TTM 值"原义实现，不做缩放。
