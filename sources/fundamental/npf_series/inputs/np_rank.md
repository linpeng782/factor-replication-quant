np_rank：最近一期单季度净利润在最近 8 期单季度净利润中的排名

每个 (股票, 日期) 行向计算：当期单季度净利润 net_profit_mrq_0 在 {net_profit_mrq_0, ..., net_profit_mrq_7} 中的"min"排名（最小值=1，最大值=8）。
排名越高表示当期净利润越接近 8 期内的高点，景气度越好；因子方向 +1。

实现技巧：行向 rank 用 `1 + sum(net_profit_mrq_0 > net_profit_mrq_i for i in 1..7)` 等价于 pandas rank(method='min', ascending=True)。
NaN 守门：把 `(net_profit_mrq_0 - net_profit_mrq_0)` 加进求和起点，使得 mrq_0 缺失时整体为 NaN。

属于景气类因子。
