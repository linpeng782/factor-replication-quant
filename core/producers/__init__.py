"""代码批量型因子生产者（区别于 spec/yolo 研报型）。

每个子包（alpha158 / 未来 mars / alpha191）= 一个 qlib 风格的整表宽算子因子库：
读后复权宽表面板 → 一次性向量化算出全部因子 → 落 factors/raw/<source>/<group>/。
与 sources/ + yolo_engine 的逐因子 spec 路径并行、互不干扰。
"""
