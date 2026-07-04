"""
ml_core —— ml/ 与 ml_ht/ 的共享管线内核（模型无关）
============================================================
设计原则（与用户讨论敲定）：
  1. 管线一致、只换模型：把两条线逐字节等价的环节上提到此包，
     ml/(LightGBM) 与 ml_ht/(PyTorch MLP) 退化为"配置模型 + 选策略"的薄入口。
  2. 底座 vs 动态切分：
       - 底座（universe + can_buy + has_label）模型/因子均无关 → ml_core.universe
       - has_factor 随因子集 + 模型完整性策略变化 → 归 features 层动态现算（不物化）
  3. 词汇统一：用 has_factor / can_buy / has_label 取代旧 can_buy_mask / not_limit_up_mask
       can_buy   ≡ 旧 can_buy_mask = NOT(st|suspended|new)@T+1   （模型/因子无关）
       has_label = forward_return_Nd 非 NaN                  （模型/因子无关）
       has_factor= 因子完整性（MLP=全非NaN / LGBM=trivially-true）（模型策略驱动）
  4. 公共网格（列空间）= dquant instruments 全集（每日全市场的并集，含退市股，
     但退市股被三谓词自动滤掉、永不成样本）。
  5. 不强行统一"标准化"与"标签变换"——它们是模型适配的刻意选择，做成可插拔策略：
       标准化   WholeSetRobustZ(LGBM, 持久化) | DailyCrossSectionMAD(MLP, 无状态)
       标签     ExcessReturn(LGBM, 回归)       | BinaryMedian(MLP, 二分类)

规划中的模块布局（逐步落地，本次先落 universe）：
  universe.py   ✅ 底座层：build_universe() → (universe, can_buy, has_label)
  features.py   ⏳ 统一因子读取（多源 discover + load_factor_grid）+ has_factor 策略
  labels.py     ⏳ LabelTransform: ExcessReturn | BinaryMedian
  scaling.py    ⏳ Standardizer: WholeSetRobustZ | DailyCrossSectionMAD
  splits.py     ⏳ 时间切分 + embargo（统一代码，数值各入口保留）
  metrics.py    ⏳ IC / ICIR / 多空 / 逐年（合并 ml.evaluate + ml_ht.metrics）
  signals.py    ⏳ append-only 每日 top-N 信号导出
  model.py      ⏳ ModelAdapter 接口：fit / predict / save / load

迁移纪律：纯重构（零漂移）→ 砍长表（验证迁移）→ 统一数值/做因子；
旧 ml/、ml_ht/ 在对齐验证通过前一字不动，新产物走隔离命名空间。
"""
