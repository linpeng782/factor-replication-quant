"""
roic_ttm_ind_rnk8 端到端冒烟（已弃用，留作历史记录）。

这个文件是早期手写的 3 只股票端到端测试，使用旧版算子契约（`input`/`output`/
`method: industry_rank` 等），新算子 API 已不再支持。整体 pipeline 的回归覆盖
已迁移到 `scripts/regression_check.py`（peak_interval_kurt + npf_mrq_sue8 全市场
bit-exact 对比），比 3 只股票冒烟严格得多。

如要恢复，需重写为新契约（spec.yaml 形式）+ 用 yolo_engine.run() 跑端到端，
不要再手工拼算子。
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="使用旧版算子 API；端到端覆盖已由 scripts/regression_check.py 接管"
)


def test_full_pipeline():
    pass
