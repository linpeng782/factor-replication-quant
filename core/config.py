"""兼容垫片：config.py 已上移到项目根（业务身份是项目级共享配置，非 core 专属）。

历史上本文件在 core/ 下，导致 `from core import config` 假象——好像 config 从属于
因子复现引擎。实际它服务所有业务域（core/ml/ml_core/ml_ht/...），故上移到项目根。

新代码请直接 `import config` / `from config import X`。
本垫片仅为过渡期保留旧 `from core import config` / `from core.config import X` 不报错。
"""

from config import *  # noqa: F401,F403
