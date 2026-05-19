"""
YOLO 执行层：读取 Spec YAML → 全自动执行数据获取 + 因子计算

核心设计：计算图执行引擎
- 不针对每个因子硬编码，而是根据 YAML 的 calculation_steps 动态执行
- 预定义「元操作」库（fetch, compute, filter, rank, transform 等）
- 支持依赖关系自动解析，按拓扑顺序执行

输出：
- output/<factor_name>/raw_<factor_name>.parquet
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

warnings.filterwarnings("ignore")

from core.config import FACTOR_OUTPUT_DIR as OUTPUT_DIR

# 导入 operators 触发注册
from .operators import OpRegistry
from .operators import fetch  # noqa: F401
from .operators import compute  # noqa: F401
from .operators import filter  # noqa: F401
from .operators import rank  # noqa: F401
from .operators import transform  # noqa: F401


# ──────────────────────────────────────────
# 数据获取层
# ──────────────────────────────────────────


class DataFetcher:
    """封装米筐 RQData API"""

    def __init__(self):
        self._rq = None
        self._inited = False

    def _init_rq(self):
        if self._inited:
            return
        try:
            import rqdatac as rq
            self._rq = rq
            try:
                rq.init()
                self._inited = True
            except Exception as e:
                print(f"⚠️ rqdatac.init() 失败: {e}")
        except ImportError:
            raise ImportError("使用 YOLO 引擎需要安装 rqdatac")

    def fetch_pit(self, order_book_ids: List[str], fields: List[str], start_quarter: str, end_quarter: str, statements: str = "latest") -> pd.DataFrame:
        """获取 PIT 财务数据"""
        self._init_rq()
        return self._rq.get_pit_financials_ex(
            order_book_ids=order_book_ids, fields=fields,
            start_quarter=start_quarter, end_quarter=end_quarter, statements=statements,
        )

    def get_index_components(self, index_code: str, date: str) -> List[str]:
        """获取指数成分股"""
        self._init_rq()
        return self._rq.index_components(index_code, date=date)

    def get_factor(self, order_book_ids: List[str], field: str, date: str = None, start_date: str = None, end_date: str = None):
        """获取某个因子/指标值。支持单日(date)或日期范围(start_date+end_date)。"""
        self._init_rq()
        if date:
            return self._rq.get_factor(order_book_ids, field, date=date)
        return self._rq.get_factor(order_book_ids, field, start_date=start_date, end_date=end_date)

    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表"""
        self._init_rq()
        return self._rq.get_trading_dates(start_date, end_date)

    def all_instruments(self, type_: str = "CS") -> pd.DataFrame:
        """获取全部股票列表"""
        self._init_rq()
        return self._rq.all_instruments(type=type_)


# ──────────────────────────────────────────
# 股票池构建
# ──────────────────────────────────────────


def build_universe(universe_cfg: Dict, trade_date: str, fetcher: DataFetcher) -> List[str]:
    """根据 universe 配置构建股票池。排除/过滤由回测引擎处理，此处只负责拿列表。"""
    primary = universe_cfg.get("primary_index", "000906.XSHG")
    if primary == "ALL":
        return fetcher.all_instruments(type_="CS")["order_book_id"].tolist()
    return fetcher.get_index_components(primary, trade_date)


# ──────────────────────────────────────────
# 主执行引擎
# ──────────────────────────────────────────


class YoloEngine:
    """YOLO 执行引擎：读取 Spec YAML，全自动执行计算步骤"""

    def __init__(self):
        self.fetcher = DataFetcher()
        self.ctx = {}

    def run(
        self,
        spec_yaml: dict,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        trade_date: Optional[str] = None,
    ) -> pd.DataFrame:
        factor_name = spec_yaml["factor"]["name"]
        print(f"\n🚀 YOLO 执行: {factor_name}")
        print("=" * 60)

        self.ctx = {"_factor_name": factor_name}
        if trade_date:
            self.ctx["_trade_date"] = trade_date

        # 1. 确定股票池
        universe_cfg = spec_yaml.get("universe", {})
        pool_date = trade_date or start_date
        if pool_date:
            stocks = build_universe(universe_cfg, pool_date, self.fetcher)
            self.ctx["_universe"] = stocks
            print(f"   股票池: {len(stocks)} 只")

        # 2. 设置日期范围
        if start_date and end_date:
            self.ctx["_start_date"] = start_date
            self.ctx["_end_date"] = end_date
            self.ctx["_start_quarter"] = f"{start_date[:4]}q1"
            self.ctx["_end_quarter"] = f"{end_date[:4]}q4"

        # 3. 按顺序执行 calculation_steps
        steps = spec_yaml.get("calculation_steps", [])
        for step in steps:
            action = step.get("action", "")
            step_name = step.get("name", f"Step {step.get('step', '?')}")
            print(f"   ▶ {step_name} [action={action}]")

            op_func = OpRegistry.get(action)
            result = op_func(self.ctx, step, self.fetcher)

            if isinstance(result, pd.DataFrame):
                print(f"     → DataFrame shape: {result.shape}, columns: {list(result.columns)}")

        # 4. 提取最终因子值
        final_output = steps[-1]["output"] if steps else "factor"
        factor_df = self.ctx.get(final_output)

        if factor_df is None or not isinstance(factor_df, pd.DataFrame):
            raise ValueError(f"最终输出未找到或不是 DataFrame: {final_output}")

        # 5. 转为宽表 (date × order_book_id) 并保存
        factor_dir = OUTPUT_DIR / factor_name
        factor_dir.mkdir(parents=True, exist_ok=True)

        if "date" in factor_df.columns and "order_book_id" in factor_df.columns:
            wide_df = factor_df.pivot(index="date", columns="order_book_id", values=factor_name)
            wide_df.index = pd.to_datetime(wide_df.index)
            output_path = factor_dir / f"raw_{factor_name}.parquet"
            wide_df.to_parquet(output_path)
            print(f"\n✅ 宽表已保存: {output_path} (shape={wide_df.shape})")
            return wide_df
        else:
            output_path = factor_dir / f"raw_{factor_name}.parquet"
            factor_df.to_parquet(output_path)
            print(f"\n✅ 因子值已保存: {output_path}")
            return factor_df


# ── 便捷入口 ───────────────────────────────


def run_factor(factor_name: str, spec_yaml: Optional[dict] = None, **kwargs) -> pd.DataFrame:
    """便捷函数：给定因子名，自动读取 YAML 并执行"""
    if spec_yaml is None:
        spec_path = Path(__file__).parent.parent / "specs" / factor_name / "spec.yaml"
        with open(spec_path, "r", encoding="utf-8") as f:
            spec_yaml = yaml.safe_load(f)

    engine = YoloEngine()
    return engine.run(spec_yaml, **kwargs)
