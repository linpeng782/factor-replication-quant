"""
刷新所有【在用】的 minute superset 缓存（日更第6步独立入口）。
============================================================
spec = "哪些 superset 在用" 的唯一真相源：
  扫 sources/*/*/specs/*/spec.yaml → 找 action∈REDUCER_BY_ACTION 的步
  → 去重 unique (reducer 实例 = action+cache_key+params，按 cache_dir 唯一)
  → 每个 MinuteAggregateEngine(reducer).refresh_cache(universe)（append-only 增量）。

用法:
  python pipeline/refresh_supersets.py --dry-run   # 只列出在用 superset，不刷新
  python pipeline/refresh_supersets.py             # 刷新所有 superset 到 raw 最新日
"""
from __future__ import annotations

import argparse
import glob

import yaml
from loguru import logger

import core.yolo_engine  # noqa: F401 触发所有 reducer 注册
from core.operators.minute_engine import REDUCER_BY_ACTION, MinuteAggregateEngine

ACCOUNTS = ("13522652015", "123456")
SPEC_GLOB = "sources/*/*/specs/*/spec.yaml"


def collect_active_reducers():
    """扫所有 spec → 去重的 reducer 实例（按 cache_dir 唯一）。返回 [(reducer, [来源spec名])]。"""
    by_dir = {}  # cache_dir -> (reducer, action, [spec_paths])
    for p in sorted(glob.glob(SPEC_GLOB)):
        try:
            spec = yaml.safe_load(open(p, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"跳过无法解析的 spec {p}: {e}")
            continue
        for step in (spec.get("calculation_steps") or []):
            action = step.get("action")
            if action in REDUCER_BY_ACTION:
                reducer = REDUCER_BY_ACTION[action].from_step(step)
                key = str(reducer.cache_dir())
                if key not in by_dir:
                    by_dir[key] = (reducer, action, [])
                by_dir[key][2].append(p)
    return list(by_dir.values())


def main():
    ap = argparse.ArgumentParser(description="刷新所有在用 minute superset 缓存")
    ap.add_argument("--dry-run", action="store_true", help="只列出在用 superset，不刷新")
    a = ap.parse_args()

    configs = collect_active_reducers()
    logger.info(f"扫到 {len(configs)} 个在用 superset（按 cache_dir 去重）:")
    for reducer, action, specs in configs:
        logger.info(
            f"  action={action} cache_key={reducer.cache_key} warmup={reducer.warmup} "
            f"params={reducer.params} → {reducer.cache_dir().name} | {len(specs)} 个 spec 引用"
        )
    if a.dry_run:
        logger.info("--dry-run：不刷新，退出。")
        return

    import rqdatac
    rqdatac.init(*ACCOUNTS)
    universe = sorted(rqdatac.all_instruments(type="CS")["order_book_id"].tolist())
    logger.info(f"universe(all_instruments CS) = {len(universe)} 只；开始逐个刷新…")
    for reducer, action, _ in configs:
        logger.info(f"▶ 刷新 {reducer.cache_key} ({action})")
        MinuteAggregateEngine(reducer).refresh_cache(universe)
    logger.success(f"全部 {len(configs)} 个 superset 刷新完成")


if __name__ == "__main__":
    main()
