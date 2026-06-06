"""
刷新所有【在用】的 minute superset 缓存（日更第6步独立入口）。
============================================================
spec = "哪些 superset 在用" 的唯一真相源：
  扫 sources/*/*/specs/*/spec.yaml → 找 action∈REDUCER_BY_ACTION 的步
  → 去重 unique (reducer 实例 = action+cache_key+params，按 cache_dir 唯一)
  → 每个 MinuteAggregateEngine(reducer).refresh_cache(universe)（append-only 增量）。

用法:
  python pipeline/refresh_supersets.py                       # 增量刷新所有在用 superset
  python pipeline/refresh_supersets.py --dry-run             # 只列出，不刷新（别名 --list）
  python pipeline/refresh_supersets.py --cache-key sm_v1     # 只刷指定 cache_key（可逗号分隔多个）
  python pipeline/refresh_supersets.py --cache-key sm_v1 --rebuild  # 删缓存后强制全量重建
"""
from __future__ import annotations

import argparse
import glob
import shutil

import yaml
from loguru import logger

import core.yolo_engine  # noqa: F401 触发所有 reducer 注册
from core.operators.minute_engine import REDUCER_BY_ACTION, MinuteAggregateEngine

ACCOUNTS = ("13522652015", "123456")
SPEC_GLOB = "sources/*/*/specs/*/spec.yaml"


def collect_active_reducers():
    """扫所有 spec → 去重的 reducer 实例（按 cache_dir 唯一）。返回 [(reducer, action, [来源spec名])]。"""
    by_dir: dict = {}  # cache_dir_str -> (reducer, action, [spec_paths])
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


def _print_configs(configs: list) -> None:
    logger.info(f"扫到 {len(configs)} 个在用 superset（按 cache_dir 去重）:")
    for reducer, action, specs in configs:
        cdir = reducer.cache_dir()
        exists = cdir.exists()
        status = "✅已建" if exists else "❌未建"
        logger.info(
            f"  {status}  action={action}  cache_key={reducer.cache_key}  "
            f"warmup={reducer.warmup}  params={reducer.params}"
            f"\n         → {cdir.name}  ({len(specs)} 个 spec 引用)"
        )


def main():
    ap = argparse.ArgumentParser(
        description="刷新所有在用 minute superset 缓存",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--dry-run", "--list",
        dest="dry_run", action="store_true",
        help="只列出在用 superset 及构建状态，不刷新（--list 是别名）",
    )
    ap.add_argument(
        "--cache-key", dest="cache_key", default=None,
        metavar="KEY[,KEY2,...]",
        help="只刷指定 cache_key（逗号分隔多个，默认全部）",
    )
    ap.add_argument(
        "--rebuild", action="store_true",
        help="强制全量重建：删除指定 superset 缓存目录后从头重算（必须配合 --cache-key 使用，防误删）",
    )
    a = ap.parse_args()

    # ── 安全校验 ──
    if a.rebuild and not a.cache_key:
        ap.error("--rebuild 必须配合 --cache-key 使用（拒绝全量误删所有缓存）")

    all_configs = collect_active_reducers()
    _print_configs(all_configs)

    # ── 过滤 ──
    if a.cache_key:
        keys = {k.strip() for k in a.cache_key.split(",")}
        configs = [(r, act, sp) for r, act, sp in all_configs if r.cache_key in keys]
        missing = keys - {r.cache_key for r, _, _ in configs}
        if missing:
            logger.warning(f"以下 cache_key 在所有 spec 中找不到对应的 step: {missing}")
        if not configs:
            logger.error("过滤后无匹配 superset，退出。")
            return
        logger.info(f"--cache-key 过滤后剩 {len(configs)} 个 superset: {[r.cache_key for r,_,_ in configs]}")
    else:
        configs = all_configs

    if a.dry_run:
        logger.info("--dry-run：不刷新，退出。")
        return

    # ── rebuild：删缓存目录 ──
    if a.rebuild:
        for reducer, action, _ in configs:
            cdir = reducer.cache_dir()
            if cdir.exists():
                logger.warning(f"  --rebuild: 删除 {cdir.name}  ({sum(1 for _ in cdir.iterdir())} 文件)")
                shutil.rmtree(cdir)
            else:
                logger.info(f"  --rebuild: {cdir.name} 不存在，跳过删除")

    # ── 刷新 ──
    import rqdatac
    rqdatac.init(*ACCOUNTS)
    universe = sorted(rqdatac.all_instruments(type="CS")["order_book_id"].tolist())
    logger.info(f"universe(all_instruments CS) = {len(universe)} 只；开始逐个刷新…")

    for reducer, action, _ in configs:
        logger.info(f"▶ 刷新 {reducer.cache_key}  (action={action})")
        MinuteAggregateEngine(reducer).refresh_cache(universe)

    logger.success(f"全部 {len(configs)} 个 superset 刷新完成")


if __name__ == "__main__":
    main()
