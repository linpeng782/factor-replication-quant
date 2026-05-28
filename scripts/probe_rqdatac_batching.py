"""
米筐 get_factor "分批并行 vs 一次性" A/B 探针。

用法：
    python probe_rqdatac_batching.py
    python probe_rqdatac_batching.py --field pb_ratio_lf
    python probe_rqdatac_batching.py --field net_cash_flow_growth_ratio_ttm --batches 12 --workers 12

依赖：rqdatac（已 init 过）。脚本自身不依赖项目其它代码。
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_a_parallel(rq, stocks, field, start, end, batch_size, workers):
    """实验 A：股票分批 + 多线程并行；每线程拉一批 × 单字段。"""
    batches = [stocks[i : i + batch_size] for i in range(0, len(stocks), batch_size)]
    log(f"实验 A: {field} | {len(batches)} 批 × {batch_size} 股 × {workers} 线程并行")

    def fetch_one(idx, batch):
        t0 = time.time()
        try:
            df = rq.get_factor(batch, field, start, end)
            return idx, time.time() - t0, (len(df) if df is not None else 0), None
        except Exception as e:
            return idx, time.time() - t0, 0, f"{type(e).__name__}: {str(e)[:120]}"

    t_total = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_one, i + 1, b) for i, b in enumerate(batches)]
        for fut in as_completed(futures):
            idx, el, n, err = fut.result()
            if err:
                log(f"  A batch#{idx:02d} ✗ {el:.1f}s {err}")
            else:
                log(f"  A batch#{idx:02d} ✓ {el:.1f}s rows={n}")
            results.append((idx, err))
    ok = sum(1 for _, e in results if e is None)
    log(
        f"A: total={time.time() - t_total:.1f}s, ok={ok}/{len(batches)}, "
        f"fail={len(batches) - ok}"
    )
    return ok, len(batches) - ok


def run_b_single_shot(rq, stocks, field, start, end):
    """实验 B：一次性拉取整个股票池。"""
    log(f"实验 B: {field} | 一次性 ({len(stocks)} 股)")
    t0 = time.time()
    try:
        df = rq.get_factor(stocks, field, start, end)
        el = time.time() - t0
        if df is None:
            log(f"B: FAIL None after {el:.1f}s")
            return False
        log(
            f"B: OK shape={df.shape}, elapsed={el:.1f}s, "
            f"non_null={df.notna().sum().sum()}"
        )
        return True
    except Exception as e:
        log(f"B: EXCEPTION after {time.time() - t0:.1f}s: {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="net_cash_flow_growth_ratio_ttm")
    ap.add_argument("--start", default="20150501")
    ap.add_argument("--end", default="20260501")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--skip-a", action="store_true", help="跳过实验 A")
    ap.add_argument("--skip-b", action="store_true", help="跳过实验 B")
    args = ap.parse_args()

    import rqdatac

    log("rqdatac.init() ...")
    rqdatac.init()
    log("init done")

    stocks = rqdatac.all_instruments("CS").order_book_id.tolist()
    log(f"#stocks = {len(stocks)}")
    log(f"field = {args.field}, range = {args.start}..{args.end}")

    if not args.skip_a:
        log("=" * 60)
        run_a_parallel(
            rqdatac, stocks, args.field, args.start, args.end,
            args.batch_size, args.workers,
        )

    if not args.skip_b:
        log("=" * 60)
        run_b_single_shot(rqdatac, stocks, args.field, args.start, args.end)


if __name__ == "__main__":
    main()
