"""
阶段2 V2.1(引擎版)：通过真实算子引擎 _refresh_superset_cache 跑全量 → 对账旧 v4 缓存。
测试 chunking + fork COW 池 + append-only 缓存 + 跨块 warmup-overlap 正确性。
写入【独立临时目录】，绝不碰 golden 缓存。
"""
import argparse, shutil
import pandas as pd
from core import config
from core.operators.minute_intraday_aggregate import _refresh_superset_cache, _SUPERSET_COLUMNS
from scripts.validate_minute_l2 import _cmp_col, _col_pass, EXACT_COLS

GOLDEN = config.INTERMEDIATE_CACHE_DIR / "prv_v3__hc09d46528f"
TMP = config.INTERMEDIATE_CACHE_DIR / "_valtest_l2_engine"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", default="000001.XSHE,600000.XSHG,000651.XSHE")
    a = ap.parse_args()
    stocks = a.stocks.split(",")
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    print(f"引擎全量刷新 {len(stocks)} 股 → {TMP}")
    _refresh_superset_cache(TMP, stocks, 20, 1.0)

    n_ok = 0
    worst = {}
    for ob in stocks:
        tp, gp = TMP / f"{ob}.parquet", GOLDEN / f"{ob}.parquet"
        if not tp.exists() or not gp.exists():
            print(f"  {ob}: 缺文件, skip"); continue
        new, old = pd.read_parquet(tp), pd.read_parquet(gp)
        m = new.merge(old, on="date", suffixes=("_new", "_old"))
        print(f"  {ob}: 引擎产出 {len(new)} 行, 与 golden 重叠 {len(m)} 行 "
              f"({m['date'].min().date()}~{m['date'].max().date()})")
        bad = []
        for c in _SUPERSET_COLUMNS:
            nm, mr, ma = _cmp_col(m[f"{c}_new"].to_numpy(float), m[f"{c}_old"].to_numpy(float))
            if not _col_pass(c, mr, ma, nm):
                bad.append(c)
            p = worst.get(c, (0.0, 0.0, 0))
            worst[c] = (max(p[0], mr), max(p[1], ma), p[2] + nm)
        print(f"    {'✅' if not bad else '❌ '+str(bad)}")
        if not bad:
            n_ok += 1
    print("\n=== 逐列汇总 ===")
    for c in _SUPERSET_COLUMNS:
        mr, ma, nm = worst[c]
        print(f"  {'✅' if _col_pass(c,mr,ma,nm) else '❌'} {c:32s}max_rel={mr:.2e} max_abs={ma:.2e} 不等={nm}")
    print(f"\n结果: {n_ok}/{len(stocks)} 股 bit 级通过 (引擎全链路)")
    shutil.rmtree(TMP)


if __name__ == "__main__":
    main()
