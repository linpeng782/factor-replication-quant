"""
阶段2/工厂 引擎回归：通过【新 Engine+Reducer】全量重建 superset → bit 对账旧 v4 golden。
测试 抽出的 MinuteAggregateEngine + PeakRidgeValleyReducer + chunking + fork池 + 缓存 + warmup。
写入【独立临时 cache_key】，绝不碰 golden。
"""
import argparse, shutil
import pandas as pd
from core import config
from core.operators.minute_intraday_aggregate import PeakRidgeValleyReducer, _SUPERSET_COLUMNS
from core.operators.minute_engine import MinuteAggregateEngine
from scripts.validate_minute_l2 import _cmp_col, _col_pass, EXACT_COLS

GOLDEN = config.INTERMEDIATE_CACHE_DIR / "prv_v3__hc09d46528f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", default="000001.XSHE,600000.XSHG,000651.XSHE")
    a = ap.parse_args()
    stocks = a.stocks.split(",")

    reducer = PeakRidgeValleyReducer("VALTEST_engine", 20, 1.0)   # 临时 cache_key → 独立目录
    engine = MinuteAggregateEngine(reducer)
    TMP = engine.cache_dir
    if TMP.exists():
        shutil.rmtree(TMP)
    print(f"新引擎全量重建 {len(stocks)} 股 → {TMP}")
    engine.refresh_cache(stocks)

    n_ok, worst = 0, {}
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
        mr, ma, nm = worst.get(c, (0, 0, 0))
        print(f"  {'✅' if _col_pass(c,mr,ma,nm) else '❌'} {c:30s}max_rel={mr:.2e} max_abs={ma:.2e} 不等={nm}")
    print(f"\n结果: {n_ok}/{len(stocks)} 股 bit 级通过 (Engine+Reducer 全链路)")
    shutil.rmtree(TMP)


if __name__ == "__main__":
    main()
