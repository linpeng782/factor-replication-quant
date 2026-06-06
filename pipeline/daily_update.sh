#!/usr/bin/env bash
# 日更编排器（daily_update）—— 串起数据线(fetching仓) + 因子线(本仓)，fail-fast。
# 本地/服务器通用：改下面 4 个变量即可。详见 docs/daily_update.md。
set -uo pipefail

# ── 环境（本地默认；服务器改这里或用环境变量覆盖）──
export FACTOR_REPL_DATA_ROOT="${FACTOR_REPL_DATA_ROOT:-/Users/didi/DATA}"
export MINUTE_WORKERS="${MINUTE_WORKERS:-6}"
export MINUTE_CHUNK_DAYS="${MINUTE_CHUNK_DAYS:-30}"     # 服务器内存大可设 250
VENV="${VENV:-/Users/didi/kdj/peterdidi/bin/activate}"
REPO="${REPO:-/Users/didi/kdj/factor-repilcation-quant}"
FETCH="${FETCH:-$REPO/data_fetching}"      # 数据线已并入本仓 data_fetching/（原独立 stock-data-fetching）
# 因子线只跑哪些 spec（glob，相对 REPO）；默认全部。HF 专跑可改为 'sources/kysec/paper_27_microstructure/specs/*'
FACTOR_GLOB="${FACTOR_GLOB:-sources/*/*/specs/*}"

source "$VENV"
step() { echo ""; echo "========== $* =========="; }
die()  { echo "❌ 失败于: $*"; exit 1; }

# ── 数据线（增量，fail-fast）──
step "1/8 ex_factors";   (cd "$FETCH" && python ex_factors.py)   || die "ex_factors"
step "2/8 raw_ohlcv";    (cd "$FETCH" && python raw_ohlcv.py)    || die "raw_ohlcv"
step "3/8 minute_ohlcv"; (cd "$FETCH" && python minute_ohlcv.py) || die "minute_ohlcv"
step "3b minute --full(幂等补缺)"; (cd "$FETCH" && python minute_ohlcv.py --full) || die "minute --full"
step "4/8 industry";     (cd "$FETCH" && python industry.py)     || die "industry"
step "5/8 market_cap";   (cd "$FETCH" && python market_cap.py)   || die "market_cap"

# ── 因子线 ──
step "6/8 refresh_supersets（刷新所有 superset 到最新）"
(cd "$REPO" && PYTHONPATH=. python pipeline/refresh_supersets.py) || die "refresh_supersets"

step "7/8 L3 因子重算（cache-hit superset；逐个，失败仅告警不中断）"
cd "$REPO"
# 因子面板结束日 = 最新 raw 交易日（动态；否则用死的 DEFAULT_END 会停在旧日期，新日进不了面板）
END_DATE="${END_DATE:-$(ls "$FACTOR_REPL_DATA_ROOT"/market-data/minute/raw/[0-9]*.parquet 2>/dev/null | tail -1 | xargs -n1 basename | sed 's/\.parquet//; s/-//g')}"
echo "  end-date=$END_DATE（最新 raw 交易日）"
n_ok=0; n_fail=0
for d in $FACTOR_GLOB/; do
  [ -f "$d/spec.yaml" ] || continue
  qp=$(echo "$d" | sed -E 's#^sources/([^/]+)/([^/]+)/specs/([^/]+)/?$#\1/\2/\3#')
  if PYTHONPATH=. python run.py "$qp" --end-date "$END_DATE" >/dev/null 2>&1; then n_ok=$((n_ok+1)); else echo "  ⚠️ FAIL $qp"; n_fail=$((n_fail+1)); fi
done
echo "  因子完成: ok=$n_ok fail=$n_fail"

step "8/8 labels 回填"
(cd "$REPO" && PYTHONPATH=. python ml/labels.py) || echo "  ⚠️ labels 跳过（增量回填待补，非阻塞）"

echo ""; echo "✅ daily_update 完成"
