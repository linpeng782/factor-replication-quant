#!/usr/bin/env bash
# 日更编排器（daily_update）—— 串起数据线(fetching仓) + 因子线(本仓)，fail-fast。
# 本地/服务器通用：改下面 4 个变量即可。详见 docs/daily_update.md。
set -uo pipefail

# ── 环境（服务器固定默认；如需可用环境变量覆盖）──
export FACTOR_REPL_DATA_ROOT="${FACTOR_REPL_DATA_ROOT:-/nfs/ofs-prediction/peterzhenglinpeng}"
export MINUTE_WORKERS="${MINUTE_WORKERS:-64}"          # 128 核机；生产吃满并行
export MINUTE_CHUNK_DAYS="${MINUTE_CHUNK_DAYS:-250}"   # 800G 内存可设大块
export FETCHER_WORKERS="${FETCHER_WORKERS:-24}"        # get_factor 多线程拉取
VENV="${VENV:-/nfs/volume-1593-1/peterzhenglinpeng/peterdidi/bin/activate}"
REPO="${REPO:-/nfs/volume-1593-1/peterzhenglinpeng/factor-replication-quant-new}"
FETCH="${FETCH:-$REPO/data_fetching}"      # 数据线已并入本仓 data_fetching/
# 因子线范围（默认=生产模型 cxl_a158_p27_raw_shap_v2 用到的源：cxl + paper_27；alpha158 见 7b）。
# 其余 source（founder/guosen/其它 kysec paper）默认不更新。要全跑：FACTOR_GLOB='sources/*/*/specs/*'
FACTOR_GLOB="${FACTOR_GLOB:-sources/cxl/*/specs/* sources/kysec/paper_27_microstructure/specs/*}"
# 只刷这些 superset（paper_27 用 prv_v3）；置空 SUPERSET_KEY= 则刷全部
SUPERSET_KEY="${SUPERSET_KEY:-prv_v3}"

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
# 基本面 PIT 基础层（cxl 因子线读本地，必须在因子线之前 append 到最新）。见 docs/cxl_fundamental_incremental_design.md
step "5b fundamentals（基本面 PIT 快照 append）"; (cd "$REPO" && PYTHONPATH=. python data_fetching/fundamentals.py) || die "fundamentals"
# 重述审计：比对本地冻结快照 vs API 当前，只告警绝不覆盖（监控财报重述）。非阻塞。
(cd "$REPO" && PYTHONPATH=. python data_fetching/fundamentals.py --audit) || echo "  ⚠️ 重述审计跳过（非阻塞）"

# ── 因子线 ──
step "6/8 refresh_supersets（刷新 ${SUPERSET_KEY:-全部} superset 到最新）"
if [ -n "$SUPERSET_KEY" ]; then
  (cd "$REPO" && PYTHONPATH=. python pipeline/refresh_supersets.py --cache-key "$SUPERSET_KEY") || die "refresh_supersets"
else
  (cd "$REPO" && PYTHONPATH=. python pipeline/refresh_supersets.py) || die "refresh_supersets"
fi

step "7/8 L3 因子重算（范围=$FACTOR_GLOB；逐个，失败仅告警不中断）"
# run.py 自动:面板已存在→增量(尾窗只算新日 append); cxl 中 filter→rolling / change_on 的 5 个因子
# 自动全量重算(从冻结源确定性, 见 spec_resolver.incremental_safe)。
cd "$REPO"
# 因子面板结束日 = 最新 raw 交易日（动态；否则用死的 DEFAULT_END 会停在旧日期，新日进不了面板）
END_DATE="${END_DATE:-$(ls "$FACTOR_REPL_DATA_ROOT"/market-data/minute/raw/[0-9]*.parquet 2>/dev/null | tail -1 | xargs -n1 basename | sed 's/\.parquet//; s/-//g')}"
echo "  end-date=$END_DATE（最新 raw 交易日）"
n_ok=0; n_fail=0
for d in $FACTOR_GLOB; do                       # 支持多个 glob（空格分隔），逐个 spec 目录
  [ -f "$d/spec.yaml" ] || continue
  qp=$(echo "$d" | sed -E 's#^sources/([^/]+)/([^/]+)/specs/([^/]+)/?$#\1/\2/\3#')
  # --yolo-only：只产 factors/raw 面板（信号/推理只读 raw）；不跑评估画图（更快，且 ok/fail 只反映面板更新）
  if PYTHONPATH=. python run.py "$qp" --yolo-only --end-date "$END_DATE" >/dev/null 2>&1; then n_ok=$((n_ok+1)); else echo "  ⚠️ FAIL $qp"; n_fail=$((n_fail+1)); fi
done
echo "  因子完成: ok=$n_ok fail=$n_fail"

step "7b alpha158 L3 增量（无 spec，独立脚本；读本地 raw_ohlcv，零 API，失败仅告警）"
# ⚠️ alpha158 没有 spec，step 7 的 glob 扫不到它，必须独立调用，否则 alpha158 因子永不更新、
#    下游信号(predict_live 自动末日=min(各因子末日))会被 alpha158 旧日期卡死。
(cd "$REPO" && PYTHONPATH=. python scripts/alpha158_daily_update.py) || echo "  ⚠️ alpha158 失败（非阻塞）"

step "8/8 labels 回填"
(cd "$REPO" && PYTHONPATH=. python ml/labels.py) || echo "  ⚠️ labels 跳过（增量回填待补，非阻塞）"

echo ""; echo "✅ daily_update 完成"
