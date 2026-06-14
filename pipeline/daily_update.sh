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
# 因子线分两类（默认=生产模型 cxl_a158_p27_raw_shap_v2 的源；alpha158 见 7c）：
#  · 分钟 superset 因子 → 批量"读一次算多个"(refresh_factors_batch，~14× 快)
MINUTE_FACTOR_GLOB="${MINUTE_FACTOR_GLOB:-sources/kysec/paper_27_microstructure/specs/*}"
SUPERSET_KEY="${SUPERSET_KEY:-prv_v3}"        # paper_27 用 prv_v3；置空 SUPERSET_KEY= 则刷全部 superset
#  · 非 superset 因子（cxl 读本地基本面）→ 逐个 run.py 并行
RUNPY_FACTOR_GLOB="${RUNPY_FACTOR_GLOB:-sources/cxl/*/specs/*}"
# 其余 source（founder/guosen/其它 kysec paper）默认不更新；要更全就把上面两个 glob 改宽 + SUPERSET_KEY=

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

cd "$REPO"
N_FACTOR_JOBS="${N_FACTOR_JOBS:-12}"            # run.py 并行度（800G 内存放得下；NFS 读为瓶颈，12~16 即够）
# 因子面板结束日 = 最新 raw 交易日（动态；否则用死的 DEFAULT_END 会停在旧日期，新日进不了面板）
END_DATE="${END_DATE:-$(ls "$FACTOR_REPL_DATA_ROOT"/market-data/minute/raw/[0-9]*.parquet 2>/dev/null | tail -1 | xargs -n1 basename | sed 's/\.parquet//; s/-//g')}"
echo "  因子面板结束日 end-date=$END_DATE"

step "7a/8 分钟 superset 因子：批量 L3（读 superset 一次算多个，~14×）：$MINUTE_FACTOR_GLOB"
# 产出与逐个 run.py bit 一致（口径复刻）；不安全/非 superset 因子会被自动跳过。失败仅告警。
(PYTHONPATH=. python pipeline/refresh_factors_batch.py --factor-glob "$MINUTE_FACTOR_GLOB" --end-date "$END_DATE") || echo "  ⚠️ 批量 L3 失败（非阻塞）"

step "7b/8 非superset因子：run.py 并行 $N_FACTOR_JOBS（cxl 读本地基本面）：$RUNPY_FACTOR_GLOB"
# run.py 自动:面板已存在→增量(尾窗只算新日 append); cxl 中 filter→rolling/change_on 的 5 个因子
# 自动全量重算(从冻结源确定性, 见 spec_resolver.incremental_safe)。并行安全:各写各面板、--yolo-only 只产 raw。
specs=""
for d in $RUNPY_FACTOR_GLOB; do                  # 支持多个 glob（空格分隔），逐个 spec 目录
  [ -f "$d/spec.yaml" ] || continue
  specs="$specs $(echo "$d" | sed -E 's#^sources/([^/]+)/([^/]+)/specs/([^/]+)/?$#\1/\2/\3#')"
done
res=$(printf '%s\n' $specs | xargs -P "$N_FACTOR_JOBS" -I{} sh -c \
  "PYTHONPATH=. python run.py '{}' --yolo-only --end-date $END_DATE >/dev/null 2>&1 && echo OK || echo 'FAIL {}'")
n_ok=$(printf '%s\n' "$res" | grep -c '^OK'); n_fail=$(printf '%s\n' "$res" | grep -c '^FAIL')
printf '%s\n' "$res" | grep '^FAIL' | sed 's/^/  ⚠️ /'
echo "  因子完成: ok=$n_ok fail=$n_fail"

step "7c/8 alpha158 L3 增量（无 spec，独立脚本；读本地 raw_ohlcv，零 API，失败仅告警）"
# ⚠️ alpha158 没有 spec，glob 扫不到，必须独立调用，否则 alpha158 永不更新、下游信号被其旧日期卡死。
(PYTHONPATH=. python scripts/alpha158_daily_update.py) || echo "  ⚠️ alpha158 失败（非阻塞）"

step "8/8 labels 回填"
(cd "$REPO" && PYTHONPATH=. python ml/labels.py) || echo "  ⚠️ labels 跳过（增量回填待补，非阻塞）"

echo ""; echo "✅ daily_update 完成"
