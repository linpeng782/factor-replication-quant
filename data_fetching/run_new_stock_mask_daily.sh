#!/bin/bash
# ===================================================================
# new_stock_mask 每日增量更新（dquant）
#
# 功能：每交易日把 cache_dir_dquant/new_stock_mask_long.parquet 追平到
#       combo_mask 的最新交易日，日志落 data_fetching/logs/。
# 依赖：combo_mask 须先刷新 —— 由 stock-data-fetching/run_daily_update.sh
#       （建议 19:00）产出；故本脚本建议排在其后（如 19:30）。
# 幂等：new_stock 已追平 combo_mask 末日时秒退，重复跑无副作用。
#
# crontab 示例（每交易日 19:30；是否交易日由脚本内 latest_trading_date 兜底）：
#   30 19 * * 1-5 /nfs/volume-1593-1/peterzhenglinpeng/factor-replication-quant-new/data_fetching/run_new_stock_mask_daily.sh
# ===================================================================

PROJECT_ROOT="/nfs/volume-1593-1/peterzhenglinpeng/factor-replication-quant-new"
VENV_PATH="/nfs/volume-1593-1/peterzhenglinpeng/peterdidi"
LOG_DIR="${PROJECT_ROOT}/data_fetching/logs"

DATE_STR=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/run_new_stock_mask_daily_${DATE_STR}.log"
mkdir -p "${LOG_DIR}"

echo "========================================" >> "${LOG_FILE}"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE}"
echo "========================================" >> "${LOG_FILE}"

cd "${PROJECT_ROOT}"
source "${VENV_PATH}/bin/activate"

python -m data_fetching.new_stock_mask_dquant >> "${LOG_FILE}" 2>&1
EXIT_CODE=$?

echo "" >> "${LOG_FILE}"
echo "========================================" >> "${LOG_FILE}"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE}"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "执行状态: 成功" >> "${LOG_FILE}"
else
    echo "执行状态: 失败 (退出码: ${EXIT_CODE})" >> "${LOG_FILE}"
fi
echo "========================================" >> "${LOG_FILE}"

# 清理 30 天前旧日志
find "${LOG_DIR}" -name "run_new_stock_mask_daily_*.log" -mtime +30 -delete
find "${LOG_DIR}" -name "new_stock_mask_dquant_*.log" -mtime +30 -delete

exit ${EXIT_CODE}
