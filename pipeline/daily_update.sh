#!/usr/bin/env bash
# ⚠️ 此脚本已废弃（2026-07-20）
# ------------------------------------------------------------
# 原因：本 sh 与 data_fetching/DAILY_UPDATE_GUIDE.md 严重分叉——
#   · 缺 A7 行业指数、A8 new_stock_mask、A10 ret20_panel 等步骤
#   · 因子线一律"失败仅告警"，无硬门槛校验，静默过期
#   · step 3b 曾误带 --full，强制重写 2005~今全部分钟文件（已删，见 scripts/_verify_minute_full_rewrite.py）
# 保留此文件占位，防止从 git 历史恢复出旧危险版本。
# ------------------------------------------------------------
# 日更请让 agent 按 data_fetching/DAILY_UPDATE_GUIDE.md 执行（A 数据线 → B 因子线，失败即停）。
echo "❌ pipeline/daily_update.sh 已废弃。" >&2
echo "   日更请让 agent 按 data_fetching/DAILY_UPDATE_GUIDE.md 执行（A 数据线 → B 因子线，失败即停）。" >&2
exit 1
