import sys
from loguru import logger
logger.remove(); logger.add(sys.stderr, level="INFO")
from data_fetching.consensus import fetch_consensus_reports_year, _init_rq, all_cs_ids, CONSENSUS_DIR
rq = _init_rq(); ids = all_cs_ids(rq)
CONSENSUS_DIR.mkdir(parents=True, exist_ok=True)
date_rule = "rice_create_tm"
for fy in range(2014, 2027):
    cache = CONSENSUS_DIR / f"reports_fy{fy}_{date_rule}.parquet"
    if cache.exists():
        logger.info(f"fy{fy} 已存在，跳过"); continue
    sd, ed = f"{fy-2}0101", f"{fy+1}0630"
    logger.info(f"拉取 reports fy{fy} {sd}~{ed}")
    df = fetch_consensus_reports_year(ids, fy, sd, ed, date_rule=date_rule, batch_size=400, workers=4, rq=rq)
    df.to_parquet(cache)
    logger.info(f"fy{fy} DONE shape={df.shape} stocks={df.order_book_id.nunique() if len(df) else 0}")
logger.info("ALL REPORTS DONE")
