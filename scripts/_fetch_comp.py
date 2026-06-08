import sys
from loguru import logger
logger.remove(); logger.add(sys.stderr, level="INFO")
from data_fetching.consensus import fetch_comp_indicators, _init_rq, all_cs_ids, CONSENSUS_DIR
rq = _init_rq(); ids = all_cs_ids(rq)
logger.info(f"start full fetch {len(ids)} stocks")
df = fetch_comp_indicators(ids, "20140101", "20260527", report_range=3, batch_size=300, workers=4, rq=rq)
CONSENSUS_DIR.mkdir(parents=True, exist_ok=True)
df.to_parquet(CONSENSUS_DIR / "comp_indicators_r3.parquet")
logger.info(f"DONE shape={df.shape} stocks={df.order_book_id.nunique()}")
