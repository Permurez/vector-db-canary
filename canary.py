import logging
import random
import time

from prometheus_client import Gauge, Histogram, start_http_server
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 15

CANARY_QUERIES = [
    "What is the history of the universe?",
    "Who wrote the play Romeo and Juliet?",
    "What is the capital city of France?",
    "How does a combustion engine work?",
    "When did the Second World War end?"
]

SEARCH_LATENCY = Histogram("qdrant_search_latency_seconds", "Latency of Qdrant search in seconds")
TOP_SCORE_GAUGE = Gauge("qdrant_canary_top_score", "Cosine similarity score of the top canary result")
CANARY_HEALTH = Gauge("qdrant_canary_health_status", "1 if canary is healthy, 0 if error")

def main() -> None:
    """Run the continuous canary health-check loop with rotating queries."""
    logger.info(f"Starting Prometheus metrics server on port {config.PROMETHEUS_PORT}...")
    start_http_server(config.PROMETHEUS_PORT)

    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    model = SentenceTransformer(config.MODEL_NAME)

    logger.info("Starting canary monitoring loop...")
    
    while True:
        try:
            test_query = random.choice(CANARY_QUERIES)
            query_vector = model.encode(test_query).tolist()

            start_time = time.time()
            search_results = client.search(
                collection_name=config.COLLECTION_NAME,
                query_vector=query_vector,
                limit=1,
            )
            latency = time.time() - start_time

            SEARCH_LATENCY.observe(latency)
            
            if search_results:
                top_score = search_results[0].score
                TOP_SCORE_GAUGE.set(top_score)
                CANARY_HEALTH.set(1)
                logger.info(f"Canary OK | Latency: {latency:.4f}s | Score: {top_score:.4f} | Query: '{test_query[:30]}...'")
            else:
                TOP_SCORE_GAUGE.set(0.0)
                CANARY_HEALTH.set(0)
                logger.warning("Canary warning: Query returned empty results.")

        except Exception as e: # noqa: BLE001
            logger.error(f"Canary check failed: {e}")
            CANARY_HEALTH.set(0)

        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()