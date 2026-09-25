import logging
import time

from prometheus_client import Gauge, Histogram, start_http_server
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
COLLECTION_NAME = "squad_collection"
MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
PROMETHEUS_PORT = 8000
CHECK_INTERVAL_SECONDS = 15

# The "Golden Record" canary query
CANARY_QUERY = "What is the history of the universe?"

# Prometheus Metrics Definition
SEARCH_LATENCY = Histogram("qdrant_search_latency_seconds", "Latency of Qdrant search in seconds")
TOP_SCORE_GAUGE = Gauge("qdrant_canary_top_score", "Cosine similarity score of the top canary result")
CANARY_HEALTH = Gauge("qdrant_canary_health_status", "1 if canary is healthy, 0 if error")


def main() -> None:
    """Run the continuous canary health-check loop."""
    logger.info(f"Starting Prometheus metrics server on port {PROMETHEUS_PORT}...")
    start_http_server(PROMETHEUS_PORT)

    logger.info("Connecting to Qdrant container...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    logger.info(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    logger.info(f"Starting canary monitoring loop (Interval: {CHECK_INTERVAL_SECONDS}s)...")
    
    while True:
        try:
            # 1. Encode the test query into a vector
            query_vector = model.encode(CANARY_QUERY).tolist()

            # 2. Perform search and measure exact latency
            start_time = time.time()
            search_results = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=1,
            )
            latency = time.time() - start_time

            # 3. Expose latency metric
            SEARCH_LATENCY.observe(latency)
            
            # 4. Verify results and expose score/health metrics
            if search_results:
                top_score = search_results[0].score
                TOP_SCORE_GAUGE.set(top_score)
                CANARY_HEALTH.set(1)
                logger.info(f"Canary OK | Latency: {latency:.4f}s | Top Score: {top_score:.4f}")
            else:
                TOP_SCORE_GAUGE.set(0.0)
                CANARY_HEALTH.set(0)
                logger.warning("Canary warning: Query returned empty results.")

        except Exception as e:
            logger.error(f"Canary check failed: {e}")
            CANARY_HEALTH.set(0)

        # Pause before the next health check
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()