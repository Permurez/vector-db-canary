import logging
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configure basic logging to track ingestion progress
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
COLLECTION_NAME = "squad_collection"
HOST = "localhost"
PORT = 6333


def main() -> None:
    """Initialize connection and load dataset slice."""
    logger.info("Establishing connection with Qdrant container...")
    client = QdrantClient(host=HOST, port=PORT)

    logger.info("Downloading a slice of the SQuAD dataset...")
    dataset = load_dataset("squad", split="train[:500]")
    contexts = dataset["context"]
    
    logger.info(f"Successfully loaded {len(contexts)} contexts from SQuAD.")


if __name__ == "__main__":
    main()