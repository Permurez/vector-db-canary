import logging

from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 64

def main() -> None:
    """Execute the core data ingestion pipeline from SQuAD to Qdrant."""
    logger.info("Establishing connection with Qdrant container...")
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    logger.info(f"Loading local embedding model: {config.MODEL_NAME}...")
    model = SentenceTransformer(config.MODEL_NAME)
    vector_size = model.get_sentence_embedding_dimension()

    logger.info(f"Initializing collection '{config.COLLECTION_NAME}' in Qdrant...")
    client.recreate_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )

    logger.info("Downloading a slice of the SQuAD dataset...")
    dataset = load_dataset("squad", split="train[:500]")
    contexts = dataset["context"]
    ids = list(range(len(contexts)))

    logger.info("Starting data ingestion and vector upsert...")
    for i in range(0, len(contexts), BATCH_SIZE):
        batch_contexts = contexts[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]

        embeddings = model.encode(batch_contexts, show_progress_bar=False)

        points = [
            models.PointStruct(
                id=batch_ids[j],
                vector=embeddings[j].tolist(),
                payload={"context": batch_contexts[j]},
            )
            for j in range(len(batch_contexts))
        ]

        client.upsert(collection_name=config.COLLECTION_NAME, points=points)
        logger.info(f"Uploaded batch {i // BATCH_SIZE + 1} ({len(batch_contexts)} records)")

    logger.info("Data ingestion pipeline completed successfully!")

if __name__ == "__main__":
    main()