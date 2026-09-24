import logging

from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Configure basic logging to track ingestion progress
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
COLLECTION_NAME = "squad_collection"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
HOST = "localhost"
PORT = 6333


def main() -> None:
    """Execute the core data ingestion pipeline from SQuAD to Qdrant."""
    logger.info("Establishing connection with Qdrant container...")
    client = QdrantClient(host=HOST, port=PORT)

    logger.info(f"Loading local embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    vector_size = model.get_sentence_embedding_dimension()

    # Recreate collection for a fresh start
    logger.info(f"Initializing collection '{COLLECTION_NAME}' in Qdrant...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
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

        # Generate dense vector embeddings for the current batch
        embeddings = model.encode(batch_contexts, show_progress_bar=False)

        # Prepare points structure containing vector, id, and payload metadata
        points = [
            models.PointStruct(
                id=batch_ids[j],
                vector=embeddings[j].tolist(),
                payload={"context": batch_contexts[j]},
            )
            for j in range(len(batch_contexts))
        ]

        # Upsert the batch into Qdrant collection
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info(f"Uploaded batch {i // BATCH_SIZE + 1} ({len(batch_contexts)} records)")

    logger.info("Data ingestion pipeline completed successfully!")


if __name__ == "__main__":
    main()