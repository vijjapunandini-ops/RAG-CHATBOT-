import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def create_embeddings(chunks):
    """Convert chunk text into normalized embeddings."""

    if not chunks:
        return np.empty(
            (0, model.get_sentence_embedding_dimension()),
            dtype=np.float32,
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return np.asarray(embeddings, dtype=np.float32)