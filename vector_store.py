from pathlib import Path
import pickle

import faiss
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = VECTOR_STORE_DIR / "faiss.index"
CHUNKS_FILE = VECTOR_STORE_DIR / "chunks.pkl"


def create_vector_store(embeddings, chunks):
    """Create and save a FAISS cosine-similarity index."""

    if len(chunks) == 0 or len(embeddings) == 0:
        raise ValueError(
            "No chunks or embeddings available to create the index."
        )

    if len(embeddings) != len(chunks):
        raise ValueError(
            "The number of embeddings must match the number of chunks."
        )

    embeddings = np.asarray(embeddings, dtype=np.float32).copy()

    # Ensure every embedding is normalized.
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # Inner product of normalized vectors is cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))

    with open(CHUNKS_FILE, "wb") as file:
        pickle.dump(chunks, file)

    return index


def load_vector_store():
    """Load the saved FAISS index and its corresponding chunks."""

    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        return None, None

    index = faiss.read_index(str(INDEX_FILE))

    with open(CHUNKS_FILE, "rb") as file:
        chunks = pickle.load(file)

    # Detect a missing or inconsistent saved chunk collection.
    if index.ntotal != len(chunks):
        raise ValueError(
            "The FAISS index and chunk metadata do not match. "
            "Please process the documents again."
        )

    return index, chunks


def search_vector_store(
    index,
    chunks,
    query_embedding,
    top_k=3,
):
    """Retrieve chunks with the highest cosine-similarity scores."""

    if index is None or not chunks or index.ntotal == 0:
        return []

    query_vector = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(1, -1).copy()

    faiss.normalize_L2(query_vector)

    number_to_search = min(
        max(1, int(top_k)),
        index.ntotal,
    )

    scores, indices = index.search(
        query_vector,
        number_to_search,
    )

    results = []

    for score, index_number in zip(scores[0], indices[0]):
        if index_number < 0:
            continue

        results.append(
            {
                "chunk": chunks[index_number],
                "score": float(score),
            }
        )

    return results