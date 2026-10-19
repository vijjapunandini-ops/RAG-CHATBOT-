from embeddings import model
from vector_store import load_vector_store, search_vector_store


def retrieve_documents(question, top_k=3):
    """Retrieve the most relevant document chunks for a question."""

    question = question.strip()

    if not question:
        return []

    index, chunks = load_vector_store()

    if index is None or chunks is None:
        return []

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    results = search_vector_store(
        index=index,
        chunks=chunks,
        query_embedding=query_embedding,
        top_k=top_k,
    )

    return results