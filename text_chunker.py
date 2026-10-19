from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents,
    chunk_size=800,
    chunk_overlap=120,
):
    """Split document text into chunks while preserving metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []

    for document in documents:
        text = document.get("text", "").strip()

        if not text:
            continue

        text_parts = splitter.split_text(text)

        for part in text_parts:
            chunks.append(
                {
                    "text": part,
                    "source": document.get("source", "Unknown"),
                    "file_type": document.get("file_type", "unknown"),
                    "page_number": document.get("page_number"),
                    "chunk_id": len(chunks),
                }
            )

    return chunks