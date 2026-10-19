import os
import hashlib
import shutil

import streamlit as st

from document_loader import load_documents
from text_chunker import chunk_documents
from embeddings import create_embeddings
from vector_store import create_vector_store
from retriever import retrieve_documents
from llm import generate_answer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Domain-Specific RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .status-card {
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 0.5rem;
    }

    .source-box {
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "loaded_documents" not in st.session_state:
    st.session_state["loaded_documents"] = []

if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

if "embeddings" not in st.session_state:
    st.session_state["embeddings"] = None

if "faiss_index" not in st.session_state:
    st.session_state["faiss_index"] = None

if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = []


# ============================================================
# HELPER FUNCTION: BUILD SOURCE LABELS
# ============================================================

def build_source_labels(retrieved_results):
    """
    Create unique source labels from retrieval results.
    Include page numbers when available.
    """

    source_labels = []
    seen_sources = set()

    for result in retrieved_results:
        chunk = result.get("chunk", {})

        source = chunk.get("source", "Unknown source")
        page_number = chunk.get("page_number")

        if page_number is not None:
            label = f"{source} - Page {page_number}"
        else:
            label = str(source)

        if label not in seen_sources:
            source_labels.append(label)
            seen_sources.add(label)

    return source_labels


# ============================================================
# HELPER FUNCTION: DISPLAY SOURCES
# ============================================================

def display_sources(sources):
    """Display citations beneath an assistant answer."""

    if sources:
        st.caption("📚 Sources used")

        for source in sources:
            st.write(f"📄 {source}")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '📚 Domain-Specific RAG Chatbot'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Upload your documents and ask questions using
        Retrieval Augmented Generation with FAISS,
        LangChain and Ollama.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📤 Documents")

    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=[
            "pdf",
            "docx",
            "txt",
            "csv",
            "xlsx",
            "pptx"
        ],
        accept_multiple_files=True
    )

    st.caption(
        "Supported: PDF, DOCX, TXT, CSV, XLSX, PPTX"
    )

    # --------------------------------------------------------
    # PROCESS DOCUMENTS
    # --------------------------------------------------------

    if st.button(
        "⚙️ Process Documents",
        type="primary",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning(
                "Please upload at least one document."
            )

        else:

            # Process only the current upload batch.
            # Older files in uploads/ are left untouched.

            batch_dir = os.path.join(
                "uploads",
                "current_batch"
            )

            os.makedirs(batch_dir, exist_ok=True)

            # Remove files from the previous batch.

            for filename in os.listdir(batch_dir):

                file_path = os.path.join(
                    batch_dir,
                    filename
                )

                if os.path.isfile(file_path):
                    os.remove(file_path)

                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            saved_files = []

            # Save the current uploaded files.

            for uploaded_file in uploaded_files:

                file_bytes = uploaded_file.getvalue()

                file_hash = hashlib.sha256(
                    file_bytes
                ).hexdigest()[:10]

                filename = (
                    f"{file_hash}_{uploaded_file.name}"
                )

                file_path = os.path.join(
                    batch_dir,
                    filename
                )

                with open(file_path, "wb") as file:
                    file.write(file_bytes)

                saved_files.append(filename)

            st.session_state["processed_files"] = (
                saved_files
            )

            # Clear the previous index before processing
            # a new batch, so a failed batch cannot leave
            # the old index available for new questions.

            st.session_state["faiss_index"] = None
            st.session_state["loaded_documents"] = []
            st.session_state["chunks"] = []
            st.session_state["embeddings"] = None

            # ------------------------------------------------
            # EXTRACT DOCUMENTS
            # ------------------------------------------------

            with st.status(
                "Processing documents...",
                expanded=True
            ) as status:

                st.write("📄 Extracting text...")

                documents = load_documents(batch_dir)

                # Guard against documents with no readable text.

                if not documents:

                    status.update(
                        label="No readable documents found.",
                        state="error"
                    )

                    st.error(
                        "No readable text was found. "
                        "Please upload a supported document "
                        "and try again."
                    )

                else:

                    st.session_state[
                        "loaded_documents"
                    ] = documents

                    st.write(
                        f"✅ Extracted {len(documents)} "
                        "document(s)"
                    )

                    # ----------------------------------------
                    # CREATE CHUNKS
                    # ----------------------------------------

                    st.write("✂️ Creating text chunks...")

                    chunks = chunk_documents(
                        documents,
                        chunk_size=500,
                        chunk_overlap=50
                    )

                    # Guard against empty chunks.

                    if not chunks:

                        status.update(
                            label="No text chunks created.",
                            state="error"
                        )

                        st.error(
                            "No text chunks were created. "
                            "Check the uploaded documents."
                        )

                    else:

                        st.session_state["chunks"] = chunks

                        st.write(
                            f"✅ Created {len(chunks)} chunks"
                        )

                        # ------------------------------------
                        # CREATE EMBEDDINGS
                        # ------------------------------------

                        st.write("🧠 Generating embeddings...")

                        embeddings = create_embeddings(chunks)

                        st.session_state[
                            "embeddings"
                        ] = embeddings

                        st.write("✅ Embeddings generated")

                        # ------------------------------------
                        # CREATE FAISS VECTOR DATABASE
                        # ------------------------------------

                        st.write(
                            "🗄️ Creating FAISS vector database..."
                        )

                        index = create_vector_store(
                            embeddings,
                            chunks
                        )

                        st.session_state[
                            "faiss_index"
                        ] = index

                        st.write(
                            f"✅ Stored {index.ntotal} vectors"
                        )

                        status.update(
                            label=(
                                "✅ Documents processed "
                                "successfully!"
                            ),
                            state="complete"
                        )

                        st.success(
                            "Your documents are ready for chat."
                        )


    # ========================================================
    # CHAT CONTROLS
    # ========================================================

    st.divider()

    st.subheader("💬 Chat Controls")

    if st.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):

        st.session_state["chat_history"] = []

        st.rerun()


# ============================================================
# MAIN CHAT AREA
# ============================================================

st.subheader("💬 Document Chat")


# ============================================================
# CHECK WHETHER DOCUMENTS ARE READY
# ============================================================

documents_ready = (
    st.session_state.get("faiss_index") is not None
)

if not documents_ready:

    st.info(
        "👈 Upload your documents from the sidebar "
        "and click **Process Documents** to start."
    )

else:

    # --------------------------------------------------------
    # DISPLAY SAVED CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state["chat_history"]:

        role = message["role"]
        content = message["content"]

        with st.chat_message(role):

            st.markdown(content)

            # Display sources beneath saved assistant answers.

            if role == "assistant":
                display_sources(
                    message.get("sources", [])
                )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about your documents..."
    )

    # ========================================================
    # AUTOMATIC RAG PIPELINE
    # ========================================================

    if question:

        # ----------------------------------------------------
        # DISPLAY AND SAVE USER QUESTION
        # ----------------------------------------------------

        with st.chat_message("user"):
            st.markdown(question)

        st.session_state["chat_history"].append(
            {
                "role": "user",
                "content": question
            }
        )

        # ----------------------------------------------------
        # RETRIEVE RELEVANT DOCUMENT CHUNKS
        # ----------------------------------------------------

        with st.spinner(
            "🔍 Searching your documents..."
        ):

            retrieval_results = retrieve_documents(
                question,
                top_k=3
            )

        # ----------------------------------------------------
        # BUILD SOURCE LABELS
        # ----------------------------------------------------

        source_labels = build_source_labels(
            retrieval_results
        )

        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        if not retrieval_results:

            answer = (
                "I could not find relevant information "
                "in the uploaded documents. "
                "Please try asking a different question."
            )

        else:

            with st.spinner(
                "🤖 Generating answer with Ollama..."
            ):

                answer = generate_answer(
                    question,
                    retrieval_results,
                    st.session_state["chat_history"]
                )

        # ----------------------------------------------------
        # DISPLAY ASSISTANT ANSWER AND SOURCES
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            st.markdown(answer)

            display_sources(source_labels)

        # ----------------------------------------------------
        # SAVE ASSISTANT ANSWER ONCE
        # ----------------------------------------------------

        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": answer,
                "sources": source_labels
            }
        )


# ============================================================
# RAG PIPELINE DETAILS
# ============================================================

st.divider()

with st.expander("🔍 View RAG Pipeline Details"):

    documents = st.session_state.get(
        "loaded_documents",
        []
    )

    chunks = st.session_state.get(
        "chunks",
        []
    )

    embeddings = st.session_state.get(
        "embeddings"
    )

    faiss_index = st.session_state.get(
        "faiss_index"
    )

    # --------------------------------------------------------
    # PROJECT METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Documents", len(documents))

    with col2:
        st.metric("Chunks", len(chunks))

    with col3:

        if embeddings is not None:
            st.metric(
                "Embedding Dimension",
                embeddings.shape[1]
            )
        else:
            st.metric("Embedding Dimension", "—")

    with col4:

        if faiss_index is not None:
            st.metric(
                "FAISS Vectors",
                faiss_index.ntotal
            )
        else:
            st.metric("FAISS Vectors", "—")

    with col5:
        st.metric("LLM", "Ollama")

    # --------------------------------------------------------
    # RAG ARCHITECTURE
    # --------------------------------------------------------

    st.markdown("### 🧩 RAG Architecture")

    st.code(
        """
User Question
      ↓
Sentence Transformer
      ↓
FAISS Vector Search
      ↓
Relevant Document Chunks
      ↓
LangChain Prompt
      ↓
Ollama LLM
      ↓
Grounded Answer + Sources
        """,
        language="text"
    )

    # --------------------------------------------------------
    # LOADED DOCUMENTS
    # --------------------------------------------------------

    if documents:

        st.markdown("### 📄 Loaded Documents")

        document_sources = sorted(
            {
                str(document.get("source", "Unknown source"))
                for document in documents
            }
        )

        for source in document_sources:
            st.write(f"📄 {source}")
