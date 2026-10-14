# RAG-CHATBOT-
# Domain-Specific RAG Chatbot

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers natural-language questions using information retrieved from uploaded documents. It is designed to make document search easier and provide answers grounded in relevant document passages.

## Features

- Extracts text from supported PDF, DOCX, TXT, CSV, XLSX and PPTX documents.
- Splits document text into smaller chunks.
- Generates text embeddings using Sentence Transformers.
- Stores vectors and chunk metadata in FAISS.
- Retrieves relevant passages for user questions.
- Uses LangChain and Ollama to generate answers using retrieved context.
- Provides a Streamlit chat interface with conversation history and source information.

## Technology Stack

- Python
- Streamlit
- pypdf
- LangChain and LangChain Text Splitters
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- Ollama (`llama3.2:3b`)

## Project Workflow

1. Upload supported documents.
2. Extract text and preserve document metadata.
3. Split text into chunks.
4. Generate normalized embeddings.
5. Save embeddings in a FAISS vector index.
6. Retrieve relevant chunks for a question.
7. Pass the question and retrieved context to the language model.
8. Display the answer and available source information.

## Setup

1. Install Python and Ollama.
2. Install the Python dependencies:

   `pip install -r requirements.txt`

3. Download the model:

   `ollama pull llama3.2:3b`

4. Run the Streamlit application from the project directory:

   `streamlit run app.py`

## Testing

Use documents with known answers and test direct questions, follow-up questions, and questions whose answers are absent from the documents. Verify both the generated answers and the retrieved source information.

Run the retrieval evaluation script with:

`python -m evaluation.evaluate_retrieval`

## Limitations

Answer quality depends on document extraction, chunking, retrieval quality and model behavior. The system can return incorrect or incomplete answers, so retrieved sources and generated responses should be checked. Scanned PDFs may require OCR before their text can be extracted.

## Future Improvements

- Expand retrieval evaluation.
- Improve relevance filtering.
- Add more robust document and page citations.
- Explore deployment and document-level access controls.
