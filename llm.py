from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


def create_llm():
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    return llm


def generate_answer(
    question,
    retrieved_results,
    chat_history=None
):
    """
    Generate an answer using retrieved document chunks
    and optional conversation history.
    """

    if not retrieved_results:
        return (
            "I could not find relevant information "
            "in the uploaded documents."
        )

    # -----------------------------------
    # Build document context
    # -----------------------------------

    context_parts = []

    for result in retrieved_results:
        chunk = result["chunk"]

        source = chunk.get("source", "Unknown source")
        page_number = chunk.get("page_number")
        text = chunk.get("text", "")

        if page_number is not None:
            source_line = f"Source: {source}, Page: {page_number}"
        else:
            source_line = f"Source: {source}"

        context_parts.append(
            f"{source_line}\n"
            f"Content:\n{text}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # -----------------------------------
    # Build conversation history
    # -----------------------------------

    history_text = ""

    if chat_history:
        history_parts = []

        for message in chat_history[-6:]:
            role = message["role"]
            content = message["content"]

            if role == "user":
                history_parts.append(
                    f"User: {content}"
                )

            elif role == "assistant":
                history_parts.append(
                    f"Assistant: {content}"
                )

        history_text = "\n".join(history_parts)

    # -----------------------------------
    # Create prompt
    # -----------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful document question-answering assistant.

Answer the user's question using only the information
provided in the document context.

Use the conversation history to understand follow-up
questions and references such as "it", "this", or "that".

Do not make up information.

If the answer cannot be found in the provided document
context, say that the information was not found in the
uploaded documents.

Keep the answer clear and concise.

Mention the source file when appropriate.
"""
            ),
            (
                "human",
                """
Conversation History:

{history}

Document Context:

{context}

Current User Question:

{question}

Answer:
"""
            )
        ]
    )

    # -----------------------------------
    # Create chain
    # -----------------------------------

    llm = create_llm()

    chain = prompt | llm

    # -----------------------------------
    # Generate response
    # -----------------------------------

    response = chain.invoke(
        {
            "history": history_text,
            "context": context,
            "question": question
        }
    )

    return response.content
