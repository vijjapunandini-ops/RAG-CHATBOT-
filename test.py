from llm import create_llm


llm = create_llm()

response = llm.invoke(
    "Explain Retrieval Augmented Generation in two sentences."
)

print(response.content)