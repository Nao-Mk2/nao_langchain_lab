from langchain.chains import RetrievalQA
import llm.google
import rag.vector_store


def main():
  model = llm.google.Gemini()

  rag.vector_store.load_knowledge()
  retriever = rag.vector_store.create_retriver()

  qa_chain = RetrievalQA.from_chain_type(
      llm=model,
      retriever=retriever,
      return_source_documents=True
  )

  query = input("Enter your message for the AI: ")

  result = qa_chain.invoke({"query": query})

  print(result['result'])


if __name__ == "__main__":
  main()