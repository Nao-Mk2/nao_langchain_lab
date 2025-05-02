from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import llm.google
import rag.vector_store

def build_chain():
  model = llm.google.Gemini()

  context_prompt = PromptTemplate.from_template(
    "以下の情報に基づいて質問に答えてください。\n\n{context}\n\n質問: {question}"
  )
  context_chain = context_prompt | model

  fallback_prompt = PromptTemplate.from_template(
    "一般知識に基づいて質問に答えてください。\n\n質問: {question}"
  )
  fallback_chain = fallback_prompt | model

  rag.vector_store.load_knowledge()

  def retrieve(question):
    retriever = rag.vector_store.create_retriver()
    scored_docs = retriever.vectorstore.similarity_search_with_score(question, k=4)
    THRESHOLD = 0.8
    filtered_docs = [doc for doc, score in scored_docs if score <= THRESHOLD]

    if filtered_docs:
        print("debug: context chain invoked")
        context = "\n\n".join([doc.page_content for doc in filtered_docs])
        result = context_chain.invoke({"context": context, "question": question})
        return result.content
    else:
        print("debug: fallback chain invoked")
        result = fallback_chain.invoke({"question": question})
        return result.content
  
  return RunnableLambda(retrieve)

def main():
  rag_chain = build_chain()

  query = input("Enter your message for the AI: ")

  result = rag_chain.invoke(query)
  print(result)


if __name__ == "__main__":
  main()