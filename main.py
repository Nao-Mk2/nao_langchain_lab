from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import argparse
import llm.google
import llm.local
import rag.store

def build_chain(model_type="local"):
  # Use the same model_type for both LLM and embeddings
  if model_type.lower() == "local":
    print("Using local LLM via LM Studio")
    model = llm.local.LMStudio()
  else:
    print("Using cloud LLM (Google Gemini)")
    model = llm.google.Gemini()

  context_prompt = PromptTemplate.from_template(
    "以下の情報に基づいて質問に答えてください。\n\n{context}\n\n質問: {question}"
  )
  context_chain = context_prompt | model

  fallback_prompt = PromptTemplate.from_template(
    "一般知識に基づいて質問に答えてください。\n\n質問: {question}"
  )
  fallback_chain = fallback_prompt | model

  vector_store = rag.store.VectorStore(model_type)
  vector_store.load_knowledge()

  def retrieve(question):
    retriever = vector_store.create_retriever()
    scored_docs = retriever.vectorstore.similarity_search_with_score(question)
    filtered_docs = vector_store.filter_documents_by_similarity_score(scored_docs)

    print(f"\n------ debug info ------")
    print(f"query: {question}")
    print(f"retriever_metadata: {retriever.metadata}")
    print(f"search_type: {retriever.search_type}")
    print(f"search_num(k): {retriever.search_kwargs.get('k', 'default')}")
    try:
      doc_count = retriever.vectorstore._collection.count()
      print(f"doc_count_in_vector_store: {doc_count}")
    except Exception as e:
      print(f"error_on_retrieving_docs: {str(e)}")
    print(f"scored_docs_count: {len(scored_docs)}")
    print("docs_score_and_content_preview:\n")
    if scored_docs:
      for i, (doc, score) in enumerate(scored_docs):
        print(f"doc{i+1} score: {score}")
        print(f"doc{i+1} content_preview: {doc.page_content[:50]}...")
    print(f"filtered_docs_count: {len(filtered_docs)}")
    print("------------------------\n")

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
  parser = argparse.ArgumentParser(description="RAG application with choice of model type")
  parser.add_argument("--model", choices=["cloud", "local"], default="local",
                     help="Choose model type: 'cloud' for Google Gemini or 'local' for LM Studio (default: local)")
  args = parser.parse_args()
  
  # Use the same model type for both LLM and embeddings
  rag_chain = build_chain(model_type=args.model)

  query = input("Enter your message for the AI: ")

  result = rag_chain.invoke(query)
  print(result)


if __name__ == "__main__":
  main()