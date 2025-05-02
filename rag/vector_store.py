from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

_vectorstore = None

def load_knowledge():
  global _vectorstore
  if _vectorstore is not None:
    print("Info: Already read knowledge")
    return

  current_dir = os.path.dirname(os.path.abspath(__file__))
  knowledge_path = os.path.join(current_dir, "knowledge")
  embedding_path = os.path.join(current_dir, "faiss_index")

  # Try to load existing embeddings first
  if os.path.exists(embedding_path):
    try:
      embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
      _vectorstore = FAISS.load_local(embedding_path, embedding, allow_dangerous_deserialization=True)
      print("Info: Successfully loaded existing embeddings")
      return
    except Exception as e:
      print(f"Warning: Failed to load existing embeddings: {str(e)}")
      print("Info: Will regenerate embeddings")
      
  # If no existing embeddings or loading failed, generate new ones
  loader = DirectoryLoader(
    knowledge_path, 
    glob="**/*.md", 
    loader_cls=UnstructuredMarkdownLoader,
    loader_kwargs={"mode": "single", "strategy": "fast"}
  )
  docs = loader.load()
  if not docs:
    print("Warning: No documents were found in the knowledge directory.")
    return
  
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  splits = text_splitter.split_documents(docs)
  embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
  _vectorstore = FAISS.from_documents(splits, embedding)
  
  # Save the newly created faiss index
  _vectorstore.save_local(embedding_path)

def create_retriver():
  if _vectorstore is None:
    raise ValueError("Knowledge has not been loaded. Call load_knowledge() first.")
  return _vectorstore.as_retriever()
