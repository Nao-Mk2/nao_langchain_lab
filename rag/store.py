from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

class VectorStore:
  def __init__(self, embedding_type="local"):
    """
    Initialize the VectorStore with the specified embedding type
    
    Args:
        embedding_type: String, either 'local' or 'cloud'
    """
    self._vectorstore = None
    self._embedding_type = embedding_type.lower()
    self._collection_name = "documents"

    # Initialize the embedding model and similarity function based on embedding type
    if self._embedding_type == "local":
      self._embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/stsb-xlm-r-multilingual",
        encode_kwargs={"convert_to_tensor": True}
      )
      self._similarity = "cosine"  # Using cosine similarity for sentence-transformers
      self._threshold = 0.7
      self._compare_func = lambda score: score <= self._threshold
    else:
      self._embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
      self._similarity = "l2"  # Using L2 (Euclidean) similarity for gemini
      self._threshold = 0.7
      self._compare_func = lambda score: score <= self._threshold

  def load_knowledge(self):
    if self._vectorstore is not None:
      print("Info: Already read knowledge")
      return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    embedding_path = os.path.join(current_dir, "chroma_store", self._embedding_type)
    # Create directory if it doesn't exist
    os.makedirs(embedding_path, exist_ok=True)

    collection_metadata = {"hnsw:space": self._similarity}
    # Try to load existing embeddings first
    if os.path.exists(embedding_path) and os.path.isdir(embedding_path):
      try:
        self._vectorstore = Chroma(
          persist_directory=embedding_path, 
          embedding_function=self._embedding_model,
          collection_metadata=collection_metadata,
          collection_name=self._collection_name
        )
        doc_count = self._vectorstore._collection.count()
        if doc_count > 0:
          print(f"Info: Successfully loaded existing embeddings with {doc_count} documents")
          return
      except Exception as e:
        print(f"Warning: Failed to load existing embeddings: {str(e)}")
        print("Info: Will regenerate embeddings")

    # If no existing embeddings or loading failed, generate new ones
    knowledge_path = os.path.join(current_dir, "knowledge")
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
    self._vectorstore = Chroma.from_documents(
      documents=splits,
      embedding=self._embedding_model,
      persist_directory=embedding_path,
      collection_metadata=collection_metadata,
      collection_name=self._collection_name
    )

  def create_retriever(self):
    if self._vectorstore is None:
      raise ValueError("Knowledge has not been loaded. Call load_knowledge() first.")
    return self._vectorstore.as_retriever(search_kwargs={"k": 4})

  def filter_documents_by_similarity_score(self, scored_docs):
    """
    Filters documents based on their similarity scores according to the 
    appropriate threshold for the current embedding model.
    
    Args:
        scored_docs: A list of tuples (document, score) from similarity search
        
    Returns:
        list: Filtered list of documents that pass the threshold
    """
    return [doc for doc, score in scored_docs if self._compare_func(score)]
