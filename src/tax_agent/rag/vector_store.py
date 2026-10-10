from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def get_vector_store(persist_dir: str | Path, collection_name: str = "tax_deductions_collection"):
    """Initialize and return the ChromaDB vector store using OpenAI embeddings."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name=collection_name
    )
    return vector_store