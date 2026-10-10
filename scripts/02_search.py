import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. Load environment variables
load_dotenv()

# 2. Define project paths & settings
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "tax_deductions_collection"

def test_search():
    print("🔍 Initializing ChromaDB connection for search...")
    
    # 3. Initialize embeddings and load the existing vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # 4. Define a sample tax query
    query = "What are the rules for standard deductions and itemized deductions?"
    print(f"\n❓ Querying vector store with: '{query}'\n")

    # 5. Perform similarity search with score
    results = vector_store.similarity_search_with_score(query, k=4)

    # 6. Display the retrieved chunks
    print("=" * 60)
    for i, (doc, score) in enumerate(results, 1):
        print(f"Result {i} (Distance Score: {score:.4f}):")
        print(f"Source Page: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content snippet:\n{doc.page_content[:300]}...")
        print("-" * 60)

if __name__ == "__main__":
    test_search()