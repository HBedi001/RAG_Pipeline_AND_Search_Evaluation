from pathlib import Path
from tax_agent.rag.loader import load_tax_documents
from tax_agent.rag.chunker import chunk_documents
from tax_agent.rag.vector_store import get_vector_store

def run_ingestion_pipeline(pdf_dir: str | Path, persist_dir: str | Path):
    """Orchestrates the full end-to-end PDF loading, chunking, and vector embedding storage process."""
    print("🚀 Starting Tax Document Ingestion Pipeline...")
    
    # 1. Load raw PDF documents from directory
    raw_docs = load_tax_documents(pdf_dir)
    
    # 2. Split documents into optimized chunks
    chunks = chunk_documents(raw_docs)
    
    # 3. Initialize vector store and add embedded chunks
    vector_store = get_vector_store(persist_dir)
    print(f"📥 Adding {len(chunks)} chunks to ChromaDB vector store...")
    vector_store.add_documents(chunks)
    
    print("✅ Ingestion Pipeline Complete! Documents are successfully embedded and stored.")

if __name__ == "__main__":
    # Resolve paths relative to project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    PDF_DIR = PROJECT_ROOT / "Data" / "pdfs"
    CHROMA_DIR = PROJECT_ROOT / "chroma_db"
    
    run_ingestion_pipeline(PDF_DIR, CHROMA_DIR)