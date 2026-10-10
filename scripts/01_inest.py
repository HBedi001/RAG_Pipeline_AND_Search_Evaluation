import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. Load environment variables from .env file
load_dotenv()

# 2. Define project paths & settings
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "tax_deductions_collection"

def run_ingestion():
    print("🚀 Starting Tax Law Document Ingestion Pipeline...")

    # 3. Load PDF documents from data/pdfs
    if not PDF_DIR.exists() or not list(PDF_DIR.glob("*.pdf")):
        print(f"❌ Error: No PDF files found in {PDF_DIR}. Please place your tax PDFs there!")
        return

    print(f"📂 Loading PDF files from: {PDF_DIR}")
    loader = PyPDFDirectoryLoader(str(PDF_DIR))
    raw_documents = loader.load()
    print(f"📄 Loaded {len(raw_documents)} raw pages from the tax documents.")

    # 4. Split documents into sliding-window chunks
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # <--- Add this line here
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"🧩 Created {len(chunks)} total text chunks.")

    # 5. Initialize OpenAI Embeddings & ChromaDB Vector Store
    print("🧠 Initializing OpenAI Embeddings and ChromaDB vector store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create and persist vector database locally
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME
    )
    
    print(f"✅ Successfully embedded and stored {len(chunks)} chunks into ChromaDB at: {CHROMA_DIR}")

if __name__ == "__main__":
    run_ingestion()