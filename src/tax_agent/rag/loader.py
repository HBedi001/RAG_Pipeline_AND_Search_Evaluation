from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_tax_documents(pdf_dir: str | Path):
    """Load all PDF files from the specified directory."""
    print(f"📂 Loading PDF documents from: {pdf_dir}")
    loader = PyPDFDirectoryLoader(str(pdf_dir))
    documents = loader.load()
    print(f"📄 Loaded {len(documents)} raw pages from the tax documents.")
    return documents