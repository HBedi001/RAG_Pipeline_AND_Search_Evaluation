from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """Split raw documents into recursive character chunks optimized for legal text."""
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"🧩 Created {len(chunks)} total text chunks.")
    return chunks