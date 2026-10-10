def get_tax_retriever(vector_store, k: int = 4):
    """Convert a vector store into a configurable retriever instance."""
    return vector_store.as_retriever(search_kwargs={"k": k})