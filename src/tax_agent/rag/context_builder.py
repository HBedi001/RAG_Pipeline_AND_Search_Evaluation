def format_docs(docs):
    """Format retrieved documents into a single string with source indicators."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown Document')
        formatted.append(f"Source: {source}\nContent: {doc.page_content}")
    return "\n\n---\n\n".join(formatted)