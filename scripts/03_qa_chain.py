import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load environment variables
load_dotenv()

# 2. Define project paths & settings
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "tax_deductions_collection"

def format_docs(docs):
    """Format retrieved documents into a single string with source indicators."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown Document')
        formatted.append(f"Source: {source}\nContent: {doc.page_content}")
    return "\n\n---\n\n".join(formatted)

def run_tax_rag():
    print("🚀 Initializing Tax RAG Generation Chain...")

    # 3. Connect to existing ChromaDB vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    # Create retriever (fetching top 4 chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # 4. Initialize the LLM (GPT-4o-mini or GPT-4o)
    llm = ChatOpenAI(model="gpt-5.1", temperature=0)

    # 5. Define a strict tax assistant prompt template
    template = """You are an expert, professional tax assistant trained on official IRS instructions and tax law. 
    Answer the user's question accurately using ONLY the provided context below. 
    If the answer cannot be found in the context, state clearly that the provided tax documents do not contain enough information.
    Always reference the specific source document/page when providing rules or thresholds.

    Context:
    {context}

    Question: {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 6. Build the LCEL (LangChain Expression Language) Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. Ask a sample tax question
    query = "When should a taxpayer choose to itemize deductions instead of taking the standard deduction?"
    print(f"\n❓ User Question: {query}\n")
    print("🤖 Generating synthesized response from tax records...\n")

    response = rag_chain.invoke(query)
    
    print("=" * 60)
    print(response)
    print("=" * 60)

if __name__ == "__main__":
    run_tax_rag()