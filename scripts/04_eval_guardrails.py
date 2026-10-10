import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "tax_deductions_collection"

def guardrail_check(user_query: str) -> bool:
    """Simple input guardrail to verify the query is tax-related."""
    # For a tax assignment, ensure keywords or general intent align with finance/tax
    tax_keywords = ["tax", "deduction", "standard", "itemize", "income", "return", "irs", "filing"]
    query_lower = user_query.lower()
    return any(keyword in query_lower for keyword in tax_keywords)

def run_guarded_tax_rag(query: str):
    print(f"🛡️ Running Input Guardrail for query: '{query}'")
    
    if not guardrail_check(query):
        print("❌ Guardrail Blocked: Query is outside the scope of tax law and IRS regulations.")
        return

    print("✅ Guardrail Passed! Initializing RAG pipeline...")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    llm = ChatOpenAI(model="gpt-5.1", temperature=0)

    template = """You are a professional tax assistant. Answer using ONLY the context provided.
    If the context lacks the answer, state that the documents do not contain enough info.
    
    Context: {context}
    Question: {question}
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    docs = retriever.invoke(query)
    context_text = "\n\n".join([d.page_content for d in docs])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context_text, "question": query})
    
    print("\n" + "=" * 50)
    print("🤖 Synthesized Response:")
    print(response)
    print("=" * 50)

if __name__ == "__main__":
    # Test 1: Valid tax query
    run_guarded_tax_rag("What are the criteria for claiming dependents or standard deductions?")
    
    print("\n" + "-" * 50 + "\n")
    
    # Test 2: Invalid/Out-of-bounds query caught by guardrail
    run_guarded_tax_rag("What is the best recipe for chocolate chip cookies?")