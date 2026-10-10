from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_tax_generation_chain(llm_model: str = "gpt-5.1"):
    """Construct and return the tax RAG generation chain."""
    llm = ChatOpenAI(model=llm_model, temperature=0)
    
    template = """You are an expert, professional tax assistant trained on official IRS instructions and tax law. 
    Answer the user's question accurately using ONLY the provided context below. 
    If the answer cannot be found in the context, state clearly that the provided tax documents do not contain enough information.
    Always reference the specific source document/page when providing rules or thresholds.

    Context:
    {context}

    Question: {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()