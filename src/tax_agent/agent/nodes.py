from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from tax_agent.rag.nlu_guard import validate_tax_intent
from tax_agent.rag.vector_store import get_vector_store
from langchain_core.tools import tool
from pathlib import Path
import os

# Initialize retriever tool locally for the agent nodes
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
vector_store = get_vector_store(CHROMA_DIR)
tax_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

@tool
def search_tax_database(query: str) -> str:
    """Search official IRS publications (Publication 501, Publication 17) for tax laws and deductions."""
    docs = tax_retriever.invoke(query)
    if not docs:
        return "No relevant tax regulations found in the database."
    formatted = [f"Source: {d.metadata.get('source', 'IRS Doc')}\nContent: {d.page_content}" for d in docs]
    return "\n\n---\n\n".join(formatted)

def guardrail_node(state: dict):
    """Node 1: Evaluates user input intent using the NLU guard."""
    messages = state["messages"]
    last_message = messages[-1].content
    
    is_safe = validate_tax_intent(last_message)
    if not is_safe:
        return {
            "messages": [AIMessage(content="❌ Guardrail Blocked: Query is outside the scope of IRS tax law and regulations.")],
            "is_safe": False
        }
    return {"is_safe": True}

def agent_node(state: dict):
    """Node 2: Invokes the LLM with tool binding to generate a response or request a tool call."""
    messages = state["messages"]
    llm = ChatOpenAI(model="gpt-5.1", temperature=0)
    tools = [search_tax_database]
    llm_with_tools = llm.bind_tools(tools)
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: dict):
    """Node 3: Executes tool calls requested by the agent."""
    messages = state["messages"]
    last_message = messages[-1]
    
    outputs = []
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "search_tax_database":
                tool_output = search_tax_database.invoke(tool_call["args"])
                outputs.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
                
    return {"messages": outputs}