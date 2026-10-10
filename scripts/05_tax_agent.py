import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# 1. Load environment variables
load_dotenv()

# 2. Define project paths & settings
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "tax_deductions_collection"

# 3. Initialize Vector Store and Retriever Tool
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)
tax_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

@tool
def search_tax_database(query: str) -> str:
    """Search official IRS publications (Publication 501, Publication 17) for rules on standard deductions, itemized deductions, filing statuses, and dependents."""
    docs = tax_retriever.invoke(query)
    if not docs:
        return "No relevant tax regulations found in the database."
    
    formatted_results = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown IRS Doc')
        formatted_results.append(f"Source: {source}\nExcerpt: {doc.page_content}")
    return "\n\n---\n\n".join(formatted_results)

def run_agent_workflow():
    print("🤖 Initializing Agentic Tax Master Class Workflow...")

    # 4. Initialize LLM with tool binding (using gpt-5.1)
    llm = ChatOpenAI(model="gpt-5.1", temperature=0)
    tools = [search_tax_database]
    llm_with_tools = llm.bind_tools(tools)

    # 5. Define a complex query requiring agentic reasoning
    query = "What are the rules regarding standard deductions for individuals who can be claimed as a dependent on someone else's tax return?"
    print(f"\n❓ Agent Mission: {query}\n")

    messages = [HumanMessage(content=query)]
    
    # First LLM call: Model decides whether to call the tool
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # 6. Execute tool calls if requested by the agent
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            print(f"🛠️ Agent decided to invoke tool: '{tool_call['name']}' with query: {tool_call['args']}")
            
            # Execute the matching tool locally
            if tool_call["name"] == "search_tax_database":
                tool_output = search_tax_database.invoke(tool_call["args"])
                messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

        # Second LLM call: Synthesize final answer using tool outputs
        final_response = llm_with_tools.invoke(messages)
        
        print("\n" + "=" * 60)
        print("🎯 Final Agent Output:")
        print(final_response.content)
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("🎯 Final Agent Output (No tool needed):")
        print(ai_msg.content)
        print("=" * 60)

if __name__ == "__main__":
    run_agent_workflow()