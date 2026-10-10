from pathlib import Path
from dotenv import load_dotenv
from tax_agent.agent.graph import build_tax_agent_graph
from tax_agent.agent.state import TaxAgentState
from langchain_core.messages import HumanMessage

# Load environment variables (OpenAI API key)
load_dotenv()

def test_tax_agent():
    print("🚀 Initializing Tax Agent Test Suite...")
    
    # 1. Compile the graph
    graph = build_tax_agent_graph()
    
    # 2. Define a complex test query requiring database retrieval
    query = "What are the rules regarding standard deductions for dependents according to IRS publications?"
    print(f"\n❓ Test Query: {query}\n")
    
    # 3. Invoke the graph with initial state
    initial_state: TaxAgentState = {
        "messages": [HumanMessage(content=query)],
        "is_safe": True
    }
    
    print("🔄 Running LangGraph workflow execution...")
    result = graph.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("🎯 Final Graph Execution Output:")
    for message in result["messages"]:
        print(f"[{message.type.upper()}] {message.content[:300]}...")
    print("=" * 60)

if __name__ == "__main__":
    test_tax_agent()