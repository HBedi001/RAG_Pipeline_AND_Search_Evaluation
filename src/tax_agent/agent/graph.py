from langgraph.graph import StateGraph, END
from tax_agent.agent.state import TaxAgentState
from tax_agent.agent.nodes import guardrail_node, agent_node, tool_node
from langchain_core.messages import AIMessage

def should_continue(state: TaxAgentState):
    """Routing function: checks if guardrail passed and if tool calls are requested."""
    if not state.get("is_safe", True):
        return "end"
    
    messages = state["messages"]
    last_message = messages[-1]
    
    # Type-safe check for AI message tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"

def build_tax_agent_graph():
    """Constructs and compiles the LangGraph state machine for the tax agent."""
    workflow = StateGraph(TaxAgentState)
    
    # 1. Add execution nodes with type ignore for partial state returns
    workflow.add_node("guardrail", guardrail_node)  # type: ignore
    workflow.add_node("agent", agent_node)          # type: ignore
    workflow.add_node("tools", tool_node)          # type: ignore
    
    # 2. Set entry point to run input guardrail first
    workflow.set_entry_point("guardrail")
    
    # 3. Conditional routing out of guardrail
    workflow.add_conditional_edges(
        "guardrail",
        lambda state: "agent" if state.get("is_safe", True) else "end",
        {
            "agent": "agent",
            "end": END
        }
    )
    
    # 4. Conditional routing out of agent (to tools or finish)
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # 5. Route tool execution back to the agent for final synthesis
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()