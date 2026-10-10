from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class TaxAgentState(TypedDict):
    """Defines the state structure for the tax agent execution graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_safe: bool