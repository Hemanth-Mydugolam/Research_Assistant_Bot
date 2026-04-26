from typing import Annotated, Any, Dict, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    query: str
    sub_queries: List[str]
    messages: Annotated[List[BaseMessage], add_messages]
    retrieved_docs: List[Dict[str, Any]]   # serialised {content, metadata} dicts
    internet_results: List[Dict[str, Any]] # {title, body, url} dicts
    route: str                             # "rag" | "web" | "both"
    docs_are_relevant: bool
    conversation_history: List[Dict[str, str]]
    collection_name: str
