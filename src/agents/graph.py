from langgraph.graph import END, StateGraph

from .nodes import query_analyzer, rag_retriever, route_after_rag, web_searcher
from .state import AgentState


def build_graph():
    """
    LangGraph workflow:

        START
          │
          ▼
      query_analyzer          (decides route + generates sub-queries)
          │
          ├─ "rag"  ──────► rag_retriever ──► [relevant?]
          │                                        ├─ yes ──► END
          │                                        └─ no  ──► web_searcher ──► END
          │
          ├─ "both" ──────► rag_retriever ──────────────────► web_searcher ──► END
          │
          └─ "web"  ────────────────────────────────────────► web_searcher ──► END
    """
    wf = StateGraph(AgentState)

    wf.add_node("query_analyzer", query_analyzer)
    wf.add_node("rag_retriever", rag_retriever)
    wf.add_node("web_searcher", web_searcher)

    wf.set_entry_point("query_analyzer")

    wf.add_conditional_edges(
        "query_analyzer",
        lambda s: s["route"],
        {"rag": "rag_retriever", "web": "web_searcher", "both": "rag_retriever"},
    )

    wf.add_conditional_edges(
        "rag_retriever",
        route_after_rag,
        {"web_search": "web_searcher", "synthesize": END},
    )

    wf.add_edge("web_searcher", END)

    return wf.compile()


graph = build_graph()
