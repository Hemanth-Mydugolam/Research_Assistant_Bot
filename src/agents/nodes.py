import functools
import json
import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from ..config import settings
from ..rag.pipeline import RAGPipeline
from .state import AgentState

logger = logging.getLogger(__name__)

# ── pipeline cache (collection → RAGPipeline) ─────────────────────────────────
_pipelines: Dict[str, RAGPipeline] = {}


def get_pipeline(collection: str) -> RAGPipeline:
    if collection not in _pipelines:
        _pipelines[collection] = RAGPipeline(collection)
    return _pipelines[collection]


@functools.lru_cache(maxsize=1)
def _get_llm():
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.anthropic_api_key,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )


def _parse_json(text: str) -> Dict[str, Any]:
    """Strip markdown fences and parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    return json.loads(text)


def _doc_to_dict(doc: Document) -> Dict[str, Any]:
    return {"content": doc.page_content, "metadata": doc.metadata}


# ── node: query_analyzer ──────────────────────────────────────────────────────

def query_analyzer(state: AgentState) -> Dict[str, Any]:
    """
    Decides routing and generates sub-queries for RAG-Fusion.

    route = "rag"  → query local vector store only
    route = "web"  → query DuckDuckGo only
    route = "both" → query both and fuse results
    """
    query = state["query"]
    collection = state.get("collection_name", "research")
    has_docs = get_pipeline(collection).has_documents
    history = state.get("conversation_history", [])
    history_str = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])

    prompt = f"""Analyse this research query and return JSON with exactly two keys:
1. "route": "rag" (local documents), "web" (internet), or "both"
2. "sub_queries": list of 2-3 alternative phrasings to improve retrieval (RAG-Fusion)

Routing rules:
- "rag"  → query is clearly about the uploaded documents (topic/terminology overlap)
- "web"  → query needs current/real-time information, or no documents are uploaded
- "both" → query is complex, ambiguous, or could benefit from both sources

has_uploaded_documents: {has_docs}
recent_conversation: {history_str or "none"}
query: {query}

Return ONLY valid JSON, no other text."""

    try:
        resp = _get_llm().invoke([HumanMessage(content=prompt)])
        data = _parse_json(resp.content)
    except Exception as exc:
        logger.warning("query_analyzer failed (%s), using fallback", exc)
        data = {"route": "both" if has_docs else "web", "sub_queries": []}

    route = data.get("route", "both")
    if not has_docs and route in ("rag", "both"):
        route = "web"

    return {"route": route, "sub_queries": data.get("sub_queries", [])}


# ── node: rag_retriever ───────────────────────────────────────────────────────

def rag_retriever(state: AgentState) -> Dict[str, Any]:
    """
    Hybrid retrieval (BM25 + semantic + RRF + cross-encoder) followed by
    CRAG-style relevance grading: irrelevant documents are filtered out.
    If fewer than 2 documents pass grading, docs_are_relevant=False triggers
    a fallback to web search.
    """
    query = state["query"]
    sub_queries = state.get("sub_queries", [])
    collection = state.get("collection_name", "research")

    docs = get_pipeline(collection).retrieve(query=query, sub_queries=sub_queries)

    relevant_docs: List[Document] = []
    if docs:
        grade_prompt = f"""For the query: "{query}"
Grade each document as "Y" (relevant) or "N" (not relevant).
Return JSON: {{"grades": ["Y","N",...]}} — one entry per document in the same order.

Documents:
{chr(10).join(f'[{i}] {d.page_content[:350]}' for i, d in enumerate(docs))}

Return ONLY valid JSON."""
        try:
            resp = _get_llm().invoke([HumanMessage(content=grade_prompt)])
            grades = _parse_json(resp.content).get("grades", ["Y"] * len(docs))
        except Exception as exc:
            logger.warning("Relevance grading failed (%s) — accepting all docs", exc)
            grades = ["Y"] * len(docs)

        relevant_docs = [d for d, g in zip(docs, grades) if str(g).upper() == "Y"]

    return {
        "retrieved_docs": [_doc_to_dict(d) for d in relevant_docs],
        "docs_are_relevant": len(relevant_docs) >= 2,
    }


# ── node: web_searcher ────────────────────────────────────────────────────────

def web_searcher(state: AgentState) -> Dict[str, Any]:
    """DuckDuckGo search with the original query."""
    query = state["query"]
    results: List[Dict[str, Any]] = []

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = ddgs.text(query, max_results=6)
        results = [
            {"title": r.get("title", ""), "body": r.get("body", ""), "url": r.get("href", "")}
            for r in (raw or [])
        ]
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)

    return {"internet_results": results}


# ── routing helper ────────────────────────────────────────────────────────────

def route_after_rag(state: AgentState) -> str:
    """
    After RAG retrieval:
    - "both" route → always also search the web
    - "rag"  route, docs NOT relevant → fall back to web (Corrective RAG)
    - "rag"  route, docs relevant → go straight to synthesis
    """
    if state.get("route") == "both":
        return "web_search"
    if not state.get("docs_are_relevant", False):
        return "web_search"
    return "synthesize"
