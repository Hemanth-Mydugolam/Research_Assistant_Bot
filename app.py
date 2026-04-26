"""
Research Assistant Bot — Advanced RAG application.

Advancements over the baseline (unikill066/research_rag):
  1. Claude claude-sonnet-4-6 as LLM (vs GPT-3.5-turbo)
  2. Parent-child chunking  — small chunks for retrieval, large for context
  3. Hybrid BM25 + semantic search with Reciprocal Rank Fusion (RAG-Fusion)
  4. Cross-encoder reranking via sentence-transformers (vs LLM-based)
  5. CRAG (Corrective RAG) — relevance-grades docs, falls back to web if poor
  6. Multi-query sub-query generation for better recall
  7. Streaming synthesis via Anthropic SDK + st.write_stream
  8. Conversation memory for multi-turn research sessions
  9. Drag-and-drop file upload + URL + ArXiv ingestion in sidebar
 10. Pydantic settings from .env (no hardcoded keys)
"""

import logging
import tempfile
from pathlib import Path
from typing import Generator, List, Dict, Any

import streamlit as st

from src.agents.graph import graph
from src.agents.nodes import get_pipeline
from src.agents.state import AgentState
from src.config import settings
from src.ingest.processor import DocumentProcessor, SUPPORTED_EXTENSIONS
from src.ingest.web import load_arxiv, load_url

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── cached singletons ─────────────────────────────────────────────────────────


@st.cache_resource
def _doc_processor() -> DocumentProcessor:
    return DocumentProcessor()


@st.cache_resource
def _anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


@st.cache_resource
def _openai_client():
    from openai import OpenAI
    return OpenAI(api_key=settings.openai_api_key)


# ── ingestion helpers ─────────────────────────────────────────────────────────


def _ingest_uploaded(files, collection: str) -> Dict[str, Any]:
    processor = _doc_processor()
    all_docs = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            p = Path(tmp) / f.name
            p.write_bytes(f.read())
            try:
                all_docs.extend(processor.load_file(str(p)))
            except Exception as exc:
                st.warning(f"Could not load **{f.name}**: {exc}")
    if not all_docs:
        return {"error": "No documents could be loaded from the uploaded files."}
    return get_pipeline(collection).ingest(all_docs)


def _ingest_url(url: str, collection: str) -> Dict[str, Any]:
    docs = load_url(url)
    if not docs:
        return {"error": f"Could not fetch content from: {url}"}
    return get_pipeline(collection).ingest(docs)


def _ingest_arxiv(query_or_id: str, collection: str, max_results: int) -> Dict[str, Any]:
    docs = load_arxiv(query_or_id, max_results)
    if not docs:
        return {"error": f"No ArXiv results found for: {query_or_id}"}
    return get_pipeline(collection).ingest(docs)


# ── synthesis prompt builder ──────────────────────────────────────────────────


def _build_prompt(
    query: str,
    history: List[Dict[str, str]],
    retrieved_docs: List[Dict[str, Any]],
    internet_results: List[Dict[str, Any]],
) -> str:
    lines = [
        "You are an expert research assistant. Answer the query using only the provided context.",
        "Use inline citations: [1], [2] for documents and [W1], [W2] for web results.",
        "Be accurate, structured, and concise. If context is insufficient, say so explicitly.",
        "",
    ]

    if history:
        lines.append("## Conversation History")
        for m in history[-6:]:
            lines.append(f"**{m['role'].title()}**: {m['content']}")
        lines.append("")

    if retrieved_docs:
        lines.append("## Retrieved Documents")
        for i, d in enumerate(retrieved_docs, 1):
            meta = d.get("metadata", {})
            name = meta.get("file_name", meta.get("source", "Unknown"))
            page = meta.get("page", "")
            page_str = f" (page {page})" if page else ""
            lines.append(f"[{i}] **{name}{page_str}**")
            lines.append(d["content"][:900])
            lines.append("")

    if internet_results:
        lines.append("## Web Search Results")
        for i, r in enumerate(internet_results, 1):
            lines.append(f"[W{i}] **{r.get('title', 'Result')}**  —  {r.get('url', '')}")
            lines.append(r.get("body", "")[:500])
            lines.append("")

    lines += [f"## Query\n{query}", "", "## Answer (with citations)"]
    return "\n".join(lines)


# ── streaming synthesis ───────────────────────────────────────────────────────


def _stream_answer(
    query: str,
    history: List[Dict[str, str]],
    retrieved_docs: List[Dict[str, Any]],
    internet_results: List[Dict[str, Any]],
) -> Generator[str, None, None]:
    prompt = _build_prompt(query, history, retrieved_docs, internet_results)

    if settings.llm_provider == "anthropic":
        client = _anthropic_client()
        with client.messages.stream(
            model=settings.llm_model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text
    else:
        client = _openai_client()
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Research Assistant")
    st.caption("Advanced RAG + Web Search")
    st.divider()

    collection = st.text_input(
        "Collection",
        value="research",
        help="Documents are grouped into named collections. Change the name to start a fresh knowledge base.",
    )

    # ── file upload ───────────────────────────────────────────────────────────
    st.subheader("Add Documents")
    ext_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    uploaded = st.file_uploader(
        f"Upload files ({ext_list})",
        accept_multiple_files=True,
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
    )
    if uploaded and st.button("Ingest Files", type="primary", use_container_width=True):
        with st.spinner("Processing…"):
            result = _ingest_uploaded(uploaded, collection)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(
                f"Ingested **{result['documents_ingested']}** file(s) → "
                f"**{result['child_chunks']}** chunks"
            )

    st.divider()

    # ── URL ───────────────────────────────────────────────────────────────────
    url_input = st.text_input("Ingest URL", placeholder="https://…")
    if st.button("Load URL", use_container_width=True) and url_input:
        with st.spinner("Fetching…"):
            result = _ingest_url(url_input, collection)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Loaded URL → {result['child_chunks']} chunks")

    st.divider()

    # ── ArXiv ─────────────────────────────────────────────────────────────────
    arxiv_input = st.text_input("Ingest ArXiv", placeholder="2301.00001 or search query")
    arxiv_n = st.slider("Max papers", 1, 5, 2)
    if st.button("Load ArXiv", use_container_width=True) and arxiv_input:
        with st.spinner("Fetching…"):
            result = _ingest_arxiv(arxiv_input, collection, arxiv_n)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Loaded {result['documents_ingested']} paper(s) → {result['child_chunks']} chunks")

    st.divider()

    # ── display options ───────────────────────────────────────────────────────
    show_sources = st.toggle("Show sources", value=True)
    show_trace = st.toggle("Show agent trace", value=False)

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # collection status
    pipeline = get_pipeline(collection)
    if pipeline.has_documents:
        st.caption(f"Collection **{collection}** has documents ✓")
    else:
        st.caption(f"Collection **{collection}** is empty — upload docs or the bot will use web search.")


# ── main chat area ────────────────────────────────────────────────────────────

st.title("Research Assistant")
st.caption("Ask questions about your documents, ArXiv papers, or any research topic.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_sources and msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    page = f" — page {src['page']}" if src.get("page") else ""
                    st.markdown(f"- **{src['name']}**{page}  \n  `{src['url']}`")
        if show_trace and msg.get("trace"):
            with st.expander("Agent trace"):
                st.json(msg["trace"])

# chat input
if user_query := st.chat_input("Ask a research question…"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        # ── run LangGraph (retrieval + routing) ───────────────────────────────
        with st.spinner("Retrieving context…"):
            initial_state: AgentState = {
                "query": user_query,
                "sub_queries": [],
                "messages": [],
                "retrieved_docs": [],
                "internet_results": [],
                "route": "both",
                "docs_are_relevant": False,
                "conversation_history": history,
                "collection_name": collection,
            }
            try:
                final_state = graph.invoke(initial_state)
            except Exception as exc:
                logger.error("Graph error: %s", exc)
                st.error(f"Agent error: {exc}")
                st.stop()

        retrieved_docs = final_state.get("retrieved_docs", [])
        internet_results = final_state.get("internet_results", [])

        # ── stream final answer ───────────────────────────────────────────────
        try:
            response_text: str = st.write_stream(
                _stream_answer(user_query, history, retrieved_docs, internet_results)
            )
        except Exception as exc:
            logger.error("Streaming error: %s", exc)
            st.error(f"LLM error: {exc}")
            st.stop()

        # ── build source list for display ─────────────────────────────────────
        sources = []
        for d in retrieved_docs:
            meta = d.get("metadata", {})
            sources.append({
                "name": meta.get("file_name", "Document"),
                "url": meta.get("source_file", meta.get("source", "")),
                "page": meta.get("page", ""),
            })
        for r in internet_results:
            sources.append({
                "name": r.get("title", "Web result"),
                "url": r.get("url", ""),
                "page": "",
            })

        if show_sources and sources:
            with st.expander(f"Sources ({len(sources)})"):
                for src in sources:
                    page = f" — page {src['page']}" if src.get("page") else ""
                    st.markdown(f"- **{src['name']}**{page}  \n  `{src['url']}`")

        trace = {
            "route": final_state.get("route"),
            "sub_queries": final_state.get("sub_queries"),
            "docs_retrieved": len(retrieved_docs),
            "web_results": len(internet_results),
        }
        if show_trace:
            with st.expander("Agent trace"):
                st.json(trace)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources,
        "trace": trace,
    })
