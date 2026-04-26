# Research Assistant Bot

An advanced Retrieval-Augmented Generation (RAG) chatbot for research, combining hybrid search, cross-encoder reranking, corrective RAG, and a streaming Streamlit UI.

---

## Features

- **Parent-child chunking** — small chunks for precise retrieval, large chunks for rich LLM context
- **Hybrid BM25 + semantic search with RRF** — keyword and dense retrieval fused via Reciprocal Rank Fusion
- **Cross-encoder reranking** — sentence-transformers cross-encoder for high-precision final ranking
- **CRAG (Corrective RAG)** — relevance-grades retrieved docs, falls back to web search if quality is poor
- **RAG-Fusion** — generates multiple sub-queries and merges results for better recall
- **Token-level streaming** — responses stream word-by-word via OpenAI / Anthropic SDK
- **Multi-turn conversation memory** — previous exchanges inform each new answer
- **File upload + URL + ArXiv ingestion** — all accessible from the sidebar
- **Pydantic settings** — clean `.env`-based configuration, no hardcoded keys

---

## Architecture

```
app.py  (Streamlit UI)
  │
  ├── LangGraph agent graph
  │     ├── query_analyzer   → decides route + generates sub-queries (RAG-Fusion)
  │     ├── rag_retriever    → hybrid BM25+semantic → RRF → cross-encoder → CRAG grade
  │     └── web_searcher     → DuckDuckGo (fallback or "both" route)
  │
  └── Streaming synthesis    → Anthropic / OpenAI SDK streams directly to UI

src/
├── config.py                 Pydantic settings from .env
├── ingest/
│   ├── processor.py          PDF, DOCX, TXT, MD, HTML, CSV loader
│   └── web.py                URL scraping (trafilatura) + ArXiv ingestion
└── rag/
    ├── chunker.py            Parent-child recursive chunking
    ├── store.py              ChromaDB (children) + JSON sidecar (parents)
    ├── bm25.py               Persistent BM25 keyword index (rank-bm25)
    ├── retriever.py          Hybrid retrieval + Reciprocal Rank Fusion
    ├── reranker.py           Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
    └── pipeline.py           Orchestrates ingest → chunk → index → retrieve
```

### Key technique: Parent-child chunking

Small child chunks (400 chars) are stored in ChromaDB for precise embedding retrieval. When a child is matched, its parent chunk (1500 chars) is returned to the LLM — combining retrieval precision with broader context.

### Key technique: Hybrid search with RRF

BM25 captures exact keyword matches that dense embeddings miss. Both ranked lists are merged with Reciprocal Rank Fusion across all sub-queries (RAG-Fusion), then a cross-encoder reranks the top candidates.

### Key technique: CRAG (Corrective RAG)

After retrieval, each document is graded by the LLM for relevance. If fewer than 2 documents pass, the system automatically falls back to web search instead of hallucinating an answer from weak context.

---

## Quick start

**1. Clone and install**
```bash
git clone https://github.com/Hemanth-Mydugolam/Research_Assistant_Bot.git
cd Research_Assistant_Bot
pip install -r requirements.txt
```

**2. Configure API key**
```bash
# Edit .env and set your key
OPENAI_API_KEY=sk-...
```

**3. Run**
```bash
streamlit run app.py
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required (or set `ANTHROPIC_API_KEY`) |
| `LLM_MODEL` | `gpt-4o-mini` | Any OpenAI or Anthropic model |
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` (free) or `openai` |
| `VECTOR_STORE_PATH` | `./vector_db` | Where ChromaDB and BM25 index are stored |
| `PARENT_CHUNK_SIZE` | `1500` | Parent chunk character size |
| `CHILD_CHUNK_SIZE` | `400` | Child chunk character size (indexed) |
| `TOP_K_RERANK` | `5` | Final number of chunks passed to LLM |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |

---

## Supported document formats

| Format | Extension |
|---|---|
| PDF | `.pdf` |
| Word | `.docx` |
| Markdown | `.md` |
| Plain text | `.txt` |
| HTML | `.html`, `.htm` |
| CSV | `.csv` |
| Web URL | via sidebar URL input |
| ArXiv paper | via sidebar ArXiv ID or keyword |

---

## Dependencies

- [LangChain](https://github.com/langchain-ai/langchain) — document loading, embeddings, LangGraph
- [ChromaDB](https://github.com/chroma-core/chroma) — vector store
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) — embeddings + cross-encoder reranking
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 keyword index
- [Streamlit](https://streamlit.io) — UI
- [trafilatura](https://github.com/adbar/trafilatura) — web scraping
- [arxiv](https://github.com/lukasschwab/arxiv.py) — ArXiv ingestion
- [duckduckgo-search](https://github.com/deedy5/duckduckgo_search) — web search

---

## License

MIT
