import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from ..config import settings
from .bm25 import BM25Index
from .chunker import ParentChildChunker
from .reranker import CrossEncoderReranker
from .retriever import HybridRetriever
from .store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates the full advanced RAG pipeline:

    Ingest:  Documents → parent-child chunking → ChromaDB (children) + JSON (parents) + BM25
    Retrieve: Query + sub-queries → BM25 + semantic → RRF fusion → cross-encoder rerank → parent chunks
    """

    def __init__(self, collection: str = "research"):
        self.collection = collection
        self._chunker = ParentChildChunker()
        self._store = VectorStore(collection)
        self._bm25 = BM25Index(collection)
        self._reranker = CrossEncoderReranker()
        self._retriever = HybridRetriever(self._store, self._bm25, self._reranker)

    # ── ingest ────────────────────────────────────────────────────────────────

    def ingest(self, docs: List[Document]) -> Dict[str, Any]:
        if not docs:
            return {"error": "No documents provided"}

        parents, children = self._chunker.chunk(docs)
        self._store.add_documents(parents, children)
        self._bm25.add_documents(children)

        logger.info(
            "Ingested %d doc(s) → %d parents, %d children [%s]",
            len(docs), len(parents), len(children), self.collection,
        )
        return {
            "documents_ingested": len(docs),
            "parent_chunks": len(parents),
            "child_chunks": len(children),
            "collection": self.collection,
        }

    # ── retrieve ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        sub_queries: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[Document]:
        if self._store.is_empty:
            return []
        return self._retriever.retrieve(
            query=query,
            sub_queries=sub_queries,
            top_k_semantic=settings.top_k_semantic,
            top_k_bm25=settings.top_k_bm25,
            top_k_final=top_k or settings.top_k_rerank,
        )

    @property
    def has_documents(self) -> bool:
        return not self._store.is_empty
