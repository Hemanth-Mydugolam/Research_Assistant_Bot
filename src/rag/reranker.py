import logging
from typing import List

from langchain_core.documents import Document

from ..config import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranking using sentence-transformers.

    A cross-encoder jointly encodes (query, document) pairs and produces a
    relevance score — far more accurate than bi-encoder cosine similarity or
    LLM-based list reranking, at a fraction of the cost.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~85 MB, loads once).
    Falls back to original ordering if sentence-transformers is unavailable.
    """

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(settings.rerank_model)
            logger.info("Cross-encoder loaded: %s", settings.rerank_model)
        except Exception as exc:
            logger.warning("Cross-encoder unavailable (%s) — using original order", exc)

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        if not docs:
            return []
        self._load()
        if self._model is None:
            return docs[:top_k]

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
