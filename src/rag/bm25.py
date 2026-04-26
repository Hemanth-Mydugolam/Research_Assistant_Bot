import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
from langchain_core.documents import Document

from ..config import settings

logger = logging.getLogger(__name__)


class BM25Index:
    """
    Persistent BM25 (Okapi) index over child-chunk text.

    BM25 excels at exact-keyword and rare-term queries that dense embeddings
    miss. Combining it with semantic search via Reciprocal Rank Fusion gives
    significantly better recall than either alone.
    """

    def __init__(self, collection: str = "research"):
        self._path = Path(settings.vector_store_path) / collection / "bm25.pkl"
        self._corpus: List[str] = []
        self._docs: List[Document] = []
        self._index = None
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if not self._path.exists():
            return
        try:
            with open(self._path, "rb") as f:
                data = pickle.load(f)
            self._corpus = data["corpus"]
            self._docs = data["docs"]
            self._rebuild()
            logger.info("BM25 index loaded: %d documents", len(self._corpus))
        except Exception as exc:
            logger.warning("Could not load BM25 index: %s", exc)

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump({"corpus": self._corpus, "docs": self._docs}, f)

    def _rebuild(self):
        if not self._corpus:
            self._index = None
            return
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [text.lower().split() for text in self._corpus]
            self._index = BM25Okapi(tokenized)
        except ImportError:
            logger.warning("rank_bm25 not installed — BM25 disabled")
            self._index = None

    # ── write ─────────────────────────────────────────────────────────────────

    def add_documents(self, docs: List[Document]):
        self._corpus.extend(d.page_content for d in docs)
        self._docs.extend(docs)
        self._rebuild()
        self._save()

    # ── read ──────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 10) -> List[Document]:
        if self._index is None or not self._corpus:
            return []
        scores = self._index.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:k]
        return [self._docs[i] for i in top_idx if scores[i] > 0]
