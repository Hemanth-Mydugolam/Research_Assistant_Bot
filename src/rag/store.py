import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

from ..config import settings

logger = logging.getLogger(__name__)


def _make_embeddings():
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        )
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


class VectorStore:
    """
    ChromaDB for child-chunk embeddings + JSON sidecar for parent content.

    Retrieving small children gives high embedding precision; swapping to
    their parent gives the LLM wider, coherent context windows.
    """

    def __init__(self, collection: str = "research"):
        base = Path(settings.vector_store_path) / collection
        base.mkdir(parents=True, exist_ok=True)

        self._embeddings = _make_embeddings()
        self._chroma = Chroma(
            collection_name=f"{collection}_children",
            embedding_function=self._embeddings,
            persist_directory=str(base),
        )
        self._parent_path = base / "parents.json"
        self._parents: Dict[str, Dict[str, Any]] = self._load_parents()

    # ── persistence helpers ───────────────────────────────────────────────────

    def _load_parents(self) -> Dict[str, Dict[str, Any]]:
        if self._parent_path.exists():
            try:
                with open(self._parent_path) as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Could not load parents.json: %s", exc)
        return {}

    def _save_parents(self):
        with open(self._parent_path, "w") as f:
            json.dump(self._parents, f)

    # ── write ─────────────────────────────────────────────────────────────────

    def add_documents(
        self,
        parent_chunks: List[Document],
        child_chunks: List[Document],
    ):
        for p in parent_chunks:
            pid = p.metadata["parent_id"]
            # store only JSON-serialisable metadata
            safe_meta = {
                k: v
                for k, v in p.metadata.items()
                if isinstance(v, (str, int, float, bool))
            }
            self._parents[pid] = {"content": p.page_content, "metadata": safe_meta}
        self._save_parents()

        filtered = filter_complex_metadata(child_chunks)
        if filtered:
            self._chroma.add_documents(filtered)

    # ── read ──────────────────────────────────────────────────────────────────

    def semantic_search(self, query: str, k: int = 10) -> List[Document]:
        children = self._chroma.similarity_search(query, k=k)
        return self._children_to_parents(children)

    def _children_to_parents(self, children: List[Document]) -> List[Document]:
        seen: set = set()
        result: List[Document] = []
        for child in children:
            pid = child.metadata.get("parent_id")
            if pid and pid not in seen and pid in self._parents:
                seen.add(pid)
                p = self._parents[pid]
                result.append(
                    Document(page_content=p["content"], metadata=p["metadata"])
                )
        return result

    def get_all_children_text(self) -> List[str]:
        """Return raw text of every child chunk (for BM25 corpus rebuild)."""
        try:
            data = self._chroma.get()
            return data.get("documents", [])
        except Exception:
            return []

    @property
    def is_empty(self) -> bool:
        return len(self._parents) == 0
