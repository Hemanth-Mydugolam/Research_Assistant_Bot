from typing import Dict, List

from langchain_core.documents import Document


def reciprocal_rank_fusion(
    rankings: List[List[Document]], k: int = 60
) -> List[Document]:
    """
    Merge multiple ranked lists with Reciprocal Rank Fusion (RRF).

    RRF score for document d = Σ 1/(k + rank(d)) over all lists.
    Documents appearing in more lists and at higher ranks score higher.
    k=60 is the standard smoothing constant from the original paper.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = doc.page_content[:200]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

    return [doc_map[key] for key in sorted(scores, key=scores.__getitem__, reverse=True)]


class HybridRetriever:
    """
    Fuses dense semantic search (ChromaDB) and sparse keyword search (BM25)
    using Reciprocal Rank Fusion, then re-ranks with a cross-encoder.

    For complex queries the system generates sub-queries (RAG-Fusion) and
    runs retrieval for each, further improving recall via RRF.
    """

    def __init__(self, vector_store, bm25_index, reranker):
        self._vs = vector_store
        self._bm25 = bm25_index
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        sub_queries: List[str] | None = None,
        top_k_semantic: int = 10,
        top_k_bm25: int = 10,
        top_k_final: int = 5,
    ) -> List[Document]:
        queries = [query] + (sub_queries or [])

        # Gather per-query rankings from both retrieval channels
        semantic_lists: List[List[Document]] = []
        bm25_lists: List[List[Document]] = []
        for q in queries:
            semantic_lists.append(self._vs.semantic_search(q, k=top_k_semantic))
            bm25_lists.append(self._bm25.search(q, k=top_k_bm25))

        # Fuse within each channel, then fuse across channels
        fused_semantic = reciprocal_rank_fusion(semantic_lists)
        fused_bm25 = reciprocal_rank_fusion(bm25_lists)
        combined = reciprocal_rank_fusion([fused_semantic, fused_bm25])

        # Deduplicate by parent_id
        seen: set = set()
        unique: List[Document] = []
        for doc in combined:
            pid = doc.metadata.get("parent_id", doc.page_content[:60])
            if pid not in seen:
                seen.add(pid)
                unique.append(doc)

        candidates = unique[: top_k_final * 3]
        return self._reranker.rerank(query, candidates, top_k=top_k_final)
