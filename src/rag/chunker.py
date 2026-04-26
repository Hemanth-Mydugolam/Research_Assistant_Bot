import uuid
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings


class ParentChildChunker:
    """
    Two-level chunking strategy:
    - Large *parent* chunks (default 1500 chars) give the LLM rich context.
    - Small *child* chunks (default 400 chars) are indexed for precise retrieval.

    At query time we retrieve children (high precision) then return their
    parents (high context), avoiding the lost-in-the-middle problem from
    stuffing large chunks directly into the embedding space.
    """

    def __init__(self):
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.parent_chunk_size,
            chunk_overlap=settings.parent_chunk_overlap,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.child_chunk_size,
            chunk_overlap=settings.child_chunk_overlap,
        )

    def chunk(
        self, docs: List[Document]
    ) -> Tuple[List[Document], List[Document]]:
        """Return (parent_chunks, child_chunks). Each child carries a parent_id."""
        parents: List[Document] = []
        children: List[Document] = []

        for doc in docs:
            parent_splits = self._parent_splitter.create_documents(
                [doc.page_content], metadatas=[dict(doc.metadata)]
            )
            for parent in parent_splits:
                pid = str(uuid.uuid4())
                parent.metadata["parent_id"] = pid
                parents.append(parent)

                child_splits = self._child_splitter.create_documents(
                    [parent.page_content], metadatas=[dict(parent.metadata)]
                )
                for idx, child in enumerate(child_splits):
                    child.metadata["parent_id"] = pid
                    child.metadata["child_index"] = idx
                    children.append(child)

        return parents, children
