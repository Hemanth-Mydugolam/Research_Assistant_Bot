import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_url(url: str) -> List[Document]:
    """Scrape a web page and return its text as a Document."""
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise ValueError(f"Nothing fetched from {url}")
        text = trafilatura.extract(
            downloaded, include_comments=False, include_tables=True
        )
        if not text:
            raise ValueError(f"No text extracted from {url}")
        return [
            Document(
                page_content=text,
                metadata={"source": url, "file_type": "url", "file_name": url},
            )
        ]
    except Exception as exc:
        logger.error("URL load failed for %s: %s", url, exc)
        return []


def load_arxiv(query_or_id: str, max_results: int = 3) -> List[Document]:
    """
    Load ArXiv paper(s) by ID (e.g. '2301.00001') or by keyword query.
    Returns abstract + metadata as Documents.
    """
    try:
        import arxiv

        raw = query_or_id.strip().rstrip("/").split("/")[-1]
        # Looks like an arxiv ID if it's mostly digits/dots/v
        is_id = all(c in "0123456789.v" for c in raw) and len(raw) >= 8

        client = arxiv.Client()
        if is_id:
            search = arxiv.Search(id_list=[raw])
        else:
            search = arxiv.Search(query=query_or_id, max_results=max_results)

        docs: List[Document] = []
        for paper in client.results(search):
            content = (
                f"Title: {paper.title}\n\n"
                f"Authors: {', '.join(a.name for a in paper.authors)}\n\n"
                f"Published: {paper.published.date()}\n\n"
                f"Abstract:\n{paper.summary}"
            )
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(paper.entry_id),
                        "file_type": "arxiv",
                        "file_name": paper.title,
                        "arxiv_id": paper.get_short_id(),
                        "title": paper.title,
                        "published": str(paper.published.date()),
                    },
                )
            )
        logger.info("Loaded %d ArXiv paper(s) for '%s'", len(docs), query_or_id)
        return docs
    except Exception as exc:
        logger.error("ArXiv load failed: %s", exc)
        return []
