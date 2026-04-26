import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
    CSVLoader,
)

logger = logging.getLogger(__name__)

_LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".md": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
}

SUPPORTED_EXTENSIONS = set(_LOADERS.keys())


class DocumentProcessor:
    def load_file(
        self,
        path: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        p = Path(path)
        ext = p.suffix.lower()
        if ext not in _LOADERS:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

        loader = _LOADERS[ext](str(p))
        docs = loader.load()

        base_meta: Dict[str, Any] = {
            "source_file": str(p),
            "file_name": p.name,
            "file_type": ext,
        }
        if extra_metadata:
            base_meta.update(extra_metadata)

        for doc in docs:
            doc.metadata.update(base_meta)

        logger.info("Loaded %d page(s) from %s", len(docs), p.name)
        return docs

    def load_directory(self, directory: str, recursive: bool = True) -> List[Document]:
        d = Path(directory)
        if not d.exists():
            raise FileNotFoundError(f"Directory not found: {d}")

        glob_fn = d.rglob if recursive else d.glob
        docs: List[Document] = []
        for f in glob_fn("*"):
            if f.is_file() and f.suffix.lower() in _LOADERS:
                try:
                    docs.extend(self.load_file(str(f)))
                except Exception as exc:
                    logger.warning("Skipping %s: %s", f.name, exc)
        return docs
