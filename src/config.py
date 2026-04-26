from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # Embeddings — HuggingFace by default (no API key needed)
    embedding_provider: Literal["huggingface", "openai"] = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Storage
    vector_store_path: str = "./vector_db"

    # Parent-child chunking
    parent_chunk_size: int = 1500
    parent_chunk_overlap: int = 150
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50

    # Retrieval
    top_k_semantic: int = 10
    top_k_bm25: int = 10
    top_k_rerank: int = 5
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @property
    def llm_provider(self) -> Literal["anthropic", "openai"]:
        if self.openai_api_key:
            return "openai"
        if self.anthropic_api_key:
            return "anthropic"
        raise ValueError(
            "No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file."
        )


settings = Settings()
