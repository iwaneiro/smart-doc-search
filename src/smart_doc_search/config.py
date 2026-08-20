"""
Central application configuration powered by Pydantic Settings.

All environment variables are loaded and validated in one place.
No scattered os.getenv() calls throughout the codebase.
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Application settings loaded and validated from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM provider selection
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OLLAMA,
        description="Active LLM provider.",
    )

    # OpenAI
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key. Required when llm_provider=openai.",
    )
    openai_llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI chat model name.",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name.",
    )

    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Local Ollama server URL.",
    )
    ollama_llm_model: str = Field(
        default="llama3.2",
        description="Ollama chat model name.",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model name.",
    )

    # ChromaDB
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="ChromaDB persistence directory path.",
    )
    chroma_collection_name: str = Field(
        default="smart_doc_search",
        description="ChromaDB collection name.",
    )

    # RAG parameters
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Text chunk size in characters.",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=500,
        description="Overlap between adjacent text chunks.",
    )
    retriever_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks returned by the retriever.",
    )

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        """Passthrough — full validation happens at runtime in llm_factory.py."""
        return v

    @property
    def is_openai(self) -> bool:
        """Return True if the active provider is OpenAI."""
        return self.llm_provider == LLMProvider.OPENAI

    @property
    def is_ollama(self) -> bool:
        """Return True if the active provider is Ollama."""
        return self.llm_provider == LLMProvider.OLLAMA

    @property
    def collection_name_for_provider(self) -> str:
        """Chroma collection name namespaced by the active LLM provider."""
        return f"{self.chroma_collection_name}_{self.llm_provider.value}"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()