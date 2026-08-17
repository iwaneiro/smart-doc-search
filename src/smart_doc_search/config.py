"""
Moduł konfiguracji aplikacji Smart Doc Search.

Używa Pydantic Settings do walidacji i wczytywania zmiennych środowiskowych.
Zapewnia jedno, centralne miejsce dla całej konfiguracji aplikacji,
eliminując rozrzucone wywołania os.getenv() po całym kodzie.
"""

from enum import Enum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Obsługiwani dostawcy modeli językowych."""

    OPENAI = "openai"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """
    Centralna konfiguracja aplikacji wczytywana z pliku .env.

    Pydantic automatycznie waliduje typy i zgłasza czytelne błędy,
    gdy wymagana zmienna środowiskowa jest nieobecna lub ma zły typ.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignoruj nieznane zmienne środowiskowe
    )

    # Dostawca LLM
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Aktywny dostawca modelu językowego.",
    )

    # OpenAI
    openai_api_key: str = Field(
        default="",
        description="Klucz API OpenAI. Wymagany gdy llm_provider=openai.",
    )
    openai_llm_model: str = Field(
        default="gpt-4o-mini",
        description="Nazwa modelu czatu OpenAI.",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Nazwa modelu embeddingów OpenAI.",
    )

    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Adres lokalnego serwera Ollama.",
    )
    ollama_llm_model: str = Field(
        default="llama3.2",
        description="Nazwa modelu czatu w Ollama.",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Nazwa modelu embeddingów w Ollama.",
    )

    # ChromaDB
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="Ścieżka do katalogu persystacji ChromaDB.",
    )
    chroma_collection_name: str = Field(
        default="smart_doc_search",
        description="Nazwa kolekcji w ChromaDB.",
    )

    # Parametry RAG
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Rozmiar fragmentu tekstu w tokenach.",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=500,
        description="Nakładanie się sąsiednich fragmentów tekstu.",
    )
    retriever_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Liczba fragmentów zwracanych przez retriever.",
    )

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str, info: object) -> str:
        """Ostrzega gdy brakuje klucza OpenAI przy wybranym dostawcy."""
        # Pełna walidacja przy runtime w llm_factory.py
        return v

    @property
    def is_openai(self) -> bool:
        """Zwraca True jeśli aktywnym dostawcą jest OpenAI."""
        return self.llm_provider == LLMProvider.OPENAI

    @property
    def is_ollama(self) -> bool:
        """Zwraca True jeśli aktywnym dostawcą jest Ollama."""
        return self.llm_provider == LLMProvider.OLLAMA


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Zwraca singleton instancji Settings.

    Używa lru_cache, żeby obiekt Settings był tworzony tylko raz
    i współdzielony przez całą aplikację — bez ponownego odczytu .env.

    Returns:
        Settings: Zwalidowana konfiguracja aplikacji.
    """
    return Settings()