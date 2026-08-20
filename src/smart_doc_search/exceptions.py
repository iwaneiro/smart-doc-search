class SmartDocSearchError(Exception):
    """Base exception for all application errors."""


class ConfigurationError(SmartDocSearchError):
    """Raised when the application configuration is invalid or incomplete.

    Example: LLM_PROVIDER=openai is set but OPENAI_API_KEY is missing.
    """


class LLMProviderError(SmartDocSearchError):
    """Raised when LLM provider initialization or invocation fails."""


class DocumentLoadError(SmartDocSearchError):
    """Raised when a document cannot be loaded or processed."""


class VectorStoreError(SmartDocSearchError):
    """Raised when a ChromaDB vector store operation fails."""


class GenerationError(SmartDocSearchError):
    """Raised when the LLM fails to generate an answer at query time."""