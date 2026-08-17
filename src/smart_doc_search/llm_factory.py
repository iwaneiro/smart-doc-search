"""
Factory Pattern implementation for LLM providers.
Decouples the application from specific LLM vendors.
"""

from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from loguru import logger

from smart_doc_search.config import LLMProvider, Settings
from smart_doc_search.exceptions import ConfigurationError, LLMProviderError


class LLMProviderBase(ABC):
    """Abstract interface for LLM providers."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @abstractmethod
    def get_chat_model(self) -> BaseChatModel:
        """Return a configured chat model."""
        pass

    @abstractmethod
    def get_embedding_model(self) -> Embeddings:
        """Return a configured embedding model."""
        pass


class OpenAILLMProvider(LLMProviderBase):
    """LLM provider backed by the OpenAI API."""

    def get_chat_model(self) -> BaseChatModel:
        if not self._settings.openai_api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Provide a key or switch to Ollama."
            )
        try:
            from langchain_openai import ChatOpenAI

            logger.info(f"Initializing ChatOpenAI: {self._settings.openai_llm_model}")
            return ChatOpenAI(
                model=self._settings.openai_llm_model,
                api_key=self._settings.openai_api_key,
                temperature=0.1,
            )
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize ChatOpenAI: {e}") from e

    def get_embedding_model(self) -> Embeddings:
        if not self._settings.openai_api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Provide a key or switch to Ollama."
            )
        try:
            from langchain_openai import OpenAIEmbeddings

            logger.info(
                f"Initializing OpenAIEmbeddings: {self._settings.openai_embedding_model}"
            )
            return OpenAIEmbeddings(
                model=self._settings.openai_embedding_model,
                api_key=self._settings.openai_api_key,
            )
        except Exception as e:
            raise LLMProviderError(
                f"Failed to initialize OpenAIEmbeddings: {e}"
            ) from e


class OllamaLLMProvider(LLMProviderBase):
    """LLM provider backed by a local Ollama server."""

    def get_chat_model(self) -> BaseChatModel:
        try:
            from langchain_ollama import ChatOllama

            logger.info(
                f"Initializing ChatOllama: {self._settings.ollama_llm_model} "
                f"@ {self._settings.ollama_base_url}"
            )
            return ChatOllama(
                model=self._settings.ollama_llm_model,
                base_url=self._settings.ollama_base_url,
                temperature=0.1,
            )
        except Exception as e:
            raise LLMProviderError(
                f"Failed to initialize ChatOllama: {e}\n"
                f"Make sure the server is running."
            ) from e

    def get_embedding_model(self) -> Embeddings:
        try:
            from langchain_ollama import OllamaEmbeddings

            logger.info(
                f"Initializing OllamaEmbeddings: {self._settings.ollama_embedding_model}"
            )
            return OllamaEmbeddings(
                model=self._settings.ollama_embedding_model,
                base_url=self._settings.ollama_base_url,
            )
        except Exception as e:
            raise LLMProviderError(
                f"Failed to initialize OllamaEmbeddings: {e}\n"
                f"Make sure the server is running."
            ) from e


_PROVIDER_REGISTRY: dict[str, type[LLMProviderBase]] = {
    LLMProvider.OPENAI.value: OpenAILLMProvider,
    LLMProvider.OLLAMA.value: OllamaLLMProvider,
}


def get_llm_provider(settings: Settings) -> LLMProviderBase:
    """
    Factory function returning the configured LLM provider instance.

    Args:
        settings: Validated application configuration.

    Returns:
        An instance of a class implementing LLMProviderBase.
    """
    provider_name = settings.llm_provider.value
    provider_class = _PROVIDER_REGISTRY.get(provider_name)

    if provider_class is None:
        raise LLMProviderError(
            f"No implementation registered for provider: '{provider_name}'. "
            f"Available providers: {list(_PROVIDER_REGISTRY.keys())}"
        )

    logger.debug(f"Selected LLM provider: {provider_name} -> {provider_class.__name__}")
    return provider_class(settings)