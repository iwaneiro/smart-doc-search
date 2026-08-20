import pytest

from smart_doc_search.config import LLMProvider, Settings
from smart_doc_search.exceptions import ConfigurationError
from smart_doc_search.llm_factory import (
    OllamaLLMProvider,
    OpenAILLMProvider,
    get_llm_provider,
)


def make_settings(**kwargs) -> Settings:
    """Create a Settings instance bypassing the .env file."""
    return Settings(_env_file=None, **kwargs)


def test_factory_returns_ollama_provider() -> None:
    """Factory returns OllamaLLMProvider when llm_provider=ollama."""
    settings = make_settings(llm_provider=LLMProvider.OLLAMA)
    provider = get_llm_provider(settings)
    assert isinstance(provider, OllamaLLMProvider)


def test_factory_returns_openai_provider() -> None:
    """Factory returns OpenAILLMProvider when llm_provider=openai."""
    settings = make_settings(
        llm_provider=LLMProvider.OPENAI,
        openai_api_key="sk-test-key",
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, OpenAILLMProvider)


def test_openai_chat_model_raises_without_api_key() -> None:
    """OpenAI provider raises ConfigurationError when API key is missing."""
    settings = make_settings(
        llm_provider=LLMProvider.OPENAI,
        openai_api_key="",
    )
    provider = OpenAILLMProvider(settings)
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        provider.get_chat_model()


def test_openai_embedding_raises_without_api_key() -> None:
    """OpenAI provider raises ConfigurationError for embeddings without API key."""
    settings = make_settings(
        llm_provider=LLMProvider.OPENAI,
        openai_api_key="",
    )
    provider = OpenAILLMProvider(settings)
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        provider.get_embedding_model()


def test_default_provider_is_ollama() -> None:
    """Default provider is Ollama - free, requires no API key."""
    settings = make_settings()
    assert settings.llm_provider == LLMProvider.OLLAMA