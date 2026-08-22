"""
Integration tests for the RAG Engine.

These tests use live Ollama models and a temporary ChromaDB instance
to verify the full RAG pipeline end-to-end.
"""

from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, AIMessage

from smart_doc_search.config import Settings
from smart_doc_search.document_loader import DocumentProcessor
from smart_doc_search.exceptions import GenerationError, VectorStoreError
from smart_doc_search.llm_factory import LLMProviderBase
from smart_doc_search.rag_engine import RAGEngine, RAGResult
from smart_doc_search.vector_store import VectorStore


def make_settings(**kwargs) -> Settings:
    """Create a Settings instance bypassing the .env file."""
    return Settings(_env_file=None, **kwargs)


def make_provider(responses: list[str]) -> Mock:
    """Fake LLMProviderBase returning a FakeListChatModel with canned replies."""
    provider = Mock(spec=LLMProviderBase)
    provider.get_chat_model.return_value = FakeListChatModel(responses=responses)
    provider.get_embedding_model.return_value = Mock()
    return provider


def make_engine(
    responses: list[str],
    document_processor: Mock | None = None,
    vector_store: Mock | None = None,
) -> RAGEngine:
    settings = make_settings()
    provider = make_provider(responses)
    return RAGEngine(
        settings,
        provider,
        document_processor=document_processor or Mock(spec=DocumentProcessor),
        vector_store=vector_store or Mock(spec=VectorStore),
    )


# ingest()
def test_ingest_delegates_to_document_processor_and_vector_store() -> None:
    """ingest() loads+splits via DocumentProcessor, then stores via VectorStore."""
    chunks = [Document(page_content="chunk 1"), Document(page_content="chunk 2")]
    doc_processor = Mock(spec=DocumentProcessor)
    doc_processor.load_and_split.return_value = chunks
    vector_store = Mock(spec=VectorStore)
    vector_store.add_documents.return_value = 2

    engine = make_engine([], document_processor=doc_processor, vector_store=vector_store)
    count = engine.ingest("some/file.txt")

    assert count == 2
    doc_processor.load_and_split.assert_called_once_with("some/file.txt")
    vector_store.add_documents.assert_called_once_with(chunks)


# query() — no chat history
def test_query_with_no_history_skips_rephrase_and_returns_answer() -> None:
    """Without chat history, the raw question is used for retrieval (no rephrase call)."""
    retrieved = [Document(page_content="relevant context", metadata={"source": "doc.txt"})]
    vector_store = Mock(spec=VectorStore)
    vector_store.similarity_search.return_value = retrieved

    engine = make_engine(["final answer"], vector_store=vector_store)
    result = engine.query("What is X?")

    assert isinstance(result, RAGResult)
    assert result.answer == "final answer"
    assert result.source_documents == retrieved
    assert result.question == "What is X?"
    vector_store.similarity_search.assert_called_once_with("What is X?")


def test_query_empty_question_returns_early_without_touching_vector_store() -> None:
    """A blank/whitespace question short-circuits before any retrieval or LLM call."""
    vector_store = Mock(spec=VectorStore)

    engine = make_engine([], vector_store=vector_store)
    result = engine.query("   ")

    assert "Please provide" in result.answer
    vector_store.similarity_search.assert_not_called()


# query() — with chat history (rephrase step)
def test_query_with_history_rephrases_before_retrieval() -> None:
    """With chat history, the question is rephrased first and the rephrased
    version — not the original — is used for the similarity search."""
    vector_store = Mock(spec=VectorStore)
    vector_store.similarity_search.return_value = []
    history = [HumanMessage(content="Tell me about RAG"), AIMessage(content="It's...")]

    # First response satisfies the rephrase chain, second the final answer chain.
    engine = make_engine(
        ["standalone rephrased question", "final answer"], vector_store=vector_store
    )
    result = engine.query("What about its limitations?", chat_history=history)

    vector_store.similarity_search.assert_called_once_with("standalone rephrased question")
    assert result.answer == "final answer"


# query() — error propagation


def test_query_propagates_vector_store_error() -> None:
    """VectorStoreError from similarity_search is not swallowed by the engine."""
    vector_store = Mock(spec=VectorStore)
    vector_store.similarity_search.side_effect = VectorStoreError("chroma is down")

    engine = make_engine(["irrelevant"], vector_store=vector_store)

    with pytest.raises(VectorStoreError, match="chroma is down"):
        engine.query("What is X?")


def test_query_wraps_generation_failure_in_generation_error() -> None:
    """If the LLM invocation fails, RAGEngine raises GenerationError, not the raw exception."""
    vector_store = Mock(spec=VectorStore)
    vector_store.similarity_search.return_value = []

    engine = make_engine([], vector_store=vector_store)
    # Force the built chain to fail regardless of LCEL internals.
    engine._prompt_chain = Mock(invoke=Mock(side_effect=RuntimeError("model unreachable")))

    with pytest.raises(GenerationError, match="model unreachable"):
        engine.query("What is X?")


# get_document_count() / clear_documents()
def test_get_document_count_delegates_to_vector_store() -> None:
    vector_store = Mock(spec=VectorStore)
    vector_store.get_document_count.return_value = 42

    engine = make_engine([], vector_store=vector_store)

    assert engine.get_document_count() == 42


def test_clear_documents_delegates_to_vector_store() -> None:
    vector_store = Mock(spec=VectorStore)

    engine = make_engine([], vector_store=vector_store)
    engine.clear_documents()

    vector_store.clear.assert_called_once()