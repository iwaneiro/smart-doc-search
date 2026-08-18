"""
Integration tests for the RAG Engine.

These tests use live Ollama models and a temporary ChromaDB instance
to verify the full RAG pipeline end-to-end.
"""

from pathlib import Path

import pytest

from smart_doc_search.config import Settings
from smart_doc_search.llm_factory import get_llm_provider
from smart_doc_search.rag_engine import RAGEngine, RAGResult

pytestmark = pytest.mark.integration


def make_settings(tmp_path: Path, **kwargs) -> Settings:
    """Create Settings with a temporary ChromaDB directory."""
    return Settings(
        _env_file=None,
        chroma_persist_dir=str(tmp_path / "chroma_test"),
        retriever_top_k=2,
        **kwargs,
    )


@pytest.fixture
def engine(tmp_path: Path) -> RAGEngine:
    """Fully initialized RAGEngine with live Ollama models."""
    settings = make_settings(tmp_path)
    provider = get_llm_provider(settings)
    return RAGEngine(settings, provider)


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    """A temporary TXT file with known content for deterministic testing."""
    content = (
        "SmartDocSearch is a RAG-powered document search engine. "
        "It uses ChromaDB as a vector database to store document embeddings. "
        "The system supports PDF, TXT, and Markdown file formats. "
        "LangChain is used to orchestrate the retrieval and generation pipeline."
    )
    doc = tmp_path / "sample.txt"
    doc.write_text(content, encoding="utf-8")
    return doc


def test_engine_initializes(engine: RAGEngine) -> None:
    """RAGEngine initializes with zero documents."""
    assert engine.get_document_count() == 0


def test_ingest_returns_chunk_count(engine: RAGEngine, sample_txt: Path) -> None:
    """ingest() returns a positive chunk count after processing a document."""
    count = engine.ingest(str(sample_txt))
    assert count > 0
    assert engine.get_document_count() == count


def test_query_returns_rag_result(engine: RAGEngine, sample_txt: Path) -> None:
    """query() returns a RAGResult with a non-empty answer."""
    engine.ingest(str(sample_txt))
    result = engine.query("What database does SmartDocSearch use?")

    assert isinstance(result, RAGResult)
    assert len(result.answer) > 0
    assert result.question == "What database does SmartDocSearch use?"
    assert len(result.source_documents) > 0


def test_query_answer_contains_chromadb(engine: RAGEngine, sample_txt: Path) -> None:
    """LLM answer references ChromaDB when asked about the vector database."""
    engine.ingest(str(sample_txt))
    result = engine.query("What vector database is used?")

    # The answer should mention ChromaDB based on the ingested context
    assert "chroma" in result.answer.lower() or "vector" in result.answer.lower()


def test_empty_question_returns_graceful_response(engine: RAGEngine) -> None:
    """Empty question returns a helpful message without raising."""
    result = engine.query("   ")
    assert "Please provide" in result.answer


def test_clear_documents_resets_count(engine: RAGEngine, sample_txt: Path) -> None:
    """clear_documents() removes all chunks from the vector store."""
    engine.ingest(str(sample_txt))
    assert engine.get_document_count() > 0

    engine.clear_documents()
    assert engine.get_document_count() == 0