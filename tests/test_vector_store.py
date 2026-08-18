from pathlib import Path

import pytest

from smart_doc_search.config import Settings
from smart_doc_search.exceptions import VectorStoreError
from smart_doc_search.llm_factory import get_llm_provider
from smart_doc_search.vector_store import VectorStore
from langchain_core.documents import Document


def make_settings(tmp_path: Path, **kwargs) -> Settings:
    """Create Settings with a temporary ChromaDB directory."""
    return Settings(
        _env_file=None,
        chroma_persist_dir=str(tmp_path / "chroma_test"),
        **kwargs,
    )


@pytest.fixture
def embeddings(tmp_path: Path):
    """Real OllamaEmbeddings fixture — requires running Ollama server."""
    settings = make_settings(tmp_path)
    provider = get_llm_provider(settings)
    return provider.get_embedding_model()


@pytest.fixture
def vector_store(tmp_path: Path, embeddings) -> VectorStore:
    """VectorStore instance backed by a temporary ChromaDB directory."""
    settings = make_settings(tmp_path)
    return VectorStore(settings, embeddings)


def test_vector_store_initializes(vector_store: VectorStore) -> None:
    """VectorStore initializes and connects to ChromaDB without errors."""
    assert vector_store.get_document_count() == 0


def test_add_documents_returns_count(vector_store: VectorStore) -> None:
    """add_documents returns the number of chunks successfully stored."""
    docs = [
        Document(page_content="Python is a programming language.", metadata={"source": "test"}),
        Document(page_content="ChromaDB is a vector database.", metadata={"source": "test"}),
    ]
    count = vector_store.add_documents(docs)
    assert count == 2
    assert vector_store.get_document_count() == 2


def test_add_empty_list_returns_zero(vector_store: VectorStore) -> None:
    """add_documents with empty list returns 0 without raising."""
    count = vector_store.add_documents([])
    assert count == 0


def test_similarity_search_returns_relevant_chunk(vector_store: VectorStore) -> None:
    """Similarity search returns the most relevant chunk for a query."""
    docs = [
        Document(page_content="Python is a high-level programming language.", metadata={"source": "test"}),
        Document(page_content="The Eiffel Tower is located in Paris, France.", metadata={"source": "test"}),
    ]
    vector_store.add_documents(docs)

    results = vector_store.similarity_search("What programming language is Python?")

    assert len(results) > 0
    assert "Python" in results[0].page_content


def test_as_retriever_returns_retriever(vector_store: VectorStore) -> None:
    """as_retriever returns a LangChain VectorStoreRetriever instance."""
    from langchain_core.vectorstores import VectorStoreRetriever
    retriever = vector_store.as_retriever()
    assert isinstance(retriever, VectorStoreRetriever)