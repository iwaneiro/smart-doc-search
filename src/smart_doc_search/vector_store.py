"""
Vector store layer backed by ChromaDB.

Manages document embeddings: adding new documents, persisting the
collection to disk, and retrieving the most semantically similar
chunks for a given query.
"""

from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from loguru import logger

from smart_doc_search.config import Settings
from smart_doc_search.exceptions import VectorStoreError


class VectorStore:
    """Manages document embeddings and similarity search via ChromaDB."""

    def __init__(self, settings: Settings, embeddings: Embeddings) -> None:
        """Manages document embeddings and similarity search via ChromaDB."""
        self._settings = settings
        self._embeddings = embeddings

        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._collection_name = settings.collection_name_for_provider

        try:
            logger.info(
                f"Connecting to ChromaDB at '{persist_dir}' "
                f"(collection: '{self._collection_name}')"
            )
            self._chroma = Chroma(
                collection_name=self._collection_name,
                embedding_function=embeddings,
                persist_directory=str(persist_dir),
            )
            logger.info("ChromaDB connection established.")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB: {e}") from e

    def add_documents(self, documents: list[Document]) -> int:
        """Embed and store document chunks. Returns the number of added chunks."""
        if not documents:
            logger.warning("add_documents called with an empty list - skipping.")
            return 0

        try:
            logger.info(f"Embedding and storing {len(documents)} chunks...")
            self._chroma.add_documents(documents)
            logger.info(f"Successfully stored {len(documents)} chunks.")
            return len(documents)
        except Exception as e:
            raise VectorStoreError(f"Failed to add documents to ChromaDB: {e}") from e

    def as_retriever(self) -> VectorStoreRetriever:
        """Return LangChain retriever interface configured with top_k."""
        logger.debug(f"Creating retriever with top_k={self._settings.retriever_top_k}")
        return self._chroma.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self._settings.retriever_top_k},
        )

    def similarity_search(self, query: str) -> list[Document]:
        """Return top-k relevant document chunks for the query."""
        try:
            results = self._chroma.similarity_search(
                query, k=self._settings.retriever_top_k
            )
            logger.debug(f"similarity_search returned {len(results)} results.")
            return results
        except Exception as e:
            raise VectorStoreError(f"Similarity search failed: {e}") from e

    def get_document_count(self) -> int:
        """Return total chunk count in the collection."""
        try:
            ids = self._chroma.get(include=[])["ids"]
        except Exception as e:
            raise VectorStoreError(f"Failed to count documents: {e}") from e
        count = len(ids)
        logger.debug(f"Collection '{self._collection_name}' has {count} chunks.")
        return count

    def clear(self) -> None:
        """Delete all documents from the collection."""
        try:
            ids = self._chroma.get(include=[])["ids"]
            if ids:
                self._chroma.delete(ids=ids)
            logger.info("Vector store cleared.")
        except Exception as e:
            raise VectorStoreError(f"Failed to clear vector store: {e}") from e