"""
RAG (Retrieval-Augmented Generation) engine.

Orchestrates the full pipeline:
  1. Retrieve — find the most relevant document chunks from ChromaDB
  2. Augment  — inject retrieved context into the prompt
  3. Generate — produce an answer using the configured LLM

Built with LangChain Expression Language (LCEL) for composability
and streaming support.
"""

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from smart_doc_search.config import Settings
from smart_doc_search.document_loader import DocumentProcessor
from smart_doc_search.exceptions import DocumentLoadError, VectorStoreError
from smart_doc_search.llm_factory import LLMProviderBase
from smart_doc_search.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a precise document analysis assistant.
Answer the user's question using ONLY the context provided below.
If the answer cannot be found in the context, say exactly:
"I could not find an answer to your question in the provided documents."
Do not use any knowledge outside of the provided context.

Context:
{context}""",
        ),
        ("human", "{question}"),
    ]
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RAGResult:
    """Structured result returned by the RAG engine.

    Attributes:
        answer: Generated answer from the LLM.
        source_documents: Document chunks used as context.
        question: Original user question.
    """

    answer: str
    source_documents: list[Document] = field(default_factory=list)
    question: str = ""


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------


class RAGEngine:
    """Orchestrates the Retrieval-Augmented Generation pipeline.

    Composes DocumentProcessor, VectorStore, and LLM provider into
    a single cohesive interface. Supports document ingestion and
    question answering over the ingested corpus.

    Example:
        >>> engine = RAGEngine(settings, provider)
        >>> engine.ingest("path/to/document.pdf")
        >>> result = engine.query("What is the main topic?")
        >>> print(result.answer)
    """

    def __init__(self, settings: Settings, provider: LLMProviderBase) -> None:
        self._settings = settings
        self._provider = provider

        self._document_processor = DocumentProcessor(settings)
        self._embeddings = provider.get_embedding_model()
        self._vector_store = VectorStore(settings, self._embeddings)
        self._chat_model: BaseChatModel = provider.get_chat_model()

        # Chain is now prompt-only — retrieval happens once in query()
        self._prompt_chain = self._build_chain(self._chat_model)
        logger.info("RAG engine initialized successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, file_path: str) -> int:
        """Load, chunk, embed and store a document.

        Args:
            file_path: Path to the document to ingest (PDF, TXT, MD).

        Returns:
            Number of chunks stored in the vector store.

        Raises:
            DocumentLoadError: If the document cannot be read or parsed.
            VectorStoreError: If storing embeddings fails.
        """
        logger.info(f"Ingesting document: {file_path}")
        chunks = self._document_processor.load_and_split(file_path)
        count = self._vector_store.add_documents(chunks)
        logger.info(f"Ingestion complete: {count} chunks stored.")
        return count

    def query(self, question: str) -> RAGResult:
        """Answer a question using the ingested document corpus.

        Retrieves relevant chunks once, formats them as context,
        then passes context + question directly to the prompt chain.
        This avoids the double vector search anti-pattern.

        Args:
            question: Natural language question from the user.

        Returns:
            RAGResult containing the answer and source documents.

        Raises:
            VectorStoreError: If retrieval from ChromaDB fails.
        """
        if not question.strip():
            return RAGResult(
                answer="Please provide a non-empty question.",
                question=question,
            )

        logger.info(f"Processing query: '{question}'")

        # Single vector search — reused for both context and citations
        source_docs = self._vector_store.similarity_search(question)
        logger.debug(f"Retrieved {len(source_docs)} source chunks.")

        # Format retrieved chunks into a single context string
        context_text = "\n\n---\n\n".join(doc.page_content for doc in source_docs)

        # Prompt → LLM → parser (no retriever inside the chain)
        answer = self._prompt_chain.invoke({
            "context": context_text,
            "question": question,
        })
        logger.info("Query answered successfully.")

        return RAGResult(
            answer=answer,
            source_documents=source_docs,
            question=question,
        )

    def get_document_count(self) -> int:
        """Return the number of chunks currently in the vector store.

        Returns:
            Integer count of stored chunks.
        """
        return self._vector_store.get_document_count()

    def clear_documents(self) -> None:
        """Remove all documents from the vector store.

        Raises:
            VectorStoreError: If the clear operation fails.
        """
        self._vector_store.clear()
        logger.info("All documents cleared from the vector store.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chain(chat_model: BaseChatModel):
        """Compose the prompt-only LCEL chain (retrieval handled externally).

        Chain structure:
            {context, question} ──► prompt ──► LLM ──► parser

        Args:
            chat_model: Configured LLM for answer generation.

        Returns:
            Compiled LCEL Runnable ready to invoke.
        """
        return _RAG_PROMPT | chat_model | StrOutputParser()