from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger

from smart_doc_search.config import Settings
from smart_doc_search.document_loader import DocumentProcessor
from smart_doc_search.exceptions import DocumentLoadError, GenerationError, VectorStoreError
from smart_doc_search.llm_factory import LLMProviderBase
from smart_doc_search.vector_store import VectorStore


# Prompt template

_REPHRASE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a precise document analysis assistant.
Answer the user's question using the context provided below.
You may reason and draw conclusions from the context, but do not use knowledge outside of it.
If the context contains relevant information, always provide an answer based on it.
Only if the context contains absolutely no relevant information, say:
"I could not find an answer to your question in the provided documents."

Context:
{context}""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


# Result dataclass


@dataclass
class RAGResult:
    """Structured result returned by the RAG engine."""

    answer: str
    source_documents: list[Document] = field(default_factory=list)
    question: str = ""


# RAG Engine


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

    def __init__(
        self,
        settings: Settings,
        provider: LLMProviderBase,
        document_processor: DocumentProcessor | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Compose the RAG pipeline from its collaborators.

        document_processor and vector_store are optional seams for testing. """
        self._settings = settings
        self._provider = provider

        self._embeddings = provider.get_embedding_model()
        self._document_processor = document_processor or DocumentProcessor(settings)
        self._vector_store = vector_store or VectorStore(settings, self._embeddings)
        self._chat_model: BaseChatModel = provider.get_chat_model()

        self._prompt_chain = self._build_chain(self._chat_model)
        logger.info("RAG engine initialized successfully.")
    # Public API
    def ingest(self, file_path: str) -> int:
        """Load, chunk, embed and store a document."""
        logger.info(f"Ingesting document: {file_path}")
        chunks = self._document_processor.load_and_split(file_path)
        count = self._vector_store.add_documents(chunks)
        logger.info(f"Ingestion complete: {count} chunks stored.")
        return count

    def query(self, question: str, chat_history: list | None = None) -> RAGResult:
        """Answer a question using the ingested document corpus and chat history.

        Retrieves relevant chunks based on a rephrased standalone question,
        formats them as context, then passes context + history + question
        directly to the prompt chain."""

        if not question.strip():
            return RAGResult(
                answer="Please provide a non-empty question.",
                question=question,
            )

        chat_history = chat_history or []
        logger.info(f"Processing query: '{question}'")

        if chat_history:
            rephrase_chain = _REPHRASE_PROMPT | self._chat_model | StrOutputParser()
            try:
                search_query = rephrase_chain.invoke({
                    "chat_history": chat_history,
                    "question": question
                })
            except Exception as e:
                raise GenerationError(
                    f"Failed to rephrase question using "
                    f"'{self._settings.llm_provider.value}': {e}"
                ) from e
            logger.debug(f"Rephrased question for search: '{search_query}'")
        else:
            search_query = question

        source_docs = self._vector_store.similarity_search(search_query)
        logger.debug(f"Retrieved {len(source_docs)} source chunks.")

        context_text = "\n\n---\n\n".join(doc.page_content for doc in source_docs)

        try:
            answer = self._prompt_chain.invoke({
                "context": context_text,
                "chat_history": chat_history,
                "question": question,
            })
        except Exception as e:
            raise GenerationError(
                f"Failed to generate an answer using "
                f"'{self._settings.llm_provider.value}': {e}"
            ) from e
        logger.info("Query answered successfully.")

        return RAGResult(
            answer=answer,
            source_documents=source_docs,
            question=question,
        )

    def get_document_count(self) -> int:
        """Return the number of chunks currently in the vector store."""
        return self._vector_store.get_document_count()

    def clear_documents(self) -> None:
        """Remove all documents from the vector store."""
        self._vector_store.clear()
        logger.info("All documents cleared from the vector store.")

    # Private helpers
    @staticmethod
    def _build_chain(chat_model: BaseChatModel):
        """Compose the prompt-only LCEL chain (retrieval handled externally)."""
        return _RAG_PROMPT | chat_model | StrOutputParser()