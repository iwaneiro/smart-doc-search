"""
Streamlit web interface for Smart Doc Search.

Provides a chat-style UI for document upload, ingestion,
and question answering over the uploaded corpus.
"""

from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
from loguru import logger

from smart_doc_search.config import LLMProvider, Settings, get_settings
from smart_doc_search.exceptions import (
    ConfigurationError,
    DocumentLoadError,
    GenerationError,
    LLMProviderError,
    VectorStoreError,
)
from smart_doc_search.llm_factory import get_llm_provider
from smart_doc_search.rag_engine import RAGEngine, RAGResult


# Page configuration — must be the first Streamlit call
st.set_page_config(
    page_title="Smart Doc Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)



# Session state helpers
def _init_session_state() -> None:
    """Initialize Streamlit session state with default values."""
    defaults = {
        "engine": None,           # RAGEngine instance
        "messages": [],           # Chat history
        "ingested_files": [],     # Names of successfully ingested files
        "llm_provider": LLMProvider.OLLAMA.value,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_or_create_engine(settings: Settings) -> RAGEngine | None:
    """Return the cached RAGEngine or create a new one.

    Creates a new engine when the LLM provider changes or on first load."""
    current_provider = settings.llm_provider.value

    # Rebuild engine if provider changed
    if (
        st.session_state.engine is None
        or st.session_state.llm_provider != current_provider
    ):
        try:
            with st.spinner("Initializing LLM engine..."):
                provider = get_llm_provider(settings)
                st.session_state.engine = RAGEngine(settings, provider)
                st.session_state.llm_provider = current_provider
                st.session_state.ingested_files = []
                st.session_state.messages = []
                logger.info(f"Engine initialized with provider: {current_provider}")
        except (ConfigurationError, LLMProviderError) as e:
            st.error(f"❌ Failed to initialize engine: {e}")
            logger.error(f"Engine initialization failed: {e}")
            return None

    return st.session_state.engine



# Sidebar
def _render_sidebar(settings: Settings) -> Settings:
    """Render the sidebar with configuration and file upload controls."""
    with st.sidebar:
        st.title("⚙️ Configuration")

        #  LLM Provider switcher
        st.subheader("LLM Provider")
        provider_choice = st.radio(
            label="Select provider:",
            options=[LLMProvider.OLLAMA.value, LLMProvider.OPENAI.value],
            index=0 if settings.is_ollama else 1,
            help="Ollama runs locally for free. OpenAI requires an API key.",
        )

        # Provider is stored per-session, never mutated globally.
        if provider_choice != settings.llm_provider.value:
            settings = settings.model_copy(
                update={"llm_provider": LLMProvider(provider_choice)}
            )

        if settings.is_ollama:
            st.info("🦙 Using Ollama (local, free)\n\nMake sure Ollama is running:\n`brew services start ollama`")
        else:
            st.info("🤖 Using OpenAI API\n\nMake sure OPENAI_API_KEY is set in .env")

        st.divider()

        #  Document upload
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            label="Choose files to upload:",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, Markdown",
        )

        if uploaded_files:
            _handle_file_upload(uploaded_files, settings)

        #  Ingested files list
        if st.session_state.ingested_files:
            st.divider()
            st.subheader("✅ Ingested Documents")
            for fname in st.session_state.ingested_files:
                st.markdown(f"- `{fname}`")

        #  Clear button
        if st.session_state.engine and st.session_state.ingested_files:
            st.divider()
            if st.button("🗑️ Clear all documents", use_container_width=True):
                _clear_documents()

        #  Stats
        if st.session_state.engine:
            st.divider()
            count = st.session_state.engine.get_document_count()
            st.metric("Chunks in vector store", count)

    return settings


def _handle_file_upload(uploaded_files, settings: Settings) -> None:
    """Save uploaded files to a temp directory and ingest them."""
    import tempfile
    from pathlib import Path

    engine = _get_or_create_engine(settings)
    if engine is None:
        return

    for uploaded_file in uploaded_files:
        # Skip already ingested files
        if uploaded_file.name in st.session_state.ingested_files:
            continue

        with st.spinner(f"Ingesting `{uploaded_file.name}`..."):
            # Write to a temporary file so DocumentProcessor can read it
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                count = engine.ingest(tmp_path)
                st.session_state.ingested_files.append(uploaded_file.name)
                st.success(f"✅ `{uploaded_file.name}` ingested ({count} chunks)")
                logger.info(f"Ingested '{uploaded_file.name}': {count} chunks")

            except DocumentLoadError as e:
                st.error(f"❌ Failed to load `{uploaded_file.name}`: {e}")
            except VectorStoreError as e:
                st.error(f"❌ Failed to store `{uploaded_file.name}`: {e}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)


def _clear_documents() -> None:
    """Clear all documents from the vector store and reset session state."""
    try:
        st.session_state.engine.clear_documents()
        st.session_state.ingested_files = []
        st.session_state.messages = []
        st.success("✅ All documents cleared.")
        st.rerun()
    except VectorStoreError as e:
        st.error(f"❌ Failed to clear documents: {e}")


# Main chat interface
def _render_chat(settings: Settings) -> None:
    """Render the main chat interface for question answering."""
    st.title("🔍 Smart Doc Search")
    st.caption("RAG-powered document search engine with LLM integration")

    # Guard: no documents ingested yet
    if not st.session_state.ingested_files:
        st.info(
            "👈 Upload one or more documents in the sidebar to get started.\n\n"
            "Supported formats: **PDF**, **TXT**, **Markdown**"
        )
        return

    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                _render_sources(message["sources"])

    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        _handle_query(question, settings)


def _handle_query(question: str, settings: Settings) -> None:
    """Process a user question and display the answer with sources."""
    engine = _get_or_create_engine(settings)
    if engine is None:
        return

    # Prepare chat history in LangChain format
    langchain_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langchain_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_history.append(AIMessage(content=msg["content"]))

    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and display answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Pass chat history to the engine
                result: RAGResult = engine.query(
                    question,
                    chat_history=langchain_history
                )
                st.markdown(result.answer)

                if result.source_documents:
                    _render_sources(result.source_documents)

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "sources": result.source_documents,
                })
            except VectorStoreError as e:
                error_msg = f"❌ Search failed: {e}"
                st.error(error_msg)
                logger.error(f"Query failed: {e}")

            except GenerationError as e:
                error_msg = f"❌ Answer generation failed: {e}"
                st.error(error_msg)
                logger.error(f"Generation failed: {e}")


def _render_sources(source_documents) -> None:
    """Render retrieved source chunks in a collapsible expander."""
    with st.expander(f"📚 Sources ({len(source_documents)} chunks)", expanded=False):
        for i, doc in enumerate(source_documents, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f" — page {page + 1}" if page != "" else ""

            st.markdown(f"**Chunk {i}** — `{source}`{page_info}")
            st.markdown(doc.page_content)
            if i < len(source_documents):
                st.divider()



# Entry point
def main() -> None:
    """Main entry point for the Streamlit application."""
    _init_session_state()

    try:
        settings = get_settings()
    except ConfigurationError as e:
        st.error(f"❌ Configuration error: {e}")
        st.stop()

    settings = _render_sidebar(settings)
    _render_chat(settings)


if __name__ == "__main__":
    main()