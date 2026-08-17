"""
Document loading and chunking module.

Responsible for reading files (PDF, TXT) and splitting them into
semantically meaningful chunks for vectorization.
"""

from pathlib import Path

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from smart_doc_search.config import Settings
from smart_doc_search.exceptions import DocumentLoadError


class DocumentProcessor:
    """Handles parsing and splitting of documents for RAG ingestion."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def _load_pdf(self, path: Path) -> list[Document]:
        """Load a PDF file using pypdf directly.

        Args:
            path: Path to the PDF file.

        Returns:
            List of Documents, one per page.
        """
        reader = pypdf.PdfReader(str(path))
        docs = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path), "page": page_num},
                )
            )
        logger.debug(f"Loaded {len(docs)} pages from {path.name}")
        return docs

    def _load_text(self, path: Path) -> list[Document]:
        """Load a plain text or Markdown file.

        Args:
            path: Path to the TXT or MD file.

        Returns:
            Single-element list with the full file content as a Document.
        """
        text = path.read_text(encoding="utf-8")
        return [Document(page_content=text, metadata={"source": str(path)})]

    def load_and_split(self, file_path: str | Path) -> list[Document]:
        """
        Load a document and split it into smaller overlapping chunks.

        Args:
            file_path: Path to the document (PDF, TXT).

        Returns:
            A list of LangChain Document objects ready to be embedded.

        Raises:
            DocumentLoadError: If the file format is unsupported or loading fails.
        """
        path = Path(file_path)
        if not path.exists():
            raise DocumentLoadError(f"File not found: {path.absolute()}")

        logger.info(f"Loading document: {path.name}")

        try:
            # Select appropriate loader based on file extension
            if path.suffix.lower() == ".pdf":
                docs = self._load_pdf(path)
            elif path.suffix.lower() in {".txt", ".md"}:
                # Markdown treated as plain text — sufficient for this scope
                docs = self._load_text(path)
            else:
                raise DocumentLoadError(
                    f"Unsupported file format: {path.suffix}. "
                    "Supported formats: .pdf, .txt, .md"
                )

            # Split the document into chunks
            chunks = self._text_splitter.split_documents(docs)
            logger.info(
                f"Split {path.name} into {len(chunks)} chunks "
                f"(size: {self._settings.chunk_size}, overlap: {self._settings.chunk_overlap})"
            )
            return chunks

        except DocumentLoadError:
            raise
        except Exception as e:
            raise DocumentLoadError(f"Failed to process document '{path.name}': {e}") from e