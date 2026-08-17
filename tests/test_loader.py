"""
Unit tests for the Document Loading and Chunking module.
"""

from pathlib import Path
import pytest

from smart_doc_search.config import Settings
from smart_doc_search.document_loader import DocumentProcessor
from smart_doc_search.exceptions import DocumentLoadError


def make_settings(**kwargs) -> Settings:
    """Create a Settings instance bypassing the .env file."""
    # Reuse the same pattern as in test_llm_factory.py
    return Settings(_env_file=None, **kwargs)


def test_document_processor_file_not_found() -> None:
    """Raises DocumentLoadError when the target file does not exist."""
    settings = make_settings()
    processor = DocumentProcessor(settings)

    with pytest.raises(DocumentLoadError, match="File not found"):
        processor.load_and_split("path/to/some/non_existent_file.txt")


def test_document_processor_unsupported_format(tmp_path: Path) -> None:
    """Raises DocumentLoadError for unsupported file extensions (e.g., .jpg)."""
    settings = make_settings()
    processor = DocumentProcessor(settings)

    # Create a temporary dummy image file
    dummy_file = tmp_path / "test_image.jpg"
    dummy_file.write_text("fake binary content")

    with pytest.raises(DocumentLoadError, match="Unsupported file format"):
        processor.load_and_split(dummy_file)


def test_document_processor_txt_loading_and_chunking(tmp_path: Path) -> None:
    """Successfully loads a TXT file and splits it into predictable chunks."""
    # Setup specific chunk sizes to force the splitter to act
    settings = make_settings(chunk_size=100, chunk_overlap=20)
    processor = DocumentProcessor(settings)

    # Create a temporary text file with content longer than chunk_size
    sample_text = (
        "This is the first sentence that takes up a bit of space to build volume. "
        "Here is the second sentence that is intentionally longer to ensure we definitely cross the 100 character mark. "
        "Finally, the third sentence arrives here to wrap it up and successfully create another chunk."
    )
    test_file = tmp_path / "test_doc.txt"
    test_file.write_text(sample_text, encoding="utf-8")

    # Execute
    chunks = processor.load_and_split(test_file)

    # Assertions
    assert len(chunks) > 1, "The document should be split into multiple chunks."
    assert "source" in chunks[0].metadata, "Metadata should contain the source path."
    assert chunks[0].metadata["source"] == str(test_file)
    assert "first sentence" in chunks[0].page_content