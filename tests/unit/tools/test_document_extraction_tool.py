"""
Unit tests for DocumentExtractionTool.

Tests the functionality of the DocumentExtractionTool for extracting text from PDFs and images.
Includes comprehensive tests for different extraction methods and error handling.
"""

import base64
import io
import os
import tempfile
import unittest.mock
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch, call

import pytest
from PIL import Image, UnidentifiedImageError

from form_filler.tools.document_extraction_tool import DocumentExtractionTool


@pytest.fixture
def sample_pdf_content():
    """Provide sample text content for mocked PDF extraction."""
    return "This is sample extracted Vietnamese text from a PDF document."


@pytest.fixture
def sample_image_content():
    """Provide sample text content for mocked image extraction."""
    return "This is sample extracted Vietnamese text from an image."


@pytest.fixture
def mock_fitz_document():
    """Create a mock for the PyMuPDF Document class."""
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "This is sample extracted Vietnamese text from a PDF document."
    mock_doc.load_page.return_value = mock_page
    mock_doc.page_count = 1
    return mock_doc


@pytest.fixture
def mock_pil_image():
    """Create a mock for the PIL Image."""
    mock_img = MagicMock()
    return mock_img


def test_init_traditional():
    """Test initialization with traditional extraction method."""
    tool = DocumentExtractionTool(extraction_method="traditional")
    assert tool.extraction_method == "traditional"
    assert tool.vision_model == "llava:7b"
    assert tool.ollama_llm is None


@patch("form_filler.tools.document_extraction_tool.OllamaLLM")
def test_init_ai(mock_ollama):
    """Test initialization with AI extraction method."""
    mock_ollama_instance = MagicMock()
    mock_ollama.return_value = mock_ollama_instance

    tool = DocumentExtractionTool(extraction_method="ai", vision_model="llava:13b")

    assert tool.extraction_method == "ai"
    assert tool.vision_model == "llava:13b"
    assert tool.ollama_llm is not None
    mock_ollama.assert_called_once_with(model="llava:13b", base_url="http://localhost:11434")


def test_invalid_file_path():
    """Test handling of invalid file paths."""
    tool = DocumentExtractionTool()
    with pytest.raises(FileNotFoundError):
        tool._run("/path/to/nonexistent/file.pdf")


@patch("pathlib.Path.exists")
def test_unsupported_file_type(mock_exists):
    """Test handling of unsupported file types."""
    mock_exists.return_value = True

    tool = DocumentExtractionTool()
    with pytest.raises(ValueError, match="Unsupported file type"):
        tool._run("/path/to/file.docx")


@patch("pathlib.Path.exists")
@patch("form_filler.tools.document_extraction_tool.fitz.open")
def test_pdf_traditional_extraction(mock_fitz_open, mock_exists, mock_fitz_document, sample_pdf_content):
    """Test traditional extraction from PDF files."""
    mock_exists.return_value = True
    mock_fitz_open.return_value = mock_fitz_document

    tool = DocumentExtractionTool(extraction_method="traditional")
    result = tool._run("/path/to/file.pdf")

    assert result == sample_pdf_content
    mock_fitz_open.assert_called_once_with("/path/to/file.pdf")


@patch("pathlib.Path.exists")
@patch("form_filler.tools.document_extraction_tool.pytesseract.image_to_string")
@patch("form_filler.tools.document_extraction_tool.Image.open")
def test_image_traditional_extraction(mock_image_open, mock_tesseract, mock_exists, mock_pil_image, sample_image_content):
    """Test traditional extraction from image files."""
    mock_exists.return_value = True
    mock_image_open.return_value = mock_pil_image
    mock_tesseract.return_value = sample_image_content

    tool = DocumentExtractionTool(extraction_method="traditional")
    result = tool._run("/path/to/file.jpg")

    assert result == sample_image_content
    mock_image_open.assert_called_once_with("/path/to/file.jpg")
    mock_tesseract.assert_called_once_with(mock_pil_image, lang="vie")


@patch("pathlib.Path.exists")
@patch("form_filler.tools.document_extraction_tool.fitz.open")
@patch("form_filler.tools.document_extraction_tool.tempfile.NamedTemporaryFile")
@patch("form_filler.tools.document_extraction_tool.Image.open")
@patch("form_filler.tools.document_extraction_tool.base64.b64encode")
def test_pdf_ai_extraction(mock_b64encode, mock_image_open, mock_temp_file, mock_fitz_open, mock_exists, mock_fitz_document, mock_pil_image):
    """Test AI-based extraction from PDF files."""
    # Setup mock for temporary file
    mock_temp = MagicMock()
    mock_temp.__enter__.return_value.name = tempfile.gettempdir() + "/temp_image.png"
    mock_temp_file.return_value = mock_temp

    # Setup mocks for PDF handling
    mock_exists.return_value = True
    mock_fitz_open.return_value = mock_fitz_document
    mock_image_open.return_value = mock_pil_image
    mock_b64encode.return_value = b"base64encodedimage"

    # Setup mock for AI model
    with patch.object(DocumentExtractionTool, "_extract_from_image_ai") as mock_extract_ai:
        mock_extract_ai.return_value = "AI extracted text"

        tool = DocumentExtractionTool(extraction_method="ai")
        result = tool._run("/path/to/file.pdf")

        assert result == "AI extracted text"
        mock_fitz_open.assert_called_once_with("/path/to/file.pdf")
        # Check that _extract_from_image_ai was called at least once
        assert mock_extract_ai.call_count > 0


@patch("pathlib.Path.exists")
@patch("form_filler.tools.document_extraction_tool.Image.open")
@patch("form_filler.tools.document_extraction_tool.base64.b64encode")
def test_image_ai_extraction(mock_b64encode, mock_image_open, mock_exists, mock_pil_image):
    """Test AI-based extraction from image files."""
    mock_exists.return_value = True
    mock_image_open.return_value = mock_pil_image
    mock_b64encode.return_value = b"base64encodedimage"

    # Create a tool with mocked Ollama LLM
    tool = DocumentExtractionTool(extraction_method="ai")
    tool.ollama_llm = MagicMock()
    tool.ollama_llm.invoke.return_value = {"content": "AI extracted text from image"}

    result = tool._run("/path/to/file.jpg")

    assert "AI extracted text from image" in result
    mock_image_open.assert_called_once_with("/path/to/file.jpg")
    tool.ollama_llm.invoke.assert_called_once()


@patch("pathlib.Path.exists")
@patch("form_filler.tools.document_extraction_tool.Image.open")
@patch("form_filler.tools.document_extraction_tool.base64.b64encode")
@patch("form_filler.tools.document_extraction_tool.pytesseract.image_to_string")
def test_image_ai_extraction_fallback(mock_tesseract, mock_b64encode, mock_image_open, mock_exists, mock_pil_image, sample_image_content):
    """Test fallback to traditional OCR when AI extraction fails."""
    mock_exists.return_value = True
    mock_image_open.return_value = mock_pil_image
    mock_b64encode.return_value = b"base64encodedimage"
    mock_tesseract.return_value = sample_image_content

    # Create a tool with mocked Ollama LLM that raises an exception
    tool = DocumentExtractionTool(extraction_method="ai")
    tool.ollama_llm = MagicMock()
    tool.ollama_llm.invoke.side_effect = Exception("AI model error")

    result = tool._run("/path/to/file.jpg")

    assert result == sample_image_content
    mock_image_open.assert_called()
    mock_tesseract.assert_called_with(mock_pil_image, lang="vie")
    tool.ollama_llm.invoke.assert_called_once()


@patch("pathlib.Path.exists")
@patch("form_filler.tools.document_extraction_tool.fitz.open")
@patch("form_filler.tools.document_extraction_tool.tempfile.NamedTemporaryFile")
@patch("form_filler.tools.document_extraction_tool.Image.open")
def test_pdf_ai_extraction_fallback(mock_image_open, mock_temp_file, mock_fitz_open, mock_exists, mock_fitz_document, mock_pil_image, sample_pdf_content):
    """Test fallback to traditional extraction when AI extraction of PDF fails."""
    # Setup mock for temporary file
    mock_temp = MagicMock()
    mock_temp.__enter__.return_value.name = tempfile.gettempdir() + "/temp_image.png"
    mock_temp_file.return_value = mock_temp

    # Setup mocks for PDF handling
    mock_exists.return_value = True
    mock_fitz_open.return_value = mock_fitz_document
    mock_image_open.return_value = mock_pil_image

    # Setup mock for AI extraction that raises an exception
    with patch.object(DocumentExtractionTool, "_extract_from_image_ai") as mock_extract_ai:
        mock_extract_ai.side_effect = Exception("AI extraction error")

        tool = DocumentExtractionTool(extraction_method="ai")
        result = tool._run("/path/to/file.pdf")

        assert result == sample_pdf_content  # Should get traditional extraction result
        mock_fitz_open.assert_called()
        assert mock_extract_ai.call_count > 0


@patch("pathlib.Path.exists")
@patch("form_filler.tools.document_extraction_tool.fitz.open")
def test_pdf_with_multiple_pages(mock_fitz_open, mock_exists):
    """Test extraction from a multi-page PDF document."""
    # Setup file exists mock
    mock_exists.return_value = True

    # Setup mock for PDF with multiple pages
    mock_doc = MagicMock()
    mock_doc.page_count = 3

    # Create page mocks with different content
    page1 = MagicMock()
    page1.get_text.return_value = "Page 1 content"

    page2 = MagicMock()
    page2.get_text.return_value = "Page 2 content"

    page3 = MagicMock()
    page3.get_text.return_value = "Page 3 content"

    # Configure mock document to return different pages
    mock_doc.load_page.side_effect = [page1, page2, page3]
    mock_fitz_open.return_value = mock_doc

    # Run extraction
    tool = DocumentExtractionTool(extraction_method="traditional")
    result = tool._run("/path/to/multipage.pdf")

    # Verify results
    assert "Page 1 content" in result
    assert "Page 2 content" in result
    assert "Page 3 content" in result
    assert mock_doc.load_page.call_count == 3


@patch("pathlib.Path.exists")
@patch("form_filler.tools.document_extraction_tool.Image.open")
@patch("form_filler.tools.document_extraction_tool.OllamaLLM")
def test_extract_from_image_ai_response_formats(mock_ollama, mock_image_open, mock_exists, mock_pil_image):
    """Test handling of different AI response formats."""
    # Setup mocks
    mock_exists.return_value = True
    mock_image_open.return_value = mock_pil_image

    # Test string response format
    mock_ollama_instance1 = MagicMock()
    mock_ollama_instance1.invoke.return_value = "Extracted text content"
    mock_ollama.return_value = mock_ollama_instance1

    tool1 = DocumentExtractionTool(extraction_method="ai")
    result1 = tool1._run("/path/to/image.jpg")

    assert result1 == "Extracted text content"

    # Test dictionary response format (newer Ollama versions)
    mock_ollama_instance2 = MagicMock()
    mock_ollama_instance2.invoke.return_value = {"content": "AI extracted text"}
    mock_ollama.return_value = mock_ollama_instance2

    tool2 = DocumentExtractionTool(extraction_method="ai")
    result2 = tool2._run("/path/to/image.jpg")

    assert result2 == "AI extracted text"

    # Test object response format with content attribute
    mock_response = MagicMock()
    mock_response.content = "Object response content"
    mock_ollama_instance3 = MagicMock()
    mock_ollama_instance3.invoke.return_value = mock_response
    mock_ollama.return_value = mock_ollama_instance3

    tool3 = DocumentExtractionTool(extraction_method="ai")
    result3 = tool3._run("/path/to/image.jpg")

    assert result3 == "Object response content"


@patch("pathlib.Path.exists")
@patch("form_filler.tools.document_extraction_tool.Image.open")
def test_corrupted_image_handling(mock_image_open, mock_exists):
    """Test handling of corrupted or unsupported image files."""
    mock_exists.return_value = True
    mock_image_open.side_effect = UnidentifiedImageError("Cannot identify image file")

    tool = DocumentExtractionTool()

    with pytest.raises(UnidentifiedImageError, match="Cannot identify image file"):
        tool._run("/path/to/corrupted.jpg")


@patch("pathlib.Path.exists")
@patch("form_filler.tools.document_extraction_tool.fitz.open")
def test_corrupted_pdf_handling(mock_fitz_open, mock_exists):
    """Test handling of corrupted PDF files."""
    mock_exists.return_value = True
    mock_fitz_open.side_effect = Exception("Cannot open damaged PDF file")

    tool = DocumentExtractionTool()

    with pytest.raises(Exception, match="Cannot open damaged PDF file"):
        tool._run("/path/to/corrupted.pdf")


@patch("pathlib.Path.exists")
@patch("form_filler.tools.document_extraction_tool.fitz.open")
@patch("form_filler.tools.document_extraction_tool.tempfile.NamedTemporaryFile")
@patch("form_filler.tools.document_extraction_tool.Image.open")
@patch("form_filler.tools.document_extraction_tool.base64.b64encode")
def test_extract_pdf_with_empty_pages(mock_b64encode, mock_image_open, mock_temp_file, mock_fitz_open, mock_exists):
    """Test extraction from PDF with empty pages."""
    # Setup mocks
    mock_exists.return_value = True

    # Mock temporary file
    mock_temp = MagicMock()
    mock_temp.__enter__.return_value.name = tempfile.gettempdir() + "/temp_image.png"
    mock_temp_file.return_value = mock_temp

    # Create PDF mock with empty pages
    mock_doc = MagicMock()
    mock_doc.page_count = 2

    # First page is empty, second has content
    page1 = MagicMock()
    page1.get_text.return_value = ""

    page2 = MagicMock()
    page2.get_text.return_value = "Content on second page"

    mock_doc.load_page.side_effect = [page1, page2]
    mock_fitz_open.return_value = mock_doc

    # Run extraction
    tool = DocumentExtractionTool(extraction_method="traditional")
    result = tool._run("/path/to/file.pdf")

    # Should include content from non-empty pages
    assert result == "Content on second page"
    assert mock_doc.load_page.call_count == 2
