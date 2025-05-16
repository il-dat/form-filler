"""
Unit tests for FormFillingTool.

Tests the functionality of the FormFillingTool for filling DOCX forms with translated content.
Includes tests for AI-based field mapping, fallback mechanisms, and error handling.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from form_filler.tools.form_filling_tool import FormFillingTool


@pytest.fixture
def mock_docx_document():
    """Create a mock for the python-docx Document class with placeholders."""
    # Create the main Document mock
    mock_doc = MagicMock()

    # Create paragraph mocks with different types of placeholders
    paragraph1 = MagicMock()
    paragraph1.text = "Name: ____"

    paragraph2 = MagicMock()
    paragraph2.text = "Address [enter full address]"

    paragraph3 = MagicMock()
    paragraph3.text = "Phone number:"

    # Set up the mock document's paragraphs
    mock_doc.paragraphs = [paragraph1, paragraph2, paragraph3]

    # Create table cell mocks
    cell1 = MagicMock()
    cell1.text = "Date of birth: ____"

    cell2 = MagicMock()
    cell2.text = "Occupation [specify]"

    # Create row mocks
    row1 = MagicMock()
    row1.cells = [cell1, cell2]

    # Create table mock
    table = MagicMock()
    table.rows = [row1]

    # Set up the mock document's tables
    mock_doc.tables = [table]

    return mock_doc


@pytest.fixture
def sample_translated_text():
    """Provide sample translated text for form filling."""
    return """John Smith
123 Main Street, Apartment 4B
(555) 123-4567
January 15, 1980
Software Engineer"""


@pytest.fixture
def sample_field_mappings():
    """Provide sample field mappings JSON."""
    mappings = {
        "field_mappings": [
            {"field_text": "Name: ____", "fill_with": "John Smith", "confidence": 0.95},
            {
                "field_text": "Address [enter full address]",
                "fill_with": "123 Main Street, Apartment 4B",
                "confidence": 0.9,
            },
            {"field_text": "Phone number:", "fill_with": "(555) 123-4567", "confidence": 0.9},
            {
                "field_text": "Date of birth: ____",
                "fill_with": "January 15, 1980",
                "confidence": 0.85,
            },
            {
                "field_text": "Occupation [specify]",
                "fill_with": "Software Engineer",
                "confidence": 0.9,
            },
        ]
    }
    return json.dumps(mappings)


def test_init():
    """Test initialization of the FormFillingTool."""
    with patch("form_filler.tools.form_filling_tool.ChatOllama") as mock_chat_ollama:
        mock_chat_ollama_instance = MagicMock()
        mock_chat_ollama.return_value = mock_chat_ollama_instance

        tool = FormFillingTool(model="llama3.2:3b")

        assert tool.name == "form_filler"
        assert "Fill DOCX form fields with provided content" in tool.description
        mock_chat_ollama.assert_called_once_with(
            model="llama3.2:3b", base_url="http://localhost:11434"
        )


@patch("form_filler.tools.form_filling_tool.Document")
def test_run_file_not_found(mock_document_class):
    """Test handling of non-existent form files."""
    mock_document_class.side_effect = FileNotFoundError("File not found")

    tool = FormFillingTool(model="llama3.2:3b")

    with pytest.raises(FileNotFoundError):
        tool._run("/path/to/form.docx", "Translated text", "/path/to/output.docx")


@patch("form_filler.tools.form_filling_tool.Document")
@patch("form_filler.tools.form_filling_tool.FormAnalysisTool")
@patch("form_filler.tools.form_filling_tool.ChatOllama")
def test_run_with_ai_mappings(
    mock_chat_ollama,
    mock_form_analyzer,
    mock_document_class,
    mock_docx_document,
    sample_translated_text,
    sample_field_mappings,
):
    """Test form filling using AI-generated field mappings."""
    # Setup Document mock
    mock_document_class.return_value = mock_docx_document

    # Setup FormAnalysisTool mock
    mock_analyzer_instance = MagicMock()
    mock_analyzer_instance._run.return_value = json.dumps(
        [
            {"type": "paragraph", "text": "Name: ____", "placeholder": True},
            {"type": "paragraph", "text": "Address [enter full address]", "placeholder": True},
            {"type": "paragraph", "text": "Phone number:", "placeholder": True},
            {"type": "table_cell", "text": "Date of birth: ____", "placeholder": True},
            {"type": "table_cell", "text": "Occupation [specify]", "placeholder": True},
        ]
    )
    mock_form_analyzer.return_value = mock_analyzer_instance

    # Setup ChatOllama mock
    mock_chat_ollama_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = sample_field_mappings
    mock_chat_ollama_instance.invoke.return_value = mock_response
    mock_chat_ollama.return_value = mock_chat_ollama_instance

    tool = FormFillingTool(model="llama3.2:3b")
    result = tool._run("/path/to/form.docx", sample_translated_text, "/path/to/output.docx")

    # Verify the result
    result_dict = json.loads(result)
    assert result_dict["output_path"] == "/path/to/output.docx"
    assert result_dict["fields_filled"] > 0
    assert result_dict["total_mappings"] == 5

    # Verify that Document was called correctly
    mock_document_class.assert_called_once_with("/path/to/form.docx")

    # Verify that FormAnalysisTool was called correctly
    mock_analyzer_instance._run.assert_called_once_with("/path/to/form.docx")

    # Verify that ChatOllama was used to generate mappings
    mock_chat_ollama_instance.invoke.assert_called_once()

    # Verify that document.save was called with the output path
    mock_docx_document.save.assert_called_once_with("/path/to/output.docx")


@patch("form_filler.tools.form_filling_tool.Document")
def test_run_with_provided_mappings(
    mock_document_class, mock_docx_document, sample_translated_text, sample_field_mappings
):
    """Test form filling with provided field mappings."""
    # Setup Document mock
    mock_document_class.return_value = mock_docx_document

    tool = FormFillingTool(model="llama3.2:3b")
    result = tool._run(
        "/path/to/form.docx",
        sample_translated_text,
        "/path/to/output.docx",
        sample_field_mappings,
    )

    # Verify the result
    result_dict = json.loads(result)
    assert result_dict["output_path"] == "/path/to/output.docx"
    assert result_dict["fields_filled"] > 0
    assert result_dict["total_mappings"] == 5

    # Verify that Document was called correctly
    mock_document_class.assert_called_once_with("/path/to/form.docx")

    # Verify that document.save was called with the output path
    mock_docx_document.save.assert_called_once_with("/path/to/output.docx")


@patch("form_filler.tools.form_filling_tool.Document")
@patch("form_filler.tools.form_filling_tool.FormAnalysisTool")
@patch("form_filler.tools.form_filling_tool.ChatOllama")
def test_run_with_invalid_mappings_json(
    mock_chat_ollama,
    mock_form_analyzer,
    mock_document_class,
    mock_docx_document,
    sample_translated_text,
):
    """Test form filling with invalid field mappings JSON."""
    # Setup Document mock
    mock_document_class.return_value = mock_docx_document

    # Setup ChatOllama mock to return invalid JSON
    mock_chat_ollama_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "{invalid json"
    mock_chat_ollama_instance.invoke.return_value = mock_response
    mock_chat_ollama.return_value = mock_chat_ollama_instance

    # Setup FormAnalysisTool mock
    mock_analyzer_instance = MagicMock()
    mock_analyzer_instance._run.return_value = json.dumps(
        [
            {"type": "paragraph", "text": "Name: ____", "placeholder": True},
            {"type": "paragraph", "text": "Address [enter full address]", "placeholder": True},
        ]
    )
    mock_form_analyzer.return_value = mock_analyzer_instance

    # Setup mock for _create_fallback_mappings
    with patch.object(FormFillingTool, "_create_fallback_mappings") as mock_fallback:
        mock_fallback.return_value = [
            {"field_text": "Name: ____", "fill_with": "John Smith", "confidence": 0.5},
            {
                "field_text": "Address [enter full address]",
                "fill_with": "123 Main Street",
                "confidence": 0.5,
            },
        ]

        tool = FormFillingTool(model="llama3.2:3b")
        result = tool._run(
            "/path/to/form.docx", sample_translated_text, "/path/to/output.docx"
        )

        # Verify the result indicates successful fallback
        result_dict = json.loads(result)
        assert result_dict["output_path"] == "/path/to/output.docx"
        assert result_dict["fields_filled"] > 0

        # Verify that _create_fallback_mappings was called
        mock_fallback.assert_called_once()


@patch("form_filler.tools.form_filling_tool.Document")
@patch("form_filler.tools.form_filling_tool.FormAnalysisTool")
@patch("form_filler.tools.form_filling_tool.ChatOllama")
def test_ai_mapping_error(
    mock_chat_ollama,
    mock_form_analyzer,
    mock_document_class,
    mock_docx_document,
    sample_translated_text,
):
    """Test handling of AI mapping errors by using fallback."""
    # Setup Document mock
    mock_document_class.return_value = mock_docx_document

    # Setup FormAnalysisTool mock
    mock_analyzer_instance = MagicMock()
    mock_analyzer_instance._run.return_value = json.dumps(
        [
            {"type": "paragraph", "text": "Name: ____", "placeholder": True},
            {"type": "paragraph", "text": "Address [enter full address]", "placeholder": True},
        ]
    )
    mock_form_analyzer.return_value = mock_analyzer_instance

    # Setup ChatOllama mock to throw an exception
    mock_chat_ollama_instance = MagicMock()
    mock_chat_ollama_instance.invoke.side_effect = Exception("LLM error")
    mock_chat_ollama.return_value = mock_chat_ollama_instance

    # Setup mock for _create_fallback_json
    with patch.object(FormFillingTool, "_create_fallback_json") as mock_fallback_json:
        mock_fallback_json.return_value = json.dumps(
            {
                "field_mappings": [
                    {"field_text": "Name: ____", "fill_with": "John Smith", "confidence": 0.5},
                    {
                        "field_text": "Address [enter full address]",
                        "fill_with": "123 Main Street",
                        "confidence": 0.5,
                    },
                ]
            }
        )

        tool = FormFillingTool(model="llama3.2:3b")
        result = tool._run(
            "/path/to/form.docx", sample_translated_text, "/path/to/output.docx"
        )

        # Verify the result indicates successful fallback
        result_dict = json.loads(result)
        assert result_dict["output_path"] == "/path/to/output.docx"
        assert result_dict["fields_filled"] > 0

        # Verify that _create_fallback_json was called
        mock_fallback_json.assert_called_once()


def test_create_fallback_mappings():
    """Test the _create_fallback_mappings method."""
    # Setup mock document
    mock_doc = MagicMock()
    mock_paragraph1 = MagicMock()
    mock_paragraph1.text = "Name: ____"
    mock_paragraph2 = MagicMock()
    mock_paragraph2.text = "Address [enter full address]"
    mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2]

    sample_content = "John Smith\n123 Main Street"

    tool = FormFillingTool(model="llama3.2:3b")
    mappings = tool._create_fallback_mappings(mock_doc, sample_content)

    # Verify the mappings
    assert len(mappings) == 2
    assert mappings[0]["field_text"] == "Name: ____"
    assert mappings[0]["fill_with"] == "John Smith"
    assert mappings[1]["field_text"] == "Address [enter full address]"
    assert mappings[1]["fill_with"] == "123 Main Street"


def test_create_fallback_json():
    """Test the _create_fallback_json method."""
    form_fields = json.dumps(
        [
            {"type": "paragraph", "text": "Name: ____", "placeholder": True},
            {"type": "paragraph", "text": "Address [enter full address]", "placeholder": True},
        ]
    )

    sample_content = "John Smith\n123 Main Street"

    tool = FormFillingTool(model="llama3.2:3b")
    result = tool._create_fallback_json(form_fields, sample_content)

    # Verify the result is valid JSON
    mappings_data = json.loads(result)

    # Check the structure
    assert "field_mappings" in mappings_data
    assert len(mappings_data["field_mappings"]) == 2
    assert mappings_data["field_mappings"][0]["field_text"] == "Name: ____"
    assert mappings_data["field_mappings"][0]["fill_with"] == "John Smith"


@patch("form_filler.tools.form_filling_tool.Document")
@patch("form_filler.tools.form_filling_tool.logger")
def test_logging_on_error(mock_logger, mock_document_class):
    """Test that errors are properly logged."""
    # Make the document class raise an exception
    mock_document_class.side_effect = Exception("Form filling error")

    # Create tool and run with error
    tool = FormFillingTool(model="llama3.2:3b")

    with pytest.raises(Exception):
        tool._run("/path/to/form.docx", "Translated text", "/path/to/output.docx")

    # Verify logging was called with the expected message
    mock_logger.error.assert_called_once_with("Form filling failed: Form filling error")


@patch("form_filler.tools.form_filling_tool.Document")
@patch("form_filler.tools.form_filling_tool.FormAnalysisTool")
@patch("form_filler.tools.form_filling_tool.ChatOllama")
def test_complex_paragraph_replacement(mock_chat_ollama, mock_form_analyzer, mock_document_class):
    """Test replacing placeholder text with content in more complex paragraph structures."""
    # Setup Document mock with paragraphs
    mock_doc = MagicMock()

    paragraph1 = MagicMock()
    paragraph1.text = "Name: ____ (enter full name)"

    paragraph2 = MagicMock()
    paragraph2.text = "This paragraph should be unchanged"

    mock_doc.paragraphs = [paragraph1, paragraph2]
    mock_doc.tables = []

    mock_document_class.return_value = mock_doc

    # Setup FormAnalysisTool mock
    mock_analyzer_instance = MagicMock()
    mock_analyzer_instance._run.return_value = json.dumps(
        [{"type": "paragraph", "text": "Name: ____ (enter full name)", "placeholder": True}]
    )
    mock_form_analyzer.return_value = mock_analyzer_instance

    # Setup ChatOllama mock
    mock_chat_ollama_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(
        {
            "field_mappings": [
                {
                    "field_text": "Name: ____ (enter full name)",
                    "fill_with": "John Smith",
                    "confidence": 0.95,
                }
            ]
        }
    )
    mock_chat_ollama_instance.invoke.return_value = mock_response
    mock_chat_ollama.return_value = mock_chat_ollama_instance

    tool = FormFillingTool(model="llama3.2:3b")
    result = tool._run("/path/to/form.docx", "John Smith", "/path/to/output.docx")

    # Verify the result
    result_dict = json.loads(result)
    assert result_dict["output_path"] == "/path/to/output.docx"
    assert result_dict["fields_filled"] > 0

    # Verify that paragraph text was correctly replaced
    paragraph1.clear.assert_called_once()
    paragraph1.add_run.assert_called_once()


@patch("form_filler.tools.form_filling_tool.Document")
def test_invalid_json_mapping_format(mock_document_class):
    """Test handling of invalid JSON format in field mappings."""
    # Setup Document mock
    mock_doc = MagicMock()
    mock_document_class.return_value = mock_doc

    # Setup fallback method
    with patch.object(FormFillingTool, "_create_fallback_mappings") as mock_fallback:
        mock_fallback.return_value = [
            {"field_text": "Name: ____", "fill_with": "John Smith", "confidence": 0.5}
        ]

        # Create a local json.loads that will raise an exception only for a specific input
        original_loads = json.loads

        def patched_loads(s, *args, **kwargs):
            if s == "Invalid JSON mapping":
                raise json.JSONDecodeError("Invalid JSON", "", 0)
            return original_loads(s, *args, **kwargs)

        # Apply the patch to the tool's json module
        with patch("form_filler.tools.form_filling_tool.json.loads", side_effect=patched_loads):
            tool = FormFillingTool(model="llama3.2:3b")
            result = tool._run(
                "/path/to/form.docx", "John Smith", "/path/to/output.docx", "Invalid JSON mapping"
            )

            # Verify fallback method was called
            mock_fallback.assert_called_once()

            # Use the original json.loads to verify the result
            result_dict = original_loads(result)
            assert result_dict["output_path"] == "/path/to/output.docx"


@patch("form_filler.tools.form_filling_tool.Document")
def test_empty_form_fields(mock_document_class):
    """Test handling of a document with no form fields."""
    # Setup Document mock with no form fields
    mock_doc = MagicMock()
    mock_doc.paragraphs = []
    mock_doc.tables = []
    mock_document_class.return_value = mock_doc

    tool = FormFillingTool(model="llama3.2:3b")

    # Create mappings that won't match anything
    field_mappings = json.dumps({
        "field_mappings": [
            {"field_text": "Name: ____", "fill_with": "John Smith", "confidence": 0.95}
        ]
    })

    result = tool._run(
        "/path/to/form.docx", "John Smith", "/path/to/output.docx", field_mappings
    )

    # Verify the result indicates no fields were filled
    result_dict = json.loads(result)
    assert result_dict["output_path"] == "/path/to/output.docx"
    assert result_dict["fields_filled"] == 0
    assert result_dict["total_mappings"] == 1


@patch("form_filler.tools.form_filling_tool.Document")
def test_create_fallback_mappings_insufficient_content(mock_document_class):
    """Test creating fallback mappings with less content than fields."""
    # Setup mock document with more fields than content parts
    mock_doc = MagicMock()

    paragraph1 = MagicMock()
    paragraph1.text = "Name: ____"

    paragraph2 = MagicMock()
    paragraph2.text = "Address [enter address]"

    paragraph3 = MagicMock()
    paragraph3.text = "Phone: ____"

    mock_doc.paragraphs = [paragraph1, paragraph2, paragraph3]

    # Content with only one line
    sample_content = "John Smith"

    tool = FormFillingTool(model="llama3.2:3b")
    mappings = tool._create_fallback_mappings(mock_doc, sample_content)

    # Verify mappings were created for available content only
    assert len(mappings) == 1
    assert mappings[0]["field_text"] == "Name: ____"
    assert mappings[0]["fill_with"] == "John Smith"


@patch("form_filler.tools.form_filling_tool.Document")
def test_create_fallback_json_invalid_json(mock_document_class):
    """Test creating fallback JSON when form fields input is invalid JSON."""
    # Setup with invalid JSON for form fields
    invalid_form_fields = "{invalid json"
    sample_content = "John Smith\n123 Main Street"

    tool = FormFillingTool(model="llama3.2:3b")
    result = tool._create_fallback_json(invalid_form_fields, sample_content)

    # Verify the result is valid JSON with empty mappings
    mappings_data = json.loads(result)
    assert "field_mappings" in mappings_data
    assert len(mappings_data["field_mappings"]) == 0
