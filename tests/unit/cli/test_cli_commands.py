"""
Unit tests for CLI commands.

Tests the CLI command structure and basic functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

# Import modules directly to avoid patching issues
from form_filler.cli import cli_commands
from form_filler.cli.cli_commands import cli
from form_filler.models.processing_result import ProcessingResult

# This approach doesn't rely on attribute lookups which can differ between Python versions


# Function to verify a command exists in a Click group
def verify_command_exists(cli_group, command_name):
    """Verify a command exists in a Click group without relying on implementation details."""
    # Use list_commands API which is stable across Python versions
    return command_name in cli_group.list_commands(None)


@pytest.fixture
def runner():
    """Create a CLI runner for testing Click commands."""
    return CliRunner()


@pytest.fixture
def mock_processing_result_success():
    """Create a successful processing result."""
    return ProcessingResult(
        success=True,
        data={"output_path": "output.docx", "fields_filled": 5},
        error=None,
        metadata={"extraction_method": "traditional", "text_model": "llama3.2:3b"},
    )


@pytest.fixture
def mock_processing_result_failure():
    """Create a failed processing result."""
    return ProcessingResult(
        success=False,
        data=None,
        error="Failed to process document",
        metadata=None,
    )


def test_cli_help(runner):
    """Test the help output of the main CLI command."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Vietnamese to English Document Form Filler" in result.output
    assert "--extraction-method" in result.output
    assert "--vision-model" in result.output
    assert "--openai-api-key" in result.output
    assert "--openai-model" in result.output


def test_process_command_help(runner):
    """Test the help output of the process command."""
    result = runner.invoke(cli, ["process", "--help"])
    assert result.exit_code == 0
    assert "Process a Vietnamese document" in result.output
    assert "SOURCE" in result.output
    assert "FORM" in result.output
    assert "OUTPUT" in result.output


def test_extract_command_help(runner):
    """Test the help output of the extract command."""
    result = runner.invoke(cli, ["extract", "--help"])
    assert result.exit_code == 0
    assert "Extract text from a Vietnamese document" in result.output


def test_translate_command_help(runner):
    """Test the help output of the translate command."""
    result = runner.invoke(cli, ["translate", "--help"])
    assert result.exit_code == 0
    assert "Translate Vietnamese text to English" in result.output


def test_process_command_success(runner, mock_processing_result_success):
    """Test the process command with a successful result."""
    # Create a new patcher directly on the module we imported
    patcher = patch.object(cli_commands, "DocumentProcessingCrew")
    mock_crew = patcher.start()

    try:
        # Verify command exists to ensure cross-Python version compatibility
        assert verify_command_exists(cli, "process")

        # Setup mock
        mock_instance = MagicMock()
        mock_instance.process_document.return_value = mock_processing_result_success
        mock_crew.return_value = mock_instance

        with (
            tempfile.NamedTemporaryFile(suffix=".pdf") as source_file,
            tempfile.NamedTemporaryFile(
                suffix=".docx",
            ) as form_file,
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            output_path = Path(temp_dir) / "output.docx"

            # Run the command
            result = runner.invoke(
                cli,
                [
                    "--model",
                    "llama3.2:3b",
                    "--extraction-method",
                    "traditional",
                    "process",
                    source_file.name,
                    form_file.name,
                    str(output_path),
                ],
            )

            # Check the result
            assert result.exit_code == 0
            assert "Success!" in result.output
            assert "Fields filled: 5" in result.output

            # Verify that the crew was called correctly
            mock_crew.assert_called_once_with(
                text_model="llama3.2:3b",
                extraction_method="traditional",
                vision_model="llava:7b",  # Default value
                openai_api_key=None,  # Default value
                openai_model="gpt-4-vision-preview",  # Default value
            )
            mock_instance.process_document.assert_called_once_with(
                source_file.name,
                form_file.name,
                str(output_path),
            )
    finally:
        # Always stop the patcher, even if the test fails
        patcher.stop()


def test_process_command_with_openai(runner, mock_processing_result_success):
    """Test the process command with OpenAI extraction method."""
    # Create patcher
    patcher = patch.object(cli_commands, "DocumentProcessingCrew")
    mock_crew = patcher.start()

    try:
        # Verify command exists to ensure cross-Python version compatibility
        assert verify_command_exists(cli, "process")

        # Create a modified success result that includes OpenAI info
        openai_result = mock_processing_result_success
        openai_result.metadata["extraction_method"] = "openai"
        openai_result.metadata["openai_model"] = "gpt-4o-vision"

        mock_instance = MagicMock()
        mock_instance.process_document.return_value = openai_result
        mock_crew.return_value = mock_instance

        with (
            tempfile.NamedTemporaryFile(suffix=".pdf") as source_file,
            tempfile.NamedTemporaryFile(
                suffix=".docx",
            ) as form_file,
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            output_path = Path(temp_dir) / "output.docx"

            # Run the command with OpenAI parameters
            result = runner.invoke(
                cli,
                [
                    "--model",
                    "llama3.2:3b",
                    "--extraction-method",
                    "openai",
                    "--openai-api-key",
                    "test-api-key",
                    "--openai-model",
                    "gpt-4o-vision",
                    "process",
                    source_file.name,
                    form_file.name,
                    str(output_path),
                ],
            )

            # Check the result
            assert result.exit_code == 0
            assert "Success!" in result.output
            assert "Extraction method: openai" in result.output
            assert "Fields filled: 5" in result.output

            # Verify that the crew was called with correct OpenAI parameters
            mock_crew.assert_called_once_with(
                text_model="llama3.2:3b",
                extraction_method="openai",
                vision_model="llava:7b",  # Default value
                openai_api_key="test-api-key",
                openai_model="gpt-4o-vision",
            )
            mock_instance.process_document.assert_called_once_with(
                source_file.name,
                form_file.name,
                str(output_path),
            )
    finally:
        patcher.stop()


def test_process_command_failure(runner, mock_processing_result_failure):
    """Test the process command with a failed result."""
    # Create patcher
    patcher = patch.object(cli_commands, "DocumentProcessingCrew")
    mock_crew = patcher.start()

    try:
        # Verify command exists to ensure cross-Python version compatibility
        assert verify_command_exists(cli, "process")

        mock_instance = MagicMock()
        mock_instance.process_document.return_value = mock_processing_result_failure
        mock_crew.return_value = mock_instance

        with (
            tempfile.NamedTemporaryFile(suffix=".pdf") as source_file,
            tempfile.NamedTemporaryFile(
                suffix=".docx",
            ) as form_file,
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            output_path = Path(temp_dir) / "output.docx"

            # Run the command with mocked sys.exit to prevent test from actually exiting
            with patch("sys.exit") as mock_exit:
                result = runner.invoke(
                    cli, ["process", source_file.name, form_file.name, str(output_path)]
                )

                # Check the result
                assert "Error: Failed to process document" in result.output
                # The test was failing because sys.exit is called more than once
                # We just verify that it was called with exit code 1 at some point
                mock_exit.assert_any_call(1)
    finally:
        patcher.stop()


def test_extract_command(runner):
    """Test the extract command."""
    # Create patcher
    patcher = patch.object(cli_commands, "DocumentExtractionTool")
    mock_extraction_tool = patcher.start()

    try:
        # Verify command exists to ensure cross-Python version compatibility
        assert verify_command_exists(cli, "extract")

        mock_instance = MagicMock()
        mock_instance._run.return_value = "Extracted text content"
        mock_extraction_tool.return_value = mock_instance

        with tempfile.NamedTemporaryFile(suffix=".pdf") as source_file:
            # Run the command
            result = runner.invoke(
                cli,
                ["--extraction-method", "traditional", "extract", source_file.name],
            )

            # Check the result
            assert result.exit_code == 0
            assert "Extracting text using traditional method" in result.output
            assert "Extracted text (traditional method)" in result.output
            assert "Extracted text content" in result.output
            assert "Characters: 22" in result.output  # Correct length of "Extracted text content"

            # Verify that the tool was called correctly
            mock_extraction_tool.assert_called_once_with(
                extraction_method="traditional",
                vision_model="llava:7b",  # Default value
                openai_api_key=None,
                openai_model="gpt-4-vision-preview",
            )
            mock_instance._run.assert_called_once_with(source_file.name)
    finally:
        patcher.stop()


def test_extract_command_openai(runner):
    """Test the extract command with OpenAI extraction method."""
    # Create patcher
    patcher = patch.object(cli_commands, "DocumentExtractionTool")
    mock_extraction_tool = patcher.start()

    try:
        # Verify command exists to ensure cross-Python version compatibility
        assert verify_command_exists(cli, "extract")

        mock_instance = MagicMock()
        mock_instance._run.return_value = "OpenAI extracted text content"
        mock_extraction_tool.return_value = mock_instance

        with tempfile.NamedTemporaryFile(suffix=".pdf") as source_file:
            # Run the command with OpenAI extraction method
            result = runner.invoke(
                cli,
                [
                    "--extraction-method",
                    "openai",
                    "--openai-api-key",
                    "test-api-key",
                    "--openai-model",
                    "gpt-4o-vision",
                    "extract",
                    source_file.name,
                ],
            )

            # Check the result
            assert result.exit_code == 0
            assert "Extracting text using openai method" in result.output
            assert "OpenAI extracted text content" in result.output

            # Verify that the tool was called with correct OpenAI parameters
            mock_extraction_tool.assert_called_once_with(
                extraction_method="openai",
                vision_model="llava:7b",  # Default value doesn't change
                openai_api_key="test-api-key",
                openai_model="gpt-4o-vision",
            )
            mock_instance._run.assert_called_once_with(source_file.name)
    finally:
        patcher.stop()


def test_translate_command(runner):
    """Test the translate command."""
    # Create patcher
    patcher = patch.object(cli_commands, "TranslationTool")
    mock_translation_tool = patcher.start()

    try:
        # Verify command exists to ensure cross-Python version compatibility
        assert verify_command_exists(cli, "translate")

        mock_instance = MagicMock()
        mock_instance._run.return_value = "Translated English text"
        mock_translation_tool.return_value = mock_instance

        # Run the command
        result = runner.invoke(cli, ["--model", "llama3.2:3b", "translate", "Tiếng Việt"])

        # Check the result
        assert result.exit_code == 0
        assert "Translating with model: llama3.2:3b" in result.output
        assert "Translation:" in result.output
        assert "Translated English text" in result.output

        # Verify that the tool was called correctly
        mock_translation_tool.assert_called_once_with(model="llama3.2:3b")
        mock_instance._run.assert_called_once_with("Tiếng Việt")
    finally:
        patcher.stop()


def test_version_command(runner):
    """Test the version command."""
    # Create patcher
    patcher = patch.object(cli_commands, "show_version")
    mock_show_version = patcher.start()

    try:
        # Verify command exists to ensure cross-Python version compatibility
        assert verify_command_exists(cli, "version")

        # Mock the show_version function
        mock_show_version.return_value = None

        # Test the version command
        runner.invoke(cli, ["version"])

        # Check the call to show_version
        mock_show_version.assert_called_once()
        assert mock_show_version.call_args[0][2] is True
    finally:
        patcher.stop()


def test_version_flag(runner):
    """Test the --version flag."""
    # Instead of mocking, directly test the output of the command
    result = runner.invoke(cli, ["--version"])

    # Check the result (should contain version information)
    assert result.exit_code == 0
    assert "Form-Filler version:" in result.output
    assert "Python version:" in result.output
    assert "System:" in result.output
