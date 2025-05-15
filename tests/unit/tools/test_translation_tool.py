"""
Unit tests for TranslationTool.

Tests the functionality of the TranslationTool for translating text from Vietnamese to English.
Includes tests for various response formats, error handling, and edge cases.
"""

import logging
from unittest.mock import MagicMock, patch, call, ANY

import pytest

from form_filler.tools.translation_tool import TranslationTool


@pytest.fixture
def sample_vietnamese_text():
    """Provide sample Vietnamese text for testing."""
    return "Xin chào, tôi là một văn bản tiếng Việt."


@pytest.fixture
def sample_english_translation():
    """Provide sample English translation result."""
    return "Hello, I am a Vietnamese text."


def test_init():
    """Test initialization of the TranslationTool."""
    with patch("form_filler.tools.translation_tool.ChatOllama") as mock_chat_ollama:
        mock_chat_ollama_instance = MagicMock()
        mock_chat_ollama.return_value = mock_chat_ollama_instance

        tool = TranslationTool(model="llama3.2:3b")

        assert tool.name == "translator"
        assert "Translate Vietnamese text to English" in tool.description
        mock_chat_ollama.assert_called_once_with(model="llama3.2:3b", base_url="http://localhost:11434")


def test_run_empty_text():
    """Test handling of empty text input."""
    tool = TranslationTool(model="llama3.2:3b")

    with pytest.raises(Exception, match="Cannot translate empty text"):
        tool._run("")

    with pytest.raises(Exception, match="Cannot translate empty text"):
        tool._run(None)


@patch("form_filler.tools.translation_tool.ChatOllama")
def test_run_successful_translation(mock_chat_ollama, sample_vietnamese_text, sample_english_translation):
    """Test successful translation from Vietnamese to English."""
    # Create mock ChatOllama instance
    mock_chat_ollama_instance = MagicMock()
    mock_chat_ollama.return_value = mock_chat_ollama_instance

    # Setup response with content attribute
    mock_response = MagicMock()
    mock_response.content = sample_english_translation
    mock_chat_ollama_instance.invoke.return_value = mock_response

    # Create tool and run translation
    tool = TranslationTool(model="llama3.2:3b")
    result = tool._run(sample_vietnamese_text)

    # Verify the result and that ChatOllama was called correctly
    assert result == sample_english_translation
    mock_chat_ollama_instance.invoke.assert_called_once()

    # Check that the correct messages were passed
    call_args = mock_chat_ollama_instance.invoke.call_args[0][0]
    assert len(call_args) == 2
    assert call_args[0]["role"] == "system"
    assert "professional Vietnamese to English translator" in call_args[0]["content"]
    assert call_args[1]["role"] == "user"
    assert sample_vietnamese_text in call_args[1]["content"]


@patch("form_filler.tools.translation_tool.ChatOllama")
def test_run_older_llm_response_format(mock_chat_ollama, sample_vietnamese_text, sample_english_translation):
    """Test handling of older LLM response format without content attribute."""
    # Create mock ChatOllama instance
    mock_chat_ollama_instance = MagicMock()
    mock_chat_ollama.return_value = mock_chat_ollama_instance

    # Setup response without content attribute (older LLM format)
    mock_chat_ollama_instance.invoke.return_value = sample_english_translation

    # Create tool and run translation
    tool = TranslationTool(model="llama3.2:3b")
    result = tool._run(sample_vietnamese_text)

    # Verify the result
    assert result == sample_english_translation
    mock_chat_ollama_instance.invoke.assert_called_once()


@patch("form_filler.tools.translation_tool.ChatOllama")
def test_run_llm_error(mock_chat_ollama, sample_vietnamese_text):
    """Test handling of LLM errors during translation."""
    # Create mock ChatOllama instance that raises an exception
    mock_chat_ollama_instance = MagicMock()
    mock_chat_ollama_instance.invoke.side_effect = Exception("LLM error")
    mock_chat_ollama.return_value = mock_chat_ollama_instance

    # Create tool and test error handling
    tool = TranslationTool(model="llama3.2:3b")

    with pytest.raises(Exception, match="LLM error"):
        tool._run(sample_vietnamese_text)

    mock_chat_ollama_instance.invoke.assert_called_once()


@patch("form_filler.tools.translation_tool.ChatOllama")
def test_different_models(mock_chat_ollama):
    """Test that different models are properly passed to ChatOllama."""
    mock_chat_ollama_instance = MagicMock()
    mock_chat_ollama.return_value = mock_chat_ollama_instance

    # Test with different models
    model_names = ["llama3.2:3b", "llama3.2:7b", "mistral:7b"]

    for model in model_names:
        TranslationTool(model=model)
        mock_chat_ollama.assert_called_with(model=model, base_url="http://localhost:11434")


@patch("form_filler.tools.translation_tool.ChatOllama")
@patch("form_filler.tools.translation_tool.logger")
def test_logging_on_error(mock_logger, mock_chat_ollama, sample_vietnamese_text):
    """Test that errors are properly logged."""
    # Create mock ChatOllama instance that raises an exception
    mock_chat_ollama_instance = MagicMock()
    mock_chat_ollama_instance.invoke.side_effect = Exception("Translation API error")
    mock_chat_ollama.return_value = mock_chat_ollama_instance

    # Create tool and test error handling with logging
    tool = TranslationTool(model="llama3.2:3b")

    with pytest.raises(Exception):
        tool._run(sample_vietnamese_text)

    # Verify logging was called with the expected message
    mock_logger.error.assert_called_once_with("Translation failed: Translation API error")


@patch("form_filler.tools.translation_tool.ChatOllama")
def test_whitespace_handling(mock_chat_ollama, sample_vietnamese_text):
    """Test handling of input text with extra whitespace."""
    # Create mock ChatOllama with response
    mock_chat_ollama_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Trimmed translation result"
    mock_chat_ollama_instance.invoke.return_value = mock_response
    mock_chat_ollama.return_value = mock_chat_ollama_instance

    # Test with text containing leading/trailing whitespace
    tool = TranslationTool(model="llama3.2:3b")

    # Add extra whitespace to the Vietnamese text
    whitespace_text = f"  \n  {sample_vietnamese_text}  \t  "
    result = tool._run(whitespace_text)

    assert result == "Trimmed translation result"

    # Verify the whitespace was preserved in the actual request
    call_args = mock_chat_ollama_instance.invoke.call_args[0][0]
    assert whitespace_text.strip() in call_args[1]["content"]


@patch("form_filler.tools.translation_tool.ChatOllama")
def test_long_text_handling(mock_chat_ollama):
    """Test handling of very long input text."""
    # Create mock ChatOllama with response
    mock_chat_ollama_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Long text translation result"
    mock_chat_ollama_instance.invoke.return_value = mock_response
    mock_chat_ollama.return_value = mock_chat_ollama_instance

    # Test with very long text
    long_text = "Xin chào " * 1000  # Create a long Vietnamese text

    tool = TranslationTool(model="llama3.2:3b")
    result = tool._run(long_text)

    assert result == "Long text translation result"

    # Verify the long text was passed to the LLM
    call_args = mock_chat_ollama_instance.invoke.call_args[0][0]
    assert long_text in call_args[1]["content"]


@patch("form_filler.tools.translation_tool.ChatOllama")
def test_special_characters(mock_chat_ollama):
    """Test handling of text with special characters and Vietnamese diacritics."""
    # Vietnamese text with diacritics and special characters
    special_text = "Việt Nam là một quốc gia ở Đông Nam Á. Thủ đô là Hà Nội! @ # $ % ^ & * ( ) _ + { } | : \" < > ?"

    # Create mock ChatOllama with response
    mock_chat_ollama_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Vietnam is a country in Southeast Asia. The capital is Hanoi! @ # $ % ^ & * ( ) _ + { } | : \" < > ?"
    mock_chat_ollama_instance.invoke.return_value = mock_response
    mock_chat_ollama.return_value = mock_chat_ollama_instance

    # Test translation with special characters
    tool = TranslationTool(model="llama3.2:3b")
    result = tool._run(special_text)

    assert "Vietnam is a country in Southeast Asia" in result
    assert "Hanoi!" in result

    # Verify the special characters were passed correctly to the LLM
    call_args = mock_chat_ollama_instance.invoke.call_args[0][0]
    assert special_text in call_args[1]["content"]


@patch("form_filler.tools.translation_tool.ChatOllama")
def test_non_text_attributes_in_response(mock_chat_ollama, sample_vietnamese_text):
    """Test handling of complex response objects with non-text attributes."""
    # Create a complex mock response with non-text attributes
    mock_response = MagicMock()
    mock_response.content = "Translated text"
    mock_response.metadata = {"model": "llama3", "tokens": 42}
    mock_response.non_serializable_attr = object()  # Add a non-serializable attribute

    mock_chat_ollama_instance = MagicMock()
    mock_chat_ollama_instance.invoke.return_value = mock_response
    mock_chat_ollama.return_value = mock_chat_ollama_instance

    # Test with complex response object
    tool = TranslationTool(model="llama3.2:3b")
    result = tool._run(sample_vietnamese_text)

    # Should extract just the content
    assert result == "Translated text"
