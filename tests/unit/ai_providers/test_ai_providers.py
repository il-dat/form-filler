"""
Unit tests for AI Providers implementation.

Tests the functionality of the AI provider system, including provider registration,
factory creation, and basic functionality of the abstract and concrete classes.
"""

from unittest.mock import MagicMock, patch

import pytest

from form_filler.ai_providers import AIProvider, AIProviderFactory, register_provider
from form_filler.ai_providers.anthropic_provider import AnthropicProvider
from form_filler.ai_providers.ollama_provider import OllamaProvider
from form_filler.ai_providers.openai_provider import OpenAIProvider


def test_provider_registration():
    """Test that AI providers are correctly registered."""
    # Get the list of registered providers
    providers = AIProvider.list_providers()

    # Check that we have the expected providers
    expected_providers = ["ollama", "openai", "anthropic", "deepseek", "gemini"]
    for provider in expected_providers:
        assert provider in providers


def test_provider_factory():
    """Test that AI provider factory correctly creates provider instances."""
    # Test with different providers, mocking initialization
    with patch.object(OllamaProvider, "initialize") as mock_init:
        provider = AIProviderFactory.create_provider(
            provider_name="ollama", model_name="llama3:3b"
        )
        assert isinstance(provider, OllamaProvider)
        assert provider.model_name == "llama3:3b"
        mock_init.assert_called_once()


def test_provider_factory_with_unknown_provider():
    """Test that factory raises ValueError for unknown providers."""
    with pytest.raises(ValueError, match="Provider 'unknown' not registered"):
        AIProviderFactory.create_provider(provider_name="unknown", model_name="test-model")


def test_register_provider_decorator():
    """Test that the register_provider decorator works correctly."""

    # Create a test provider class
    @register_provider
    class TestProvider(AIProvider):
        provider_name = "test_provider"

        def initialize(self):
            pass

        def text_completion(self, prompt, system_prompt=None, max_tokens=1000, temperature=0.7):
            return "Test completion"

        def chat_completion(self, messages, max_tokens=1000, temperature=0.7):
            return "Test chat completion"

        def vision_completion(self, prompt, image_data, max_tokens=1000, temperature=0.7):
            return "Test vision completion"

    # Check that the provider was registered
    providers = AIProvider.list_providers()
    assert "test_provider" in providers

    # Check that we can create the provider with the factory
    provider = AIProviderFactory.create_provider(
        provider_name="test_provider", model_name="test-model"
    )
    assert isinstance(provider, TestProvider)


def test_requires_provider_name():
    """Test that provider_name is required for registration."""
    # Try to register a provider without a provider_name
    with pytest.raises(ValueError, match="must define provider_name"):

        @register_provider
        class InvalidProvider(AIProvider):
            pass


@patch("requests.get")
def test_openai_provider_initialize(mock_get):
    """Test OpenAI provider initialization."""
    # Mock successful API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    # Test with API key
    provider = OpenAIProvider(model_name="gpt-4", api_key="test-api-key")
    provider.initialize()

    # Check that requests.get was called with correct parameters
    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert args[0] == "https://api.openai.com/v1/models"
    assert kwargs["headers"]["Authorization"] == "Bearer test-api-key"

    # Test without API key
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        provider = OpenAIProvider(model_name="gpt-4", api_key=None)
        provider.initialize()


@patch("requests.post")
def test_openai_provider_chat_completion(mock_post):
    """Test OpenAI provider chat completion."""
    # Mock successful API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "Test completion"}}]}
    mock_post.return_value = mock_response

    # Create provider and run chat completion
    with patch.object(OpenAIProvider, "initialize"):
        provider = OpenAIProvider(model_name="gpt-4", api_key="test-api-key")

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, world!"},
        ]

        result = provider.chat_completion(messages)

        # Check result
        assert result == "Test completion"

        # Check request
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.openai.com/v1/chat/completions"
        assert kwargs["json"]["model"] == "gpt-4"
        assert kwargs["json"]["messages"] == messages


@patch("form_filler.ai_providers.ollama_provider.OllamaLLM")
@patch("form_filler.ai_providers.ollama_provider.ChatOllama")
def test_ollama_provider_initialize(mock_chat_ollama, mock_ollama_llm):
    """Test Ollama provider initialization."""
    # Create mocks
    mock_chat_model = MagicMock()
    mock_llm = MagicMock()
    mock_chat_ollama.return_value = mock_chat_model
    mock_ollama_llm.return_value = mock_llm

    # Create provider
    provider = OllamaProvider(model_name="llama3:3b", api_base="http://localhost:11434")
    provider.initialize()

    # Check that both LLM and chat model were initialized
    mock_ollama_llm.assert_called_once_with(model="llama3:3b", base_url="http://localhost:11434")
    mock_chat_ollama.assert_called_once_with(model="llama3:3b", base_url="http://localhost:11434")


@patch("form_filler.ai_providers.ollama_provider.ChatOllama")
def test_ollama_provider_chat_completion(mock_chat_ollama):
    """Test Ollama provider chat completion."""
    # Create mock response
    mock_chat_model = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Ollama response"
    mock_chat_model.invoke.return_value = mock_response
    mock_chat_ollama.return_value = mock_chat_model

    # Create provider and initialize the _chat_model directly for testing
    provider = OllamaProvider(model_name="llama3:3b")
    # Skip initialization and set the mocks directly
    provider._chat_model = mock_chat_model
    provider._llm = MagicMock()

    # Run chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, world!"},
    ]

    result = provider.chat_completion(messages)

    # Check result
    assert result == "Ollama response"

    # Check that chat model was called correctly
    mock_chat_model.invoke.assert_called_once_with(messages)


def test_provider_supports_vision():
    """Test that supports_vision property works correctly for different providers."""
    # Test OpenAI
    provider = OpenAIProvider(model_name="gpt-4")
    assert not provider.supports_vision

    provider = OpenAIProvider(model_name="gpt-4-vision-preview")
    assert provider.supports_vision

    # Test Ollama
    provider = OllamaProvider(model_name="llama3:3b")
    assert not provider.supports_vision

    provider = OllamaProvider(model_name="llava:7b")
    assert provider.supports_vision

    # Test Anthropic
    provider = AnthropicProvider(model_name="claude-2")
    assert not provider.supports_vision

    provider = AnthropicProvider(model_name="claude-3-opus-20240229")
    assert provider.supports_vision
