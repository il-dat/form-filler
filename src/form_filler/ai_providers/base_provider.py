#!/usr/bin/env python3
"""
Base AI Provider for form-filler.

Defines the abstract base class and factory for AI providers.
"""

import abc
import logging
from typing import Any, ClassVar

# Setup logging
logger = logging.getLogger(__name__)


class AIProvider(abc.ABC):
    """Abstract base class for AI providers."""

    # Class variable to store provider types
    _providers: ClassVar[dict[str, type["AIProvider"]]] = {}

    # Provider name (lowercase)
    provider_name: str = ""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the AI provider.

        Args:
            model_name: The name of the model to use
            api_key: API key for the provider (if needed)
            api_base: Base URL for the API (if needed)
            **kwargs: Additional provider-specific parameters
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.kwargs = kwargs

        # Store the client instance
        self._client = None

    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the provider's client/connection.

        This method should be called to establish any necessary connections
        or initialize client libraries.
        """
        pass

    @abc.abstractmethod
    def text_completion(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a text completion.

        Args:
            prompt: The prompt to generate a completion for
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)

        Returns:
            The generated text completion
        """
        pass

    @abc.abstractmethod
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a chat completion.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)

        Returns:
            The generated chat completion text
        """
        pass

    @abc.abstractmethod
    def vision_completion(
        self,
        prompt: str,
        image_data: str | bytes,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion based on an image.

        Args:
            prompt: The text prompt to accompany the image
            image_data: The image data (base64 string or bytes)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            The generated text based on the image
        """
        pass

    @property
    def supports_vision(self) -> bool:
        """Check if this provider supports vision capabilities.

        Returns:
            True if vision capabilities are supported, False otherwise
        """
        return False

    @property
    def supports_streaming(self) -> bool:
        """Check if this provider supports streaming responses.

        Returns:
            True if streaming is supported, False otherwise
        """
        return False

    @property
    def requires_api_key(self) -> bool:
        """Check if this provider requires an API key.

        Returns:
            True if an API key is required, False otherwise
        """
        return False

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def get_provider_class(cls, provider_name: str) -> type["AIProvider"]:
        """Get the provider class for a given name.

        Args:
            provider_name: The name of the provider

        Returns:
            The provider class

        Raises:
            ValueError: If the provider is not registered
        """
        provider_name_lower = provider_name.lower()
        if provider_name_lower not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Provider '{provider_name}' not registered. Available providers: {available}"
            )
        return cls._providers[provider_name_lower]


def register_provider(cls: type[AIProvider]) -> type[AIProvider]:
    """Register a provider class.

    This decorator registers a provider class in the central registry.

    Args:
        cls: The provider class to register

    Returns:
        The original class (unchanged)

    Example:
        @register_provider
        class MyProvider(AIProvider):
            provider_name = "myprovider"
            ...
    """
    if not cls.provider_name:
        raise ValueError(f"Provider class {cls.__name__} must define provider_name")

    provider_name_lower = cls.provider_name.lower()
    AIProvider._providers[provider_name_lower] = cls
    logger.debug(f"Registered AI provider: {provider_name_lower}")
    return cls


class AIProviderFactory:
    """Factory for creating AI provider instances."""

    @staticmethod
    def create_provider(
        provider_name: str,
        model_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> AIProvider:
        """Create an instance of an AI provider.

        Args:
            provider_name: The name of the provider to create
            model_name: The name of the model to use
            api_key: API key for the provider (if needed)
            api_base: Base URL for the API (if needed)
            **kwargs: Additional provider-specific parameters

        Returns:
            An initialized AI provider instance

        Raises:
            ValueError: If the provider is not registered
        """
        provider_class = AIProvider.get_provider_class(provider_name)
        provider = provider_class(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            **kwargs,
        )

        # Initialize the provider
        provider.initialize()

        return provider
