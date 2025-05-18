#!/usr/bin/env python3
"""
Anthropic Provider implementation for form-filler.

Provides integration with Anthropic's Claude API for LLM inference.
"""

import base64
import logging
from typing import Any

import requests

from form_filler.ai_providers.base_provider import AIProvider, register_provider

# Setup logging
logger = logging.getLogger(__name__)


@register_provider
class AnthropicProvider(AIProvider):
    """AI Provider implementation for Anthropic (Claude)."""

    provider_name = "anthropic"

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Anthropic provider.

        Args:
            model_name: The name of the Claude model to use (e.g., "claude-3-opus-20240229")
            api_key: Anthropic API key (required)
            api_base: Base URL for Anthropic API (default: https://api.anthropic.com/v1)
            **kwargs: Additional provider-specific parameters
        """
        super().__init__(model_name, api_key, api_base, **kwargs)

        # Set default API base if not provided
        if not self.api_base:
            self.api_base = "https://api.anthropic.com/v1"

        # Determine if this is a vision model
        self._is_vision_model = "claude-3" in model_name.lower()

        # Set default timeout
        self.timeout = kwargs.get("timeout", 30)

        # Set Anthropic API version
        self.anthropic_version = kwargs.get("anthropic_version", "2023-06-01")

    def initialize(self) -> None:
        """Initialize the Anthropic client.

        Validates API key and connection.
        """
        if not self.api_key:
            raise ValueError("Anthropic API key is required")

        try:
            # Create a minimal request to check API key validity
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
                "Content-Type": "application/json",
            }

            # We'll make a minimal messages request with a small max_tokens value
            # just to validate the API key and connection
            payload = {
                "model": self.model_name,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "Hello"}],
            }

            # Create messages URL from base URL
            messages_url = f"{self.api_base}/messages"

            # Send request with a short timeout just to test connectivity
            response = requests.post(
                messages_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            # Check response
            if response.status_code != 200 and response.status_code != 201:
                message = f"Anthropic API returned status code {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        message = f"{message} - {error_data['error']['message']}"
                except Exception as err:
                    logger.debug(f"Could not parse error details: {err}")
                raise ValueError(message)

            logger.debug(f"Successfully connected to Anthropic API using model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {e}")
            raise

    def text_completion(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a text completion using Anthropic Claude.

        Args:
            prompt: The prompt to generate a completion for
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            The generated text completion
        """
        # For Anthropic, we'll use the messages API
        messages = [{"role": "user", "content": prompt}]

        # Use the chat completion method (which handles system prompts)
        return self.chat_completion(messages, max_tokens, temperature, system_prompt=system_prompt)

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a chat completion using Anthropic Claude.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system instructions (specific to Anthropic API)

        Returns:
            The generated chat completion text
        """
        # Validate API key
        if not self.api_key:
            raise ValueError("Anthropic API key is required")

        try:
            # Prepare headers
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
                "Content-Type": "application/json",
            }

            # Prepare payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            # Add system prompt if provided
            if system_prompt:
                payload["system"] = system_prompt

            # Add any additional parameters from kwargs
            for key, value in self.kwargs.items():
                if key not in payload and key not in ["timeout", "anthropic_version"]:
                    payload[key] = value

            # Create messages URL from base URL
            messages_url = f"{self.api_base}/messages"

            # Send request
            response = requests.post(
                messages_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            # Check response
            if response.status_code != 200 and response.status_code != 201:
                message = f"Anthropic API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        message = f"{message} - {error_data['error']['message']}"
                except Exception as err:
                    logger.debug(f"Could not parse error details: {err}")
                raise ValueError(message)

            # Parse response
            response_data = response.json()

            # Extract completion text
            completion_text = response_data["content"][0]["text"]

            return completion_text

        except Exception as e:
            logger.error(f"Anthropic chat completion failed: {e}")
            raise

    def vision_completion(
        self,
        prompt: str,
        image_data: str | bytes,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion based on an image using Anthropic Claude.

        Args:
            prompt: The text prompt to accompany the image
            image_data: The image data (base64 string or bytes)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            The generated text based on the image
        """
        if not self.supports_vision:
            raise ValueError(
                f"Anthropic model '{self.model_name}' does not support vision. "
                "Please use a Claude 3 model, which supports vision."
            )

        # Validate API key
        if not self.api_key:
            raise ValueError("Anthropic API key is required")

        try:
            # Ensure image_data is a base64 string
            if isinstance(image_data, bytes):
                base64_image = base64.b64encode(image_data).decode("utf-8")
            else:
                # Assume it's already a base64 string
                base64_image = image_data

            # Prepare headers
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
                "Content-Type": "application/json",
            }

            # Prepare message with image in Anthropic's format
            content = [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image,
                    },
                },
            ]

            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": content,
                },
            ]

            # Prepare payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            # Add any additional parameters from kwargs
            for key, value in self.kwargs.items():
                if key not in payload and key not in ["timeout", "anthropic_version"]:
                    payload[key] = value

            # Create messages URL from base URL
            messages_url = f"{self.api_base}/messages"

            # Send request
            response = requests.post(
                messages_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            # Check response
            if response.status_code != 200 and response.status_code != 201:
                message = f"Anthropic API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        message = f"{message} - {error_data['error']['message']}"
                except Exception as err:
                    logger.debug(f"Could not parse error details: {err}")
                raise ValueError(message)

            # Parse response
            response_data = response.json()

            # Extract completion text
            completion_text = response_data["content"][0]["text"]

            return completion_text

        except Exception as e:
            logger.error(f"Anthropic vision completion failed: {e}")
            raise

    @property
    def supports_vision(self) -> bool:
        """Check if this Anthropic model supports vision capabilities.

        Returns:
            True if model is a Claude 3 model, False otherwise
        """
        return self._is_vision_model

    @property
    def supports_streaming(self) -> bool:
        """Check if this provider supports streaming responses.

        Returns:
            True as Anthropic supports streaming
        """
        return True

    @property
    def requires_api_key(self) -> bool:
        """Check if this provider requires an API key.

        Returns:
            True as Anthropic requires API keys
        """
        return True
