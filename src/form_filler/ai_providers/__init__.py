"""
AI Providers module for form-filler.

This module provides a modular architecture for interacting with various AI providers.
"""

# Import all provider implementations to ensure they're registered
from form_filler.ai_providers.anthropic_provider import AnthropicProvider
from form_filler.ai_providers.base_provider import AIProvider, AIProviderFactory, register_provider
from form_filler.ai_providers.deepseek_provider import DeepseekProvider
from form_filler.ai_providers.gemini_provider import GeminiProvider
from form_filler.ai_providers.ollama_provider import OllamaProvider
from form_filler.ai_providers.openai_provider import OpenAIProvider

__all__ = [
    "AIProvider",
    "AIProviderFactory",
    "AnthropicProvider",
    "DeepseekProvider",
    "GeminiProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "register_provider",
]
