# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""LLM client interfaces and implementations."""

from .mock_client import MockLLMClient
from .base import BaseLLMProvider, LLMResponse

# Optional external providers (lazy import to avoid requiring dependencies)
__all__ = [
    "MockLLMClient",
    "BaseLLMProvider",
    "LLMResponse",
]

# Try to import OpenAI client
try:
    from .openai_client import OpenAIClient
    __all__.append("OpenAIClient")
except ImportError:
    OpenAIClient = None

# Try to import Claude client
try:
    from .anthropic_client import ClaudeClient
    __all__.append("ClaudeClient")
except ImportError:
    ClaudeClient = None

# Try to import Ollama client
try:
    from .ollama_client import OllamaClient
    __all__.append("OllamaClient")
except ImportError:
    OllamaClient = None

# Try to import LM Studio client
try:
    from .lmstudio_client import LMStudioClient
    __all__.append("LMStudioClient")
except ImportError:
    LMStudioClient = None

# Try to import vLLM client
try:
    from .vllm_client import vLLMClient
    __all__.append("vLLMClient")
except ImportError:
    vLLMClient = None


def get_llm_client(provider: str, model: str = None, api_key: str = None, **kwargs) -> BaseLLMProvider:
    """
    Factory function to create LLM client based on provider name.
    
    Args:
        provider: Provider name ('openai', 'anthropic', 'ollama', 'lmstudio', 'vllm', 'mock')
        model: Model name (provider-specific)
        api_key: API key for the provider (optional, will use environment variable if not provided)
        **kwargs: Additional arguments for the client
        
    Returns:
        LLM client instance
        
    Raises:
        ValueError: If provider is not supported or not installed
    """
    provider = provider.lower()
    
    if provider == "mock":
        return MockLLMClient(**kwargs)
    
    elif provider == "openai":
        if OpenAIClient is None:
            raise ValueError("OpenAI client not available. Install with: pip install openai")
        if api_key:
            kwargs['api_key'] = api_key
        return OpenAIClient(model=model or "gpt-4", **kwargs)
    
    elif provider == "anthropic":
        if ClaudeClient is None:
            raise ValueError("Anthropic client not available. Install with: pip install anthropic")
        if api_key:
            kwargs['api_key'] = api_key
        return ClaudeClient(model=model or "claude-3-sonnet-20240229", **kwargs)
    
    elif provider == "ollama":
        if OllamaClient is None:
            raise ValueError("Ollama client not available. Install with: pip install ollama")
        return OllamaClient(model=model or "llama2", **kwargs)
    
    elif provider == "lmstudio" or provider == "lm studio":
        if LMStudioClient is None:
            raise ValueError("LM Studio client not available")
        return LMStudioClient(model=model, **kwargs)
    
    elif provider == "vllm":
        if vLLMClient is None:
            raise ValueError("vLLM client not available")
        return vLLMClient(model=model, **kwargs)
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openai, anthropic, ollama, lmstudio, vllm, mock")


__all__.append("get_llm_client")
