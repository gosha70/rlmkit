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
