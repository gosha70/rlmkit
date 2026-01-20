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
