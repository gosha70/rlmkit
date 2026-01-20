"""
RLMKit: Recursive Language Model Toolkit

A toolkit for building LLM-based systems that can handle arbitrarily large contexts
through code generation and recursive exploration.

Based on the RLM paper (arXiv:2512.24601).
"""

from .core import (
    RLM,
    RLMResult,
    LLMClient,
    RLMError,
    BudgetExceeded,
    ExecutionError,
    SecurityError,
    BudgetTracker,
    BudgetLimits,
    TokenUsage,
    CostTracker,
    estimate_tokens,
)
from .config import RLMConfig, SecurityConfig, ExecutionConfig, MonitoringConfig
from .llm import MockLLMClient, BaseLLMProvider, LLMResponse
from .llm.config import LLMConfig, LLMProviderConfig, ModelPricing
from .prompts import format_system_prompt, get_default_system_prompt

# Optional external LLM providers
try:
    from .llm import OpenAIClient
except ImportError:
    OpenAIClient = None

try:
    from .llm import ClaudeClient
except ImportError:
    ClaudeClient = None

try:
    from .llm import OllamaClient
except ImportError:
    OllamaClient = None

try:
    from .llm import LMStudioClient
except ImportError:
    LMStudioClient = None

try:
    from .llm import vLLMClient
except ImportError:
    vLLMClient = None

__version__ = "0.1.0"

__all__ = [
    # Core RLM
    "RLM",
    "RLMResult",
    "LLMClient",
    # Configuration
    "RLMConfig",
    "SecurityConfig",
    "ExecutionConfig",
    "MonitoringConfig",
    "LLMConfig",
    "LLMProviderConfig",
    "ModelPricing",
    # Errors
    "RLMError",
    "BudgetExceeded",
    "ExecutionError",
    "SecurityError",
    # Budget Tracking
    "BudgetTracker",
    "BudgetLimits",
    "TokenUsage",
    "CostTracker",
    "estimate_tokens",
    # LLM Clients
    "MockLLMClient",
    "BaseLLMProvider",
    "LLMResponse",
    # Prompts
    "format_system_prompt",
    "get_default_system_prompt",
]

# Add optional providers to __all__ if available
if OpenAIClient is not None:
    __all__.append("OpenAIClient")
if ClaudeClient is not None:
    __all__.append("ClaudeClient")
if OllamaClient is not None:
    __all__.append("OllamaClient")
if LMStudioClient is not None:
    __all__.append("LMStudioClient")
if vLLMClient is not None:
    __all__.append("vLLMClient")
