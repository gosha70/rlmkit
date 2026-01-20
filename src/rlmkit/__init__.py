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
from .llm import MockLLMClient
from .prompts import format_system_prompt, get_default_system_prompt

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
    # Prompts
    "format_system_prompt",
    "get_default_system_prompt",
]
