"""Public API surface for Clean Architecture.

This package provides the new structured public API that uses the
Clean Architecture layers internally. The existing ``rlmkit.api`` module
is preserved for backward compatibility and continues to work.

Usage:
    from rlmkit.public import RLMKitClient
    client = RLMKitClient(provider="openai", model="gpt-4o")
    result = client.interact("document text", "question")
"""

from .client import RLMKitClient
from .types import PublicRunResult, PublicInteractResult
from .errors import (
    RLMKitError,
    ProviderError,
    BudgetError,
    SandboxError,
    ConfigError,
)

__all__ = [
    "RLMKitClient",
    "PublicRunResult",
    "PublicInteractResult",
    "RLMKitError",
    "ProviderError",
    "BudgetError",
    "SandboxError",
    "ConfigError",
]
