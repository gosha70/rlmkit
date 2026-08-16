"""Public API surface for Clean Architecture.

This package provides the new structured public API that uses the
Clean Architecture layers internally. The existing ``rlmstudio.api`` module
is preserved for backward compatibility and continues to work.

Usage:
    from rlmstudio.public import RLMKitClient
    client = RLMKitClient(provider="openai", model="gpt-4o")
    result = client.interact("document text", "question")
"""

from .client import RLMKitClient
from .errors import (
    BudgetError,
    ConfigError,
    ProviderError,
    RLMKitError,
    SandboxError,
)
from .types import PublicRunResult

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


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "PublicInteractResult":
        import warnings

        warnings.warn(
            "PublicInteractResult is deprecated. Use rlmstudio.InteractResult instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from rlmstudio.api import InteractResult

        return InteractResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
