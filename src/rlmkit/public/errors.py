"""Public exception types for the RLMKit API.

These wrap domain exceptions into a stable public API surface.
"""

from rlmkit.domain.exceptions import (
    DomainError,
    BudgetExceededError,
    ExecutionFailedError,
    SecurityViolationError,
    ConfigurationError,
)


class RLMKitError(Exception):
    """Base exception for all public RLMKit errors."""


class ProviderError(RLMKitError):
    """Raised when an LLM provider operation fails."""


class BudgetError(RLMKitError):
    """Raised when a budget limit is exceeded."""


class SandboxError(RLMKitError):
    """Raised when sandbox execution fails."""


class ConfigError(RLMKitError):
    """Raised when configuration is invalid or incomplete."""


def wrap_domain_error(exc: Exception) -> RLMKitError:
    """Convert a domain exception to the corresponding public exception.

    Args:
        exc: A domain-layer exception.

    Returns:
        The appropriate public exception wrapping the original.
    """
    if isinstance(exc, BudgetExceededError):
        return BudgetError(str(exc))
    if isinstance(exc, ExecutionFailedError):
        return SandboxError(str(exc))
    if isinstance(exc, SecurityViolationError):
        return SandboxError(str(exc))
    if isinstance(exc, ConfigurationError):
        return ConfigError(str(exc))
    if isinstance(exc, DomainError):
        return RLMKitError(str(exc))
    return RLMKitError(str(exc))
