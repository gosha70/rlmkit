"""Domain exceptions: business-rule violations with no external dependencies.

These exceptions represent domain-level error conditions. They form
a hierarchy rooted at ``DomainError`` and are free of any framework
or infrastructure concerns.
"""


class DomainError(Exception):
    """Base exception for all domain-level errors in RLMKit."""


class BudgetExceededError(DomainError):
    """Raised when a budget limit (steps, tokens, cost, time, depth) is exceeded."""


class ExecutionFailedError(DomainError):
    """Raised when code execution in the sandbox fails."""


class SecurityViolationError(DomainError):
    """Raised when an unsafe operation is attempted (e.g. blocked import)."""


class ParseFailedError(DomainError):
    """Raised when an LLM response cannot be parsed into a valid action."""


class ConfigurationError(DomainError):
    """Raised when configuration is invalid or incomplete."""
