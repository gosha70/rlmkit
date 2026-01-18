"""RLMKit exception classes."""


class RLMError(Exception):
    """Base exception for RLMKit."""


class BudgetExceeded(RLMError):
    """Raised when step or recursion depth budget is exceeded."""


class ExecutionError(RLMError):
    """Raised when code execution fails."""


class ParseError(RLMError):
    """Raised when parsing LLM response fails."""


class SecurityError(RLMError):
    """Raised when unsafe operation is attempted in sandbox mode."""
