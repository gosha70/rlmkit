"""
RLMKit: Recursive Language Model toolkit for managing large context through code generation.

This package provides tools for LLMs to explore and analyze large content by writing
Python code, without exceeding context window limitations.
"""

__version__ = "0.1.0"

from rlmkit.core.errors import RLMError, BudgetExceeded, ExecutionError, ParseError, SecurityError

__all__ = [
    "__version__",
    "RLMError",
    "BudgetExceeded",
    "ExecutionError",
    "ParseError",
    "SecurityError",
]
