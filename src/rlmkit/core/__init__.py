"""Core RLM functionality."""

from .rlm import RLM, RLMResult, LLMClient
from .errors import RLMError, BudgetExceeded, ExecutionError, SecurityError
from .budget import (
    BudgetTracker,
    BudgetLimits,
    TokenUsage,
    CostTracker,
    estimate_tokens,
)
from .comparison import (
    ComparisonResult,
    ExecutionMetrics,
)
from .parsing import (
    ParsedResponse,
    parse_response,
    extract_python_code,
    extract_final_answer,
    extract_final_var,
    format_code_for_display,
    format_result_for_llm,
)

__all__ = [
    # RLM controller
    "RLM",
    "RLMResult",
    "LLMClient",
    # Errors
    "RLMError",
    "BudgetExceeded",
    "ExecutionError",
    "SecurityError",
    # Budget tracking
    "BudgetTracker",
    "BudgetLimits",
    "TokenUsage",
    "CostTracker",
    "estimate_tokens",
    # Comparison
    "ComparisonResult",
    "ExecutionMetrics",
    # Parsing
    "ParsedResponse",
    "parse_response",
    "extract_python_code",
    "extract_final_answer",
    "extract_final_var",
    "format_code_for_display",
    "format_result_for_llm",
]
