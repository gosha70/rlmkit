# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

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
from .trace import (
    TraceStep,
    ExecutionTrace,
    ActionType,
    load_trace_from_jsonl,
    load_trace_from_json,
)
from .model_config import (
    ModelConfig,
    ModelMetrics,
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
    # Tracing
    "TraceStep",
    "ExecutionTrace",
    "ActionType",
    "load_trace_from_jsonl",
    "load_trace_from_json",
    # Multi-model
    "ModelConfig",
    "ModelMetrics",
]
