"""Domain layer: pure business entities with zero external dependencies.

This is the innermost layer of the Clean Architecture. Nothing here
imports from application, infrastructure, or any third-party package.
"""

from .entities import (
    Query,
    Response,
    TraceStep,
    ExecutionTrace,
    BudgetConfig,
    BudgetState,
)
from .value_objects import (
    TokenCount,
    Cost,
    ModelId,
    ProviderId,
    RecursionDepth,
)
from .events import (
    StepCompleted,
    BudgetExceeded,
    ExecutionStarted,
    ExecutionCompleted,
)
from .exceptions import (
    DomainError,
    BudgetExceededError,
    ExecutionFailedError,
    SecurityViolationError,
    ParseFailedError,
    ConfigurationError,
)

__all__ = [
    # Entities
    "Query",
    "Response",
    "TraceStep",
    "ExecutionTrace",
    "BudgetConfig",
    "BudgetState",
    # Value objects
    "TokenCount",
    "Cost",
    "ModelId",
    "ProviderId",
    "RecursionDepth",
    # Events
    "StepCompleted",
    "BudgetExceeded",
    "ExecutionStarted",
    "ExecutionCompleted",
    # Exceptions
    "DomainError",
    "BudgetExceededError",
    "ExecutionFailedError",
    "SecurityViolationError",
    "ParseFailedError",
    "ConfigurationError",
]
