"""Domain layer: pure business entities with zero external dependencies.

This is the innermost layer of the Clean Architecture. Nothing here
imports from application, infrastructure, or any third-party package.
"""

from .entities import (
    BudgetConfig,
    BudgetState,
    ExecutionTrace,
    Query,
    Response,
    TraceStep,
)
from .events import (
    BudgetExceeded,
    ExecutionCompleted,
    ExecutionStarted,
    StepCompleted,
)
from .exceptions import (
    BudgetExceededError,
    ConfigurationError,
    DomainError,
    ExecutionFailedError,
    ParseFailedError,
    SecurityViolationError,
)
from .value_objects import (
    Cost,
    ModelId,
    ProviderId,
    RecursionDepth,
    TokenCount,
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
