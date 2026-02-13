"""Event emitter port for real-time streaming of execution events.

Use cases can emit token, step, and metrics events through this port
to enable WebSocket streaming to clients.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class ExecutionEventEmitter(Protocol):
    """Protocol for emitting execution events during use-case runs."""

    async def on_token(self, token: str) -> None:
        """Emit a single token from LLM streaming output."""
        ...

    async def on_step(self, step_data: Dict[str, Any]) -> None:
        """Emit step completion data (trace entry)."""
        ...

    async def on_metrics(self, metrics: Dict[str, Any]) -> None:
        """Emit running metric totals."""
        ...
