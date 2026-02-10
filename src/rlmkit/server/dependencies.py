"""FastAPI dependency injection: adapters, use cases, and shared state."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rlmkit.application.dto import RunConfigDTO
from rlmkit.application.use_cases.run_direct import RunDirectUseCase
from rlmkit.application.use_cases.run_rlm import RunRLMUseCase
from rlmkit.infrastructure.llm.litellm_adapter import LiteLLMAdapter
from rlmkit.infrastructure.sandbox.sandbox_factory import create_sandbox
from rlmkit.server.models import (
    ConfigResponse,
    BudgetConfig,
    SandboxConfig,
    AppearanceConfig,
)


# ---------------------------------------------------------------------------
# In-memory stores (replaced by real persistence in production)
# ---------------------------------------------------------------------------


@dataclass
class FileRecord:
    id: str
    name: str
    size_bytes: int
    content_type: str
    text_content: str
    token_count: int
    created_at: datetime


@dataclass
class SessionRecord:
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    messages: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExecutionRecord:
    execution_id: str
    session_id: str
    query: str
    mode: str
    status: str = "running"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)


class AppState:
    """Shared application state holding in-memory stores and configuration."""

    def __init__(self) -> None:
        self.start_time = time.time()
        self.config = ConfigResponse()
        self.files: Dict[str, FileRecord] = {}
        self.sessions: Dict[str, SessionRecord] = {}
        self.executions: Dict[str, ExecutionRecord] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionRecord:
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        now = datetime.now(timezone.utc)
        sid = session_id or str(uuid.uuid4())
        session = SessionRecord(
            id=sid,
            name=f"Session {now.strftime('%Y-%m-%d %H:%M')}",
            created_at=now,
            updated_at=now,
        )
        self.sessions[sid] = session
        return session

    def create_llm_adapter(self) -> LiteLLMAdapter:
        return LiteLLMAdapter(
            model=self.config.active_model,
        )

    def create_sandbox(self):  # type: ignore[no-untyped-def]
        return create_sandbox(sandbox_type=self.config.sandbox.type)

    def create_run_config(self, mode: str = "auto") -> RunConfigDTO:
        b = self.config.budget
        return RunConfigDTO(
            mode=mode,
            max_steps=b.max_steps,
            max_tokens=b.max_tokens,
            max_cost=b.max_cost_usd,
            max_time_seconds=float(b.max_time_seconds),
            max_recursion_depth=b.max_recursion_depth,
        )


# Singleton state -- created once at startup, injected into routes
_state: Optional[AppState] = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state


def reset_state() -> None:
    """Reset state (used in tests)."""
    global _state
    _state = None
