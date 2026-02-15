"""FastAPI dependency injection: adapters, use cases, and shared state."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CONFIG_FILE = Path(".rlmkit_config.json")

from rlmkit.application.dto import RunConfigDTO
from rlmkit.application.use_cases.run_direct import RunDirectUseCase
from rlmkit.application.use_cases.run_rlm import RunRLMUseCase
from rlmkit.infrastructure.llm.litellm_adapter import LiteLLMAdapter
from rlmkit.infrastructure.sandbox.sandbox_factory import create_sandbox
from rlmkit.server.models import (
    ConfigResponse,
    BudgetConfig,
    RunProfile,
    SandboxConfig,
    AppearanceConfig,
    SystemPrompts,
)
from rlmkit.ui.data.providers_catalog import PROVIDERS_BY_KEY


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

    def __init__(self, *, load_from_disk: bool = True) -> None:
        self.start_time = time.time()
        self.config = ConfigResponse()
        self.files: Dict[str, FileRecord] = {}
        self.sessions: Dict[str, SessionRecord] = {}
        self.executions: Dict[str, ExecutionRecord] = {}
        self.user_profiles: List[RunProfile] = []
        self.system_prompts = SystemPrompts()
        if load_from_disk:
            self._load_config()

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """Load persisted config from disk if available."""
        abs_path = _CONFIG_FILE.resolve()
        logger.info(">>> LOAD CONFIG: looking for %s (exists=%s)", abs_path, _CONFIG_FILE.exists())
        if not _CONFIG_FILE.exists():
            logger.info(">>> LOAD CONFIG: file not found, using defaults")
            return
        try:
            raw = json.loads(_CONFIG_FILE.read_text())
            self.config = ConfigResponse.model_validate(raw.get("config", {}))
            self.system_prompts = SystemPrompts.model_validate(raw.get("system_prompts", {}))
            self.user_profiles = [
                RunProfile.model_validate(p) for p in raw.get("user_profiles", [])
            ]
            logger.info(">>> LOAD CONFIG: OK — active_provider=%s active_model=%s provider_configs=%d",
                         self.config.active_provider, self.config.active_model,
                         len(self.config.provider_configs))
            for pc in self.config.provider_configs:
                logger.info(">>>   provider=%s model=%s enabled=%s", pc.provider, pc.model, pc.enabled)
        except Exception as exc:
            logger.warning(">>> LOAD CONFIG FAILED: %s", exc)

    def save_config(self) -> None:
        """Persist current config, profiles, and prompts to disk."""
        abs_path = _CONFIG_FILE.resolve()
        try:
            data = {
                "config": self.config.model_dump(),
                "system_prompts": self.system_prompts.model_dump(),
                "user_profiles": [p.model_dump() for p in self.user_profiles],
            }
            _CONFIG_FILE.write_text(json.dumps(data, indent=2, default=str))
            logger.info(">>> SAVE CONFIG: wrote to %s — active_provider=%s active_model=%s provider_configs=%d",
                         abs_path, self.config.active_provider, self.config.active_model,
                         len(self.config.provider_configs))
            for pc in self.config.provider_configs:
                logger.info(">>>   provider=%s model=%s enabled=%s", pc.provider, pc.model, pc.enabled)
        except Exception as exc:
            logger.warning(">>> SAVE CONFIG FAILED: %s", exc)

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
        provider_key = self.config.active_provider
        model = self.config.active_model

        # Apply LiteLLM provider prefix (e.g. "ollama/" for Ollama models)
        prefixed_model = self._litellm_model_name(provider_key, model)

        # Resolve api_base for local providers (Ollama, LM Studio)
        api_base: Optional[str] = None
        entry = PROVIDERS_BY_KEY.get(provider_key)
        if entry and entry.default_endpoint:
            api_base = entry.default_endpoint

        return LiteLLMAdapter(
            model=prefixed_model,
            api_base=api_base,
        )

    @staticmethod
    def _litellm_model_name(provider_key: str, model: str) -> str:
        """Prepend the LiteLLM provider prefix if needed."""
        if "/" in model:
            return model
        _prefixes = {
            "anthropic": "anthropic/",
            "ollama": "ollama/",
            "lmstudio": "openai/",
        }
        prefix = _prefixes.get(provider_key, "")
        return f"{prefix}{model}"

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
    """Reset state (used in tests). Creates a fresh AppState without loading from disk."""
    global _state
    _state = AppState(load_from_disk=False)
