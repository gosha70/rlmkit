"""FastAPI dependency injection: adapters, use cases, and shared state."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rlmkit.application.dto import RunConfigDTO
from rlmkit.infrastructure.llm.litellm_adapter import LiteLLMAdapter
from rlmkit.infrastructure.sandbox.sandbox_factory import create_sandbox
from rlmkit.server.models import (
    ChatProviderConfig,
    ConfigResponse,
    RAGConfig,
    RunProfile,
    RuntimeSettings,
    SystemPrompts,
)
from rlmkit.ui.data.providers_catalog import PROVIDERS_BY_KEY

logger = logging.getLogger(__name__)

_CONFIG_FILE = Path(".rlmkit_config.json")
_SESSIONS_FILE = Path(".rlmkit_sessions.json")
_MAX_PERSISTED_SESSIONS = 50

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
    messages: list[dict[str, Any]] = field(default_factory=list)
    conversations: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict
    )  # keyed by chat_provider_id


@dataclass
class ExecutionRecord:
    execution_id: str
    session_id: str
    query: str
    mode: str
    status: str = "running"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    chat_provider_id: str | None = None
    chat_provider_name: str | None = None


class AppState:
    """Shared application state holding in-memory stores and configuration."""

    def __init__(self, *, load_from_disk: bool = True) -> None:
        self.start_time = time.time()
        self.config = ConfigResponse()
        self.files: dict[str, FileRecord] = {}
        self.sessions: dict[str, SessionRecord] = {}
        self.executions: dict[str, ExecutionRecord] = {}
        self.user_profiles: list[RunProfile] = []
        self.system_prompts = SystemPrompts()
        if load_from_disk:
            self._load_config()
            self._load_sessions()
            self._migrate_chat_providers()

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _load_config(self) -> None:
        """Load persisted config from disk if available."""
        if not _CONFIG_FILE.exists():
            return
        try:
            raw = json.loads(_CONFIG_FILE.read_text())
            self.config = ConfigResponse.model_validate(raw.get("config", {}))
            self.system_prompts = SystemPrompts.model_validate(raw.get("system_prompts", {}))
            self.user_profiles = [
                RunProfile.model_validate(p) for p in raw.get("user_profiles", [])
            ]
            self._reconcile_active_provider()
            logger.info(
                "Loaded config: provider=%s model=%s (%d provider configs)",
                self.config.active_provider,
                self.config.active_model,
                len(self.config.provider_configs),
            )
        except Exception as exc:
            logger.warning("Failed to load config: %s", exc)

    def _reconcile_active_provider(self) -> None:
        """Ensure active_provider has an enabled entry in provider_configs.

        Fixes corrupted state from the old dual-save-path bug where
        active_provider/active_model could drift from provider_configs.
        """
        from rlmkit.server.models import ProviderConfig

        active = self.config.active_provider
        configs = self.config.provider_configs

        # Find the active provider's entry
        active_entry: ProviderConfig | None = None
        for pc in configs:
            if pc.provider == active:
                active_entry = pc
                break

        changed = False

        # If the active provider has no entry, create one
        if active_entry is None:
            catalog_entry = PROVIDERS_BY_KEY.get(active)
            default_model = self.config.active_model
            if not default_model and catalog_entry and catalog_entry.models:
                default_model = catalog_entry.models[0].name
            active_entry = ProviderConfig(
                provider=active,
                model=default_model or "",
                enabled=True,
            )
            configs.append(active_entry)
            changed = True

        # Ensure the active provider's entry is enabled
        if not active_entry.enabled:
            active_entry.enabled = True
            changed = True

        # Ensure active_model matches the entry's model
        if active_entry.model and active_entry.model != self.config.active_model:
            self.config.active_model = active_entry.model
            changed = True

        # Disable all other providers (only one active at a time)
        for pc in configs:
            if pc.provider != active and pc.enabled:
                pc.enabled = False
                changed = True

        if changed:
            logger.info(
                "Reconciled config: set %s as active provider (model=%s)",
                active,
                self.config.active_model,
            )
            self.save_config()

    def _migrate_chat_providers(self) -> None:
        """Auto-create Chat Providers from existing provider_configs on first load."""
        if self.config.chat_providers:
            return  # Already have Chat Providers
        now = datetime.now(timezone.utc)
        for pc in self.config.provider_configs:
            if not pc.enabled:
                continue
            mode_label = "DIRECT"
            cp = ChatProviderConfig(
                id=str(uuid.uuid4()),
                name=f"{mode_label}-{pc.provider.upper()}",
                llm_provider=pc.provider,
                llm_model=pc.model,
                execution_mode="direct",
                runtime_settings=pc.runtime_settings.model_copy(),
                created_at=now,
                updated_at=now,
            )
            self.config.chat_providers.append(cp)
        if self.config.chat_providers:
            logger.info(
                "Migrated %d Chat Provider(s) from existing provider configs",
                len(self.config.chat_providers),
            )
            self.save_config()

    # ------------------------------------------------------------------
    # Chat Provider helpers
    # ------------------------------------------------------------------

    def get_chat_provider(self, chat_provider_id: str) -> ChatProviderConfig | None:
        """Look up a Chat Provider by ID."""
        for cp in self.config.chat_providers:
            if cp.id == chat_provider_id:
                return cp
        return None

    def get_conversation(self, session_id: str, chat_provider_id: str) -> list[dict[str, Any]]:
        """Get conversation history for a specific Chat Provider in a session."""
        session = self.sessions.get(session_id)
        if not session:
            return []
        return session.conversations.get(chat_provider_id, [])

    def add_message(
        self,
        session_id: str,
        message: dict[str, Any],
        chat_provider_id: str | None = None,
    ) -> None:
        """Add a message to a session, tracking it per Chat Provider."""
        session = self.sessions.get(session_id)
        if not session:
            return
        # Always add to legacy flat messages for backward compat
        session.messages.append(message)
        # Also add to per-Chat-Provider conversations
        if chat_provider_id:
            if chat_provider_id not in session.conversations:
                session.conversations[chat_provider_id] = []
            session.conversations[chat_provider_id].append(message)

    def create_llm_adapter_for_chat_provider(self, chat_provider_id: str) -> LiteLLMAdapter:
        """Create an LLM adapter configured for a specific Chat Provider."""
        cp = self.get_chat_provider(chat_provider_id)
        if not cp:
            raise ValueError(f"Chat Provider {chat_provider_id} not found")

        provider_key = cp.llm_provider
        model = cp.llm_model

        # Validate model against catalog
        entry = PROVIDERS_BY_KEY.get(provider_key)
        if entry and entry.models:
            catalog_names = {m.name for m in entry.models}
            if model not in catalog_names:
                logger.warning(
                    "Model %s not in provider %s catalog; falling back to %s",
                    model,
                    provider_key,
                    entry.models[0].name,
                )
                model = entry.models[0].name

        prefixed_model = self._litellm_model_name(provider_key, model)

        api_base: str | None = None
        if entry and entry.default_endpoint:
            api_base = entry.default_endpoint

        runtime = cp.runtime_settings

        return LiteLLMAdapter(
            model=prefixed_model,
            api_base=api_base,
            temperature=runtime.temperature,
            max_tokens=runtime.max_output_tokens,
            timeout=float(runtime.timeout_seconds),
            num_retries=2,
        )

    def save_config(self) -> None:
        """Persist current config, profiles, and prompts to disk."""
        try:
            data = {
                "config": self.config.model_dump(),
                "system_prompts": self.system_prompts.model_dump(),
                "user_profiles": [p.model_dump() for p in self.user_profiles],
            }
            _CONFIG_FILE.write_text(json.dumps(data, indent=2, default=str))
            logger.info(
                "Saved config: provider=%s model=%s (%d provider configs)",
                self.config.active_provider,
                self.config.active_model,
                len(self.config.provider_configs),
            )
        except Exception as exc:
            logger.warning("Failed to save config: %s", exc)

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _load_sessions(self) -> None:
        """Load persisted sessions from disk if available."""
        if not _SESSIONS_FILE.exists():
            return
        try:
            raw = json.loads(_SESSIONS_FILE.read_text())
            for s in raw:
                rec = SessionRecord(
                    id=s["id"],
                    name=s["name"],
                    created_at=datetime.fromisoformat(s["created_at"]),
                    updated_at=datetime.fromisoformat(s["updated_at"]),
                    messages=s.get("messages", []),
                    conversations=s.get("conversations", {}),
                )
                self.sessions[rec.id] = rec
            logger.info("Loaded %d sessions from disk", len(self.sessions))
        except Exception as exc:
            logger.warning("Failed to load sessions: %s", exc)

    def save_sessions(self) -> None:
        """Persist sessions to disk (most recent N only)."""
        try:
            # Sort by updated_at descending and cap
            sorted_sessions = sorted(
                self.sessions.values(),
                key=lambda s: s.updated_at,
                reverse=True,
            )[:_MAX_PERSISTED_SESSIONS]
            data = [
                {
                    "id": s.id,
                    "name": s.name,
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                    "messages": s.messages,
                    "conversations": s.conversations,
                }
                for s in sorted_sessions
            ]
            _SESSIONS_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception as exc:
            logger.warning("Failed to save sessions: %s", exc)

    def get_or_create_session(self, session_id: str | None = None) -> SessionRecord:
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

        # Validate provider/model consistency: if active_model doesn't belong
        # to active_provider's catalog, fall back to provider's default model
        entry = PROVIDERS_BY_KEY.get(provider_key)
        if entry and entry.models:
            catalog_names = {m.name for m in entry.models}
            if model not in catalog_names:
                logger.warning(
                    "Model %s not in provider %s catalog; falling back to %s",
                    model,
                    provider_key,
                    entry.models[0].name,
                )
                model = entry.models[0].name
                self.config.active_model = model

        # Apply LiteLLM provider prefix (e.g. "ollama/" for Ollama models)
        prefixed_model = self._litellm_model_name(provider_key, model)

        # Resolve api_base for local providers (Ollama, LM Studio)
        api_base: str | None = None
        if entry and entry.default_endpoint:
            api_base = entry.default_endpoint

        # Read runtime settings from active provider's config, falling back
        # to default_runtime_settings
        runtime = self.config.default_runtime_settings
        for pc in self.config.provider_configs:
            if pc.provider == provider_key and pc.enabled:
                runtime = pc.runtime_settings
                break

        return LiteLLMAdapter(
            model=prefixed_model,
            api_base=api_base,
            temperature=runtime.temperature,
            max_tokens=runtime.max_output_tokens,
            timeout=float(runtime.timeout_seconds),
            num_retries=2,
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
_state: AppState | None = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state


def reset_state() -> None:
    """Reset state (used in tests). Creates a fresh AppState without loading from disk.

    Also disables disk persistence so tests never overwrite the user's config.
    """
    global _state
    _state = AppState(load_from_disk=False)
    _state.save_config = lambda: None  # type: ignore[assignment]
    _state.save_sessions = lambda: None  # type: ignore[assignment]
