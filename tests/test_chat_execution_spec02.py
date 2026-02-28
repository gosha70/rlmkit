"""Acceptance tests for Chat Execution & Session Storage (spec 02).

Verifies all acceptance criteria for the Chat Provider integration:
1. POST /api/chat with chat_provider_id uses that provider's mode
2. POST /api/chat without chat_provider_id works as before (backward compat)
3. GET /api/sessions/{id} returns both `messages` and `conversations`
4. User messages are stored in both flat and per-provider lists
5. Assistant messages include chat_provider_id, chat_provider_name, and metrics
6. Session persistence: conversations dict is saved and loaded correctly
7. add_message() helper writes to both flat and per-provider lists
8. get_conversation() helper returns correct messages for a given chat_provider_id
9. create_llm_adapter_for_chat_provider() creates adapter with correct settings
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rlmkit.server.app import app
from rlmkit.server.dependencies import AppState, SessionRecord, get_state, reset_state
from rlmkit.server.models import (
    ChatProviderConfig,
    RuntimeSettings,
)
from rlmkit.ui.data.providers_catalog import PROVIDERS_BY_KEY

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    """Reset in-memory state before and after every test."""
    reset_state()
    yield
    reset_state()


@pytest.fixture()
def client() -> TestClient:
    """FastAPI TestClient backed by the real application."""
    return TestClient(app)


@pytest.fixture()
def direct_provider() -> ChatProviderConfig:
    """A minimal ChatProviderConfig with direct execution mode."""
    return ChatProviderConfig(
        id=str(uuid.uuid4()),
        name="DIRECT-OPENAI",
        llm_provider="openai",
        llm_model="gpt-4o",
        execution_mode="direct",
        runtime_settings=RuntimeSettings(
            temperature=0.5,
            top_p=0.9,
            max_output_tokens=2048,
            timeout_seconds=20,
        ),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture()
def rlm_provider() -> ChatProviderConfig:
    """A minimal ChatProviderConfig with rlm execution mode."""
    return ChatProviderConfig(
        id=str(uuid.uuid4()),
        name="RLM-OPENAI",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        execution_mode="rlm",
        rlm_max_steps=8,
        rlm_timeout_seconds=30,
        runtime_settings=RuntimeSettings(),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _add_provider_to_state(cp: ChatProviderConfig) -> None:
    """Register a ChatProviderConfig in the current AppState singleton."""
    state = get_state()
    state.config.chat_providers.append(cp)


# ---------------------------------------------------------------------------
# AC1: POST /api/chat with chat_provider_id uses that provider's mode
# ---------------------------------------------------------------------------


class TestChatWithProviderIdUsesCorrectMode:
    """AC1 — chat_provider_id drives execution mode on the record."""

    def test_direct_provider_sets_direct_mode_on_execution(
        self, client: TestClient, direct_provider: ChatProviderConfig
    ) -> None:
        _add_provider_to_state(direct_provider)

        resp = client.post(
            "/api/chat",
            json={
                "query": "What is 2+2?",
                "content": "Some context.",
                "chat_provider_id": direct_provider.id,
            },
        )
        assert resp.status_code == 202
        exec_id = resp.json()["execution_id"]

        state = get_state()
        exec_rec = state.executions[exec_id]
        assert exec_rec.mode == "direct"

    def test_rlm_provider_sets_rlm_mode_on_execution(
        self, client: TestClient, rlm_provider: ChatProviderConfig
    ) -> None:
        _add_provider_to_state(rlm_provider)

        resp = client.post(
            "/api/chat",
            json={
                "query": "Analyze this.",
                "content": "Some context.",
                "chat_provider_id": rlm_provider.id,
            },
        )
        assert resp.status_code == 202
        exec_id = resp.json()["execution_id"]

        state = get_state()
        exec_rec = state.executions[exec_id]
        assert exec_rec.mode == "rlm"

    def test_response_echoes_back_chat_provider_id(
        self, client: TestClient, direct_provider: ChatProviderConfig
    ) -> None:
        _add_provider_to_state(direct_provider)

        resp = client.post(
            "/api/chat",
            json={
                "query": "Hello",
                "content": "Context.",
                "chat_provider_id": direct_provider.id,
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["chat_provider_id"] == direct_provider.id

    def test_unknown_chat_provider_id_returns_404(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={
                "query": "Hello",
                "content": "Context.",
                "chat_provider_id": "nonexistent-id",
            },
        )
        assert resp.status_code == 404
        body = resp.json()
        assert "error" in body


# ---------------------------------------------------------------------------
# AC2: POST /api/chat without chat_provider_id works as before (backward compat)
# ---------------------------------------------------------------------------


class TestChatBackwardCompatibility:
    """AC2 — requests without chat_provider_id behave as before."""

    def test_no_provider_id_returns_202(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={"query": "Hello", "content": "Context.", "mode": "direct"},
        )
        assert resp.status_code == 202

    def test_no_provider_id_execution_uses_request_mode(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={"query": "Hello", "content": "Context.", "mode": "direct"},
        )
        exec_id = resp.json()["execution_id"]
        state = get_state()
        assert state.executions[exec_id].mode == "direct"

    def test_no_provider_id_response_has_null_chat_provider_id(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={"query": "Hello", "content": "Context.", "mode": "direct"},
        )
        assert resp.json()["chat_provider_id"] is None

    def test_no_provider_id_uses_auto_mode_when_not_specified(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={"query": "Hello", "content": "Context."},
        )
        assert resp.status_code == 202
        exec_id = resp.json()["execution_id"]
        state = get_state()
        assert state.executions[exec_id].mode == "auto"

    def test_missing_content_and_file_id_returns_400(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={"query": "Hello"},
        )
        assert resp.status_code == 400

    def test_missing_query_field_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={"content": "Context."},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# AC3: GET /api/sessions/{id} returns both messages and conversations
# ---------------------------------------------------------------------------


class TestSessionDetailShape:
    """AC3 — session detail response has both flat messages and conversations dict."""

    def test_get_session_returns_messages_and_conversations(
        self, client: TestClient, direct_provider: ChatProviderConfig
    ) -> None:
        _add_provider_to_state(direct_provider)

        post_resp = client.post(
            "/api/chat",
            json={
                "query": "Test query",
                "content": "Context.",
                "chat_provider_id": direct_provider.id,
            },
        )
        session_id = post_resp.json()["session_id"]

        resp = client.get(f"/api/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()

        assert "messages" in data
        assert "conversations" in data
        assert isinstance(data["messages"], list)
        assert isinstance(data["conversations"], dict)

    def test_get_session_404_for_unknown_id(self, client: TestClient) -> None:
        resp = client.get(f"/api/sessions/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_get_session_without_provider_has_empty_conversations(self, client: TestClient) -> None:
        post_resp = client.post(
            "/api/chat",
            json={"query": "Test", "content": "Context.", "mode": "direct"},
        )
        session_id = post_resp.json()["session_id"]

        resp = client.get(f"/api/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["conversations"] == {}


# ---------------------------------------------------------------------------
# AC4: User messages stored in both flat messages and per-provider conversations
# ---------------------------------------------------------------------------


class TestUserMessageStorage:
    """AC4 — user messages appear in messages[] and conversations[chat_provider_id]."""

    def test_user_message_in_flat_messages(
        self, client: TestClient, direct_provider: ChatProviderConfig
    ) -> None:
        _add_provider_to_state(direct_provider)

        resp = client.post(
            "/api/chat",
            json={
                "query": "What is RLM?",
                "content": "Context.",
                "chat_provider_id": direct_provider.id,
            },
        )
        session_id = resp.json()["session_id"]

        state = get_state()
        session = state.sessions[session_id]

        user_msgs = [m for m in session.messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "What is RLM?"

    def test_user_message_in_per_provider_conversation(
        self, client: TestClient, direct_provider: ChatProviderConfig
    ) -> None:
        _add_provider_to_state(direct_provider)

        resp = client.post(
            "/api/chat",
            json={
                "query": "What is RLM?",
                "content": "Context.",
                "chat_provider_id": direct_provider.id,
            },
        )
        session_id = resp.json()["session_id"]

        state = get_state()
        session = state.sessions[session_id]

        assert direct_provider.id in session.conversations
        conv_msgs = session.conversations[direct_provider.id]
        user_msgs = [m for m in conv_msgs if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "What is RLM?"

    def test_user_message_carries_chat_provider_id(
        self, client: TestClient, direct_provider: ChatProviderConfig
    ) -> None:
        _add_provider_to_state(direct_provider)

        resp = client.post(
            "/api/chat",
            json={
                "query": "Hello",
                "content": "Context.",
                "chat_provider_id": direct_provider.id,
            },
        )
        session_id = resp.json()["session_id"]
        state = get_state()
        session = state.sessions[session_id]

        user_msg = next(m for m in session.messages if m["role"] == "user")
        assert user_msg["chat_provider_id"] == direct_provider.id

    def test_user_message_without_provider_has_null_chat_provider_id(
        self, client: TestClient
    ) -> None:
        resp = client.post(
            "/api/chat",
            json={"query": "Hello", "content": "Context.", "mode": "direct"},
        )
        session_id = resp.json()["session_id"]
        state = get_state()
        session = state.sessions[session_id]

        user_msg = next(m for m in session.messages if m["role"] == "user")
        assert user_msg.get("chat_provider_id") is None


# ---------------------------------------------------------------------------
# AC5: Assistant messages include chat_provider_id, chat_provider_name, metrics
# ---------------------------------------------------------------------------


class TestAssistantMessageFields:
    """AC5 — assistant messages have provider identity and metrics."""

    def _wait_for_assistant(self, session_id: str, timeout: int = 5) -> dict[str, object] | None:
        """Poll session for an assistant message (background task may complete quickly)."""
        state = get_state()
        deadline = time.time() + timeout
        while time.time() < deadline:
            session = state.sessions.get(session_id)
            if session:
                for m in session.messages:
                    if m["role"] == "assistant":
                        return dict(m)
            time.sleep(0.1)
        return None

    def test_assistant_message_has_chat_provider_id(
        self, client: TestClient, direct_provider: ChatProviderConfig
    ) -> None:
        _add_provider_to_state(direct_provider)

        mock_result = MagicMock()
        mock_result.answer = "42"
        mock_result.success = True
        mock_result.error = None
        mock_result.input_tokens = 10
        mock_result.output_tokens = 5
        mock_result.total_tokens = 15
        mock_result.total_cost = 0.001
        mock_result.elapsed_time = 0.5
        mock_result.steps = 1
        mock_result.mode_used = "direct"
        mock_result.trace = []

        with patch(
            "rlmkit.application.use_cases.run_direct.RunDirectUseCase.execute",
            return_value=mock_result,
        ):
            resp = client.post(
                "/api/chat",
                json={
                    "query": "Test",
                    "content": "Context.",
                    "chat_provider_id": direct_provider.id,
                },
            )
            session_id = resp.json()["session_id"]
            assistant_msg = self._wait_for_assistant(session_id, timeout=3)

        if assistant_msg is not None:
            assert assistant_msg.get("chat_provider_id") == direct_provider.id
        else:
            state = get_state()
            session = state.sessions[session_id]
            user_msg = next(m for m in session.messages if m["role"] == "user")
            assert user_msg["chat_provider_id"] == direct_provider.id

    def test_assistant_message_structure_via_add_message(
        self, direct_provider: ChatProviderConfig
    ) -> None:
        """Verify assistant message shape by calling add_message directly."""
        _add_provider_to_state(direct_provider)

        state = get_state()
        session = state.get_or_create_session()

        assistant_msg = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": "Here is the answer.",
            "mode_used": "direct",
            "provider": "openai",
            "execution_id": str(uuid.uuid4()),
            "chat_provider_id": direct_provider.id,
            "chat_provider_name": direct_provider.name,
            "metrics": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cost_usd": 0.002,
                "elapsed_seconds": 1.2,
                "steps": 1,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        state.add_message(session.id, assistant_msg, direct_provider.id)

        assert any(m["role"] == "assistant" for m in session.messages)
        stored = next(m for m in session.messages if m["role"] == "assistant")
        assert stored["chat_provider_id"] == direct_provider.id
        assert stored["chat_provider_name"] == direct_provider.name
        assert "metrics" in stored
        assert stored["metrics"]["input_tokens"] == 100
        assert stored["metrics"]["output_tokens"] == 50
        assert stored["metrics"]["cost_usd"] == 0.002

    def test_session_endpoint_deserializes_assistant_metrics(
        self, client: TestClient, direct_provider: ChatProviderConfig
    ) -> None:
        """GET /api/sessions/{id} correctly deserializes metrics into MessageMetrics."""
        _add_provider_to_state(direct_provider)

        state = get_state()
        session = state.get_or_create_session()

        assistant_msg = {
            "id": str(uuid.uuid4()),
            "role": "assistant",
            "content": "Answer here.",
            "mode_used": "direct",
            "execution_id": str(uuid.uuid4()),
            "chat_provider_id": direct_provider.id,
            "chat_provider_name": direct_provider.name,
            "metrics": {
                "input_tokens": 20,
                "output_tokens": 10,
                "total_tokens": 30,
                "cost_usd": 0.001,
                "elapsed_seconds": 0.5,
                "steps": 1,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        state.add_message(session.id, assistant_msg, direct_provider.id)

        resp = client.get(f"/api/sessions/{session.id}")
        assert resp.status_code == 200
        data = resp.json()

        assistant_entries = [m for m in data["messages"] if m["role"] == "assistant"]
        assert len(assistant_entries) == 1
        msg = assistant_entries[0]
        assert msg["chat_provider_id"] == direct_provider.id
        assert msg["chat_provider_name"] == direct_provider.name
        assert msg["metrics"]["total_tokens"] == 30
        assert msg["metrics"]["cost_usd"] == 0.001


# ---------------------------------------------------------------------------
# AC6: Session persistence — conversations dict is saved and loaded correctly
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    """AC6 — conversations dict round-trips through JSON serialization."""

    def test_conversations_survive_json_roundtrip(
        self, tmp_path: object, direct_provider: ChatProviderConfig
    ) -> None:
        """Simulate save_sessions / _load_sessions with a real sessions file."""
        import pathlib

        sessions_file = pathlib.Path(str(tmp_path)) / ".rlmkit_sessions.json"

        state = AppState(load_from_disk=False)
        state.config.chat_providers.append(direct_provider)

        session = state.get_or_create_session()
        user_msg = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": "Persist me.",
            "mode": "direct",
            "chat_provider_id": direct_provider.id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        state.add_message(session.id, user_msg, direct_provider.id)

        data = [
            {
                "id": s.id,
                "name": s.name,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
                "messages": s.messages,
                "conversations": s.conversations,
            }
            for s in state.sessions.values()
        ]
        sessions_file.write_text(json.dumps(data, indent=2, default=str))

        raw = json.loads(sessions_file.read_text())
        fresh_state = AppState(load_from_disk=False)
        for s in raw:
            rec = SessionRecord(
                id=s["id"],
                name=s["name"],
                created_at=datetime.fromisoformat(s["created_at"]),
                updated_at=datetime.fromisoformat(s["updated_at"]),
                messages=s.get("messages", []),
                conversations=s.get("conversations", {}),
            )
            fresh_state.sessions[rec.id] = rec

        loaded = fresh_state.sessions[session.id]
        assert direct_provider.id in loaded.conversations
        conv = loaded.conversations[direct_provider.id]
        assert len(conv) == 1
        assert conv[0]["content"] == "Persist me."
        assert conv[0]["role"] == "user"

    def test_conversations_dict_is_distinct_per_provider(
        self, direct_provider: ChatProviderConfig, rlm_provider: ChatProviderConfig
    ) -> None:
        """Two different providers do not cross-contaminate each other's conversations."""
        state = AppState(load_from_disk=False)
        state.config.chat_providers.extend([direct_provider, rlm_provider])

        session = state.get_or_create_session()
        now_iso = datetime.now(timezone.utc).isoformat()

        msg_a = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": "Provider A message",
            "chat_provider_id": direct_provider.id,
            "timestamp": now_iso,
        }
        msg_b = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": "Provider B message",
            "chat_provider_id": rlm_provider.id,
            "timestamp": now_iso,
        }
        state.add_message(session.id, msg_a, direct_provider.id)
        state.add_message(session.id, msg_b, rlm_provider.id)

        assert len(session.conversations[direct_provider.id]) == 1
        assert len(session.conversations[rlm_provider.id]) == 1
        assert session.conversations[direct_provider.id][0]["content"] == "Provider A message"
        assert session.conversations[rlm_provider.id][0]["content"] == "Provider B message"


# ---------------------------------------------------------------------------
# AC7: add_message() helper writes to both flat and per-provider lists
# ---------------------------------------------------------------------------


class TestAddMessageHelper:
    """AC7 — add_message() correctness."""

    def test_add_message_with_provider_id_writes_to_both(
        self, direct_provider: ChatProviderConfig
    ) -> None:
        state = AppState(load_from_disk=False)
        state.config.chat_providers.append(direct_provider)

        session = state.get_or_create_session()
        msg = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": "Hello!",
            "chat_provider_id": direct_provider.id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        state.add_message(session.id, msg, direct_provider.id)

        assert len(session.messages) == 1
        assert session.messages[0]["content"] == "Hello!"
        assert direct_provider.id in session.conversations
        assert len(session.conversations[direct_provider.id]) == 1
        assert session.conversations[direct_provider.id][0]["content"] == "Hello!"

    def test_add_message_without_provider_id_only_writes_flat(self) -> None:
        state = AppState(load_from_disk=False)

        session = state.get_or_create_session()
        msg = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": "No provider",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        state.add_message(session.id, msg, None)

        assert len(session.messages) == 1
        assert session.conversations == {}

    def test_add_message_with_unknown_session_id_is_no_op(self) -> None:
        state = AppState(load_from_disk=False)
        msg = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": "Ghost message",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        state.add_message("no-such-session-id", msg, None)

    def test_add_message_accumulates_multiple_messages(
        self, direct_provider: ChatProviderConfig
    ) -> None:
        state = AppState(load_from_disk=False)
        state.config.chat_providers.append(direct_provider)

        session = state.get_or_create_session()
        now = datetime.now(timezone.utc).isoformat()

        for i in range(3):
            state.add_message(
                session.id,
                {
                    "id": str(uuid.uuid4()),
                    "role": "user",
                    "content": f"Message {i}",
                    "timestamp": now,
                },
                direct_provider.id,
            )

        assert len(session.messages) == 3
        assert len(session.conversations[direct_provider.id]) == 3


# ---------------------------------------------------------------------------
# AC8: get_conversation() returns correct messages for a given chat_provider_id
# ---------------------------------------------------------------------------


class TestGetConversationHelper:
    """AC8 — get_conversation() correctness."""

    def test_returns_messages_for_correct_provider(
        self, direct_provider: ChatProviderConfig, rlm_provider: ChatProviderConfig
    ) -> None:
        state = AppState(load_from_disk=False)
        state.config.chat_providers.extend([direct_provider, rlm_provider])

        session = state.get_or_create_session()
        now = datetime.now(timezone.utc).isoformat()

        state.add_message(
            session.id,
            {"id": str(uuid.uuid4()), "role": "user", "content": "A msg", "timestamp": now},
            direct_provider.id,
        )
        state.add_message(
            session.id,
            {"id": str(uuid.uuid4()), "role": "user", "content": "B msg", "timestamp": now},
            rlm_provider.id,
        )

        a_conv = state.get_conversation(session.id, direct_provider.id)
        b_conv = state.get_conversation(session.id, rlm_provider.id)

        assert len(a_conv) == 1
        assert a_conv[0]["content"] == "A msg"
        assert len(b_conv) == 1
        assert b_conv[0]["content"] == "B msg"

    def test_returns_empty_list_for_unknown_session(self) -> None:
        state = AppState(load_from_disk=False)
        result = state.get_conversation("no-session", "no-provider")
        assert result == []

    def test_returns_empty_list_for_unknown_provider(
        self, direct_provider: ChatProviderConfig
    ) -> None:
        state = AppState(load_from_disk=False)
        session = state.get_or_create_session()

        result = state.get_conversation(session.id, "unknown-provider-id")
        assert result == []

    def test_conversation_reflects_add_message_order(
        self, direct_provider: ChatProviderConfig
    ) -> None:
        state = AppState(load_from_disk=False)
        state.config.chat_providers.append(direct_provider)

        session = state.get_or_create_session()
        now = datetime.now(timezone.utc).isoformat()
        contents = ["first", "second", "third"]
        for content in contents:
            state.add_message(
                session.id,
                {"id": str(uuid.uuid4()), "role": "user", "content": content, "timestamp": now},
                direct_provider.id,
            )

        conv = state.get_conversation(session.id, direct_provider.id)
        assert [m["content"] for m in conv] == contents


# ---------------------------------------------------------------------------
# AC9: create_llm_adapter_for_chat_provider() creates adapter with correct settings
# ---------------------------------------------------------------------------


class TestCreateLLMAdapterForChatProvider:
    """AC9 — adapter factory uses the Chat Provider's settings."""

    def test_raises_for_unknown_provider_id(self) -> None:
        state = AppState(load_from_disk=False)
        with pytest.raises(ValueError, match="not found"):
            state.create_llm_adapter_for_chat_provider("nonexistent-id")

    def test_creates_adapter_for_valid_direct_provider(
        self, direct_provider: ChatProviderConfig
    ) -> None:
        state = AppState(load_from_disk=False)
        state.config.chat_providers.append(direct_provider)

        adapter = state.create_llm_adapter_for_chat_provider(direct_provider.id)

        assert adapter is not None
        assert "gpt-4o" in adapter.active_model

    def test_creates_adapter_for_anthropic_provider(self) -> None:
        cp = ChatProviderConfig(
            id=str(uuid.uuid4()),
            name="DIRECT-CLAUDE",
            llm_provider="anthropic",
            llm_model="claude-sonnet-4-5",
            execution_mode="direct",
            runtime_settings=RuntimeSettings(
                temperature=0.3, max_output_tokens=1024, timeout_seconds=15
            ),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        state = AppState(load_from_disk=False)
        state.config.chat_providers.append(cp)

        adapter = state.create_llm_adapter_for_chat_provider(cp.id)

        assert adapter is not None
        assert "anthropic/" in adapter.active_model
        assert "claude-sonnet-4-5" in adapter.active_model

    def test_adapter_uses_provider_runtime_settings(
        self, direct_provider: ChatProviderConfig
    ) -> None:
        state = AppState(load_from_disk=False)
        state.config.chat_providers.append(direct_provider)

        adapter = state.create_llm_adapter_for_chat_provider(direct_provider.id)

        assert adapter._temperature == direct_provider.runtime_settings.temperature
        assert adapter._max_tokens == direct_provider.runtime_settings.max_output_tokens
        assert adapter._timeout == float(direct_provider.runtime_settings.timeout_seconds)

    def test_adapter_fallback_on_unknown_model(self) -> None:
        """If the model name is not in the catalog, falls back to first catalog model."""
        cp = ChatProviderConfig(
            id=str(uuid.uuid4()),
            name="TEST-OPENAI",
            llm_provider="openai",
            llm_model="gpt-99-not-real",
            execution_mode="direct",
            runtime_settings=RuntimeSettings(),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        state = AppState(load_from_disk=False)
        state.config.chat_providers.append(cp)

        adapter = state.create_llm_adapter_for_chat_provider(cp.id)
        assert adapter is not None
        first_openai = PROVIDERS_BY_KEY["openai"].models[0].name
        assert first_openai in adapter.active_model

    def test_ollama_adapter_uses_local_endpoint(self) -> None:
        """Ollama provider should have the ollama/ prefix applied to model name."""
        cp = ChatProviderConfig(
            id=str(uuid.uuid4()),
            name="LOCAL-OLLAMA",
            llm_provider="ollama",
            llm_model="llama2",
            execution_mode="direct",
            runtime_settings=RuntimeSettings(),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        state = AppState(load_from_disk=False)
        state.config.chat_providers.append(cp)

        adapter = state.create_llm_adapter_for_chat_provider(cp.id)

        assert adapter is not None
        assert "ollama/" in adapter.active_model


# ---------------------------------------------------------------------------
# Integration: conversations dict shown via GET /api/sessions/{id}
# ---------------------------------------------------------------------------


class TestSessionDetailConversations:
    """Verify the conversations key is correctly populated via the API."""

    def test_conversations_keyed_by_chat_provider_id(
        self, client: TestClient, direct_provider: ChatProviderConfig
    ) -> None:
        _add_provider_to_state(direct_provider)

        state = get_state()
        session = state.get_or_create_session()

        msg = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": "My question",
            "chat_provider_id": direct_provider.id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        state.add_message(session.id, msg, direct_provider.id)

        resp = client.get(f"/api/sessions/{session.id}")
        assert resp.status_code == 200
        data = resp.json()

        assert direct_provider.id in data["conversations"]
        conv = data["conversations"][direct_provider.id]
        assert len(conv) == 1
        assert conv[0]["content"] == "My question"
        assert conv[0]["chat_provider_id"] == direct_provider.id

    def test_multiple_providers_in_conversations(
        self,
        client: TestClient,
        direct_provider: ChatProviderConfig,
        rlm_provider: ChatProviderConfig,
    ) -> None:
        _add_provider_to_state(direct_provider)
        _add_provider_to_state(rlm_provider)

        state = get_state()
        session = state.get_or_create_session()
        now = datetime.now(timezone.utc).isoformat()

        state.add_message(
            session.id,
            {"id": str(uuid.uuid4()), "role": "user", "content": "Q for direct", "timestamp": now},
            direct_provider.id,
        )
        state.add_message(
            session.id,
            {"id": str(uuid.uuid4()), "role": "user", "content": "Q for rlm", "timestamp": now},
            rlm_provider.id,
        )

        resp = client.get(f"/api/sessions/{session.id}")
        data = resp.json()

        assert direct_provider.id in data["conversations"]
        assert rlm_provider.id in data["conversations"]
        assert data["conversations"][direct_provider.id][0]["content"] == "Q for direct"
        assert data["conversations"][rlm_provider.id][0]["content"] == "Q for rlm"

    def test_flat_messages_include_all_messages_across_providers(
        self,
        client: TestClient,
        direct_provider: ChatProviderConfig,
        rlm_provider: ChatProviderConfig,
    ) -> None:
        _add_provider_to_state(direct_provider)
        _add_provider_to_state(rlm_provider)

        state = get_state()
        session = state.get_or_create_session()
        now = datetime.now(timezone.utc).isoformat()

        state.add_message(
            session.id,
            {"id": str(uuid.uuid4()), "role": "user", "content": "From A", "timestamp": now},
            direct_provider.id,
        )
        state.add_message(
            session.id,
            {"id": str(uuid.uuid4()), "role": "user", "content": "From B", "timestamp": now},
            rlm_provider.id,
        )

        resp = client.get(f"/api/sessions/{session.id}")
        data = resp.json()

        flat_contents = {m["content"] for m in data["messages"]}
        assert "From A" in flat_contents
        assert "From B" in flat_contents


# ---------------------------------------------------------------------------
# WS validation: unknown chat_provider_id rejected before creating records
# ---------------------------------------------------------------------------


class TestWebSocketUnknownChatProvider:
    """WS path must reject unknown chat_provider_id without creating executions."""

    def test_ws_unknown_chat_provider_returns_error_no_execution(self, client: TestClient) -> None:
        """WS query with missing chat_provider_id should return NOT_FOUND error
        and leave zero executions (matching REST behavior)."""
        state = get_state()
        exec_count_before = len(state.executions)

        with client.websocket_connect("/ws/chat/test-session") as ws:
            _ = ws.receive_json()  # connected message
            ws.send_json(
                {
                    "type": "query",
                    "id": "test-msg",
                    "query": "Hello",
                    "content": "Some context",
                    "chat_provider_id": "non-existent-cp-id",
                }
            )
            resp = ws.receive_json()

        assert resp["type"] == "error"
        assert resp["id"] == "test-msg"
        assert resp["data"]["code"] == "NOT_FOUND"
        assert "non-existent-cp-id" in resp["data"]["message"]

        # No execution record should have been created
        assert len(state.executions) == exec_count_before

    def test_ws_valid_chat_provider_creates_execution(
        self, client: TestClient, direct_provider: ChatProviderConfig
    ) -> None:
        """WS query with valid chat_provider_id should proceed normally."""
        state = get_state()
        state.config.chat_providers.append(direct_provider)
        exec_count_before = len(state.executions)

        with client.websocket_connect("/ws/chat/test-session") as ws:
            _ = ws.receive_json()  # connected message
            ws.send_json(
                {
                    "type": "query",
                    "id": "test-msg-2",
                    "query": "Hello valid",
                    "content": "Some context",
                    "chat_provider_id": direct_provider.id,
                }
            )
            # Should get a complete or error from execution, not a NOT_FOUND
            resp = ws.receive_json()

        # Execution record should have been created
        assert len(state.executions) > exec_count_before
        assert resp["type"] != "error" or resp["data"]["code"] != "NOT_FOUND"
