"""Acceptance tests for Chat Providers CRUD API (spec 01).

Verifies all acceptance criteria from specs/01-backend-models-crud.md:
  1. GET /api/chat-providers returns [] on fresh install (after reset_state)
  2. POST /api/chat-providers creates/persists; rejects duplicates (409) and
     invalid provider keys (400)
  3. PUT /api/chat-providers/{id} updates fields; validates name uniqueness (409);
     returns 404 for missing ID
  4. DELETE /api/chat-providers/{id} removes; returns 404 for missing; 204 on success
  5. Auto-migration creates one "DIRECT-{PROVIDER}" Chat Provider per enabled
     provider config on first load
  6. ChatRequest model accepts chat_provider_id field
  7. ChatResponse model includes chat_provider_id field
  8. SessionMessage model includes chat_provider_id and chat_provider_name fields
  9. SessionDetail model includes conversations dict field
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any

import pytest
from starlette.testclient import TestClient

from rlmkit.server.app import app
from rlmkit.server.dependencies import get_state, reset_state
from rlmkit.server.models import (
    ChatRequest,
    ChatResponse,
    ProviderConfig,
    RuntimeSettings,
    SessionDetail,
    SessionMessage,
)
from rlmkit.ui.data.providers_catalog import PROVIDERS_BY_KEY

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    """Reset in-memory state (no disk I/O) before and after each test."""
    reset_state()
    yield
    reset_state()


@pytest.fixture()
def client() -> TestClient:
    """FastAPI TestClient backed by the real application."""
    return TestClient(app)


@pytest.fixture()
def valid_provider_key() -> str:
    """Return the first key from the catalog that is recognised by the route."""
    return str(next(iter(PROVIDERS_BY_KEY)))


@pytest.fixture()
def valid_model(valid_provider_key: str) -> str:
    """Return a valid model name for the chosen provider, or a fallback."""
    entry = PROVIDERS_BY_KEY[valid_provider_key]
    if entry.models:
        return str(entry.models[0].name)
    return "test-model"


@pytest.fixture()
def created_provider(
    client: TestClient, valid_provider_key: str, valid_model: str
) -> dict[str, Any]:
    """Create one Chat Provider via the API and return the response JSON."""
    resp = client.post(
        "/api/chat-providers",
        json={
            "name": "TEST-PROVIDER",
            "llm_provider": valid_provider_key,
            "llm_model": valid_model,
            "execution_mode": "direct",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# AC-1: GET /api/chat-providers returns [] on fresh install
# ---------------------------------------------------------------------------


class TestListEmpty:
    def test_returns_empty_list_after_reset(self, client: TestClient) -> None:
        """After reset_state() the list must be empty."""
        resp = client.get("/api/chat-providers")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_response_is_json_array(self, client: TestClient) -> None:
        resp = client.get("/api/chat-providers")
        assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# AC-2: POST /api/chat-providers — create, persist, and reject bad input
# ---------------------------------------------------------------------------


class TestCreateChatProvider:
    def test_creates_with_201(
        self, client: TestClient, valid_provider_key: str, valid_model: str
    ) -> None:
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "MY-PROVIDER",
                "llm_provider": valid_provider_key,
                "llm_model": valid_model,
                "execution_mode": "direct",
            },
        )
        assert resp.status_code == 201, resp.text

    def test_response_contains_required_fields(
        self, client: TestClient, valid_provider_key: str, valid_model: str
    ) -> None:
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "MY-PROVIDER",
                "llm_provider": valid_provider_key,
                "llm_model": valid_model,
                "execution_mode": "direct",
            },
        )
        data = resp.json()
        assert "id" in data
        assert data["name"] == "MY-PROVIDER"
        assert data["llm_provider"] == valid_provider_key
        assert data["llm_model"] == valid_model
        assert data["execution_mode"] == "direct"

    def test_assigns_uuid_id(
        self, client: TestClient, valid_provider_key: str, valid_model: str
    ) -> None:
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "UUID-CHECK",
                "llm_provider": valid_provider_key,
                "llm_model": valid_model,
                "execution_mode": "direct",
            },
        )
        data = resp.json()
        # Should not raise if it is a valid UUID
        uuid.UUID(data["id"])

    def test_persists_in_state(
        self, client: TestClient, valid_provider_key: str, valid_model: str
    ) -> None:
        """Created provider must be retrievable from state without another POST."""
        client.post(
            "/api/chat-providers",
            json={
                "name": "PERSIST-CHECK",
                "llm_provider": valid_provider_key,
                "llm_model": valid_model,
                "execution_mode": "direct",
            },
        )
        state = get_state()
        names = [cp.name for cp in state.config.chat_providers]
        assert "PERSIST-CHECK" in names

    def test_appears_in_list_after_create(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """GET list should include newly created provider."""
        resp = client.get("/api/chat-providers")
        assert resp.status_code == 200
        ids = [cp["id"] for cp in resp.json()]
        assert created_provider["id"] in ids

    def test_duplicate_name_returns_409(
        self, client: TestClient, valid_provider_key: str, valid_model: str
    ) -> None:
        """Second POST with same name must return 409 CONFLICT."""
        payload = {
            "name": "DUPLICATE",
            "llm_provider": valid_provider_key,
            "llm_model": valid_model,
            "execution_mode": "direct",
        }
        client.post("/api/chat-providers", json=payload)
        resp = client.post("/api/chat-providers", json=payload)
        assert resp.status_code == 409, resp.text

    def test_duplicate_name_case_insensitive(
        self, client: TestClient, valid_provider_key: str, valid_model: str
    ) -> None:
        """Name uniqueness check must be case-insensitive."""
        client.post(
            "/api/chat-providers",
            json={
                "name": "MyProvider",
                "llm_provider": valid_provider_key,
                "llm_model": valid_model,
                "execution_mode": "direct",
            },
        )
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "myprovider",
                "llm_provider": valid_provider_key,
                "llm_model": valid_model,
                "execution_mode": "direct",
            },
        )
        assert resp.status_code == 409, resp.text

    def test_invalid_provider_key_returns_400(self, client: TestClient) -> None:
        """Unknown llm_provider must return 400."""
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "BAD-PROVIDER",
                "llm_provider": "not-a-real-provider-xyz",
                "llm_model": "some-model",
                "execution_mode": "direct",
            },
        )
        assert resp.status_code == 400, resp.text

    def test_all_execution_modes_accepted(
        self, client: TestClient, valid_provider_key: str, valid_model: str
    ) -> None:
        """All three execution modes must be accepted."""
        for mode in ("direct", "rlm", "rag"):
            resp = client.post(
                "/api/chat-providers",
                json={
                    "name": f"PROVIDER-{mode.upper()}",
                    "llm_provider": valid_provider_key,
                    "llm_model": valid_model,
                    "execution_mode": mode,
                },
            )
            assert resp.status_code == 201, f"mode={mode}: {resp.text}"

    def test_timestamps_set_on_create(
        self, client: TestClient, valid_provider_key: str, valid_model: str
    ) -> None:
        resp = client.post(
            "/api/chat-providers",
            json={
                "name": "TIMESTAMP-CHECK",
                "llm_provider": valid_provider_key,
                "llm_model": valid_model,
                "execution_mode": "direct",
            },
        )
        data = resp.json()
        assert data["created_at"] is not None
        assert data["updated_at"] is not None


# ---------------------------------------------------------------------------
# AC-3: PUT /api/chat-providers/{id} — update fields
# ---------------------------------------------------------------------------


class TestUpdateChatProvider:
    def test_update_name(self, client: TestClient, created_provider: dict[str, Any]) -> None:
        cp_id = created_provider["id"]
        resp = client.put(
            f"/api/chat-providers/{cp_id}",
            json={"name": "RENAMED"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["name"] == "RENAMED"

    def test_update_model(
        self, client: TestClient, created_provider: dict[str, Any], valid_provider_key: str
    ) -> None:
        cp_id = created_provider["id"]
        entry = PROVIDERS_BY_KEY[valid_provider_key]
        if len(entry.models) < 2:
            pytest.skip("Not enough models in catalog to test model update")
        new_model = entry.models[1].name
        resp = client.put(
            f"/api/chat-providers/{cp_id}",
            json={"llm_model": new_model},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["llm_model"] == new_model

    def test_update_execution_mode(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        cp_id = created_provider["id"]
        resp = client.put(
            f"/api/chat-providers/{cp_id}",
            json={"execution_mode": "rlm"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["execution_mode"] == "rlm"

    def test_update_runtime_settings(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        cp_id = created_provider["id"]
        resp = client.put(
            f"/api/chat-providers/{cp_id}",
            json={"runtime_settings": {"temperature": 0.2, "max_output_tokens": 1024}},
        )
        assert resp.status_code == 200, resp.text
        rt = resp.json()["runtime_settings"]
        assert rt["temperature"] == 0.2
        assert rt["max_output_tokens"] == 1024

    def test_update_updates_timestamp(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        cp_id = created_provider["id"]
        resp = client.put(
            f"/api/chat-providers/{cp_id}",
            json={"name": "UPDATED"},
        )
        assert resp.status_code == 200
        assert resp.json()["updated_at"] is not None

    def test_update_nonexistent_returns_404(self, client: TestClient) -> None:
        fake_id = str(uuid.uuid4())
        resp = client.put(
            f"/api/chat-providers/{fake_id}",
            json={"name": "WHATEVER"},
        )
        assert resp.status_code == 404, resp.text

    def test_update_name_to_duplicate_returns_409(
        self, client: TestClient, valid_provider_key: str, valid_model: str
    ) -> None:
        """Renaming to an already-used name must return 409."""
        client.post(
            "/api/chat-providers",
            json={
                "name": "FIRST",
                "llm_provider": valid_provider_key,
                "llm_model": valid_model,
                "execution_mode": "direct",
            },
        )
        resp2 = client.post(
            "/api/chat-providers",
            json={
                "name": "SECOND",
                "llm_provider": valid_provider_key,
                "llm_model": valid_model,
                "execution_mode": "direct",
            },
        )
        second_id = resp2.json()["id"]
        resp = client.put(
            f"/api/chat-providers/{second_id}",
            json={"name": "FIRST"},
        )
        assert resp.status_code == 409, resp.text

    def test_update_same_name_allowed(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """Updating to the same name (no actual change) must not raise 409."""
        cp_id = created_provider["id"]
        same_name = created_provider["name"]
        resp = client.put(
            f"/api/chat-providers/{cp_id}",
            json={"name": same_name},
        )
        assert resp.status_code == 200, resp.text

    def test_get_by_id_returns_updated_data(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        cp_id = created_provider["id"]
        client.put(f"/api/chat-providers/{cp_id}", json={"name": "GET-UPDATED"})
        resp = client.get(f"/api/chat-providers/{cp_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "GET-UPDATED"


# ---------------------------------------------------------------------------
# AC-4: DELETE /api/chat-providers/{id}
# ---------------------------------------------------------------------------


class TestDeleteChatProvider:
    def test_delete_returns_204(self, client: TestClient, created_provider: dict[str, Any]) -> None:
        cp_id = created_provider["id"]
        resp = client.delete(f"/api/chat-providers/{cp_id}")
        assert resp.status_code == 204, resp.text

    def test_delete_removes_from_list(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        cp_id = created_provider["id"]
        client.delete(f"/api/chat-providers/{cp_id}")
        resp = client.get("/api/chat-providers")
        ids = [cp["id"] for cp in resp.json()]
        assert cp_id not in ids

    def test_delete_nonexistent_returns_404(self, client: TestClient) -> None:
        fake_id = str(uuid.uuid4())
        resp = client.delete(f"/api/chat-providers/{fake_id}")
        assert resp.status_code == 404, resp.text

    def test_get_by_id_after_delete_returns_404(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        cp_id = created_provider["id"]
        client.delete(f"/api/chat-providers/{cp_id}")
        resp = client.get(f"/api/chat-providers/{cp_id}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# AC-5: Auto-migration creates "DIRECT-{PROVIDER}" on first load
# ---------------------------------------------------------------------------


class TestAutoMigration:
    def test_migration_creates_chat_provider_for_enabled_config(self) -> None:
        """When provider_configs has an enabled entry, migration should create a CP."""
        state = get_state()
        assert state.config.chat_providers == []

        pc = ProviderConfig(
            provider="openai",
            model="gpt-4o",
            enabled=True,
            runtime_settings=RuntimeSettings(),
        )
        state.config.provider_configs.append(pc)
        state._migrate_chat_providers()

        names = [cp.name for cp in state.config.chat_providers]
        assert "DIRECT-OPENAI" in names

    def test_migration_skips_disabled_provider_configs(self) -> None:
        """Disabled provider configs must not generate Chat Providers."""
        state = get_state()
        pc = ProviderConfig(
            provider="anthropic",
            model="claude-sonnet-4-5",
            enabled=False,
            runtime_settings=RuntimeSettings(),
        )
        state.config.provider_configs.append(pc)
        state._migrate_chat_providers()
        names = [cp.name for cp in state.config.chat_providers]
        assert "DIRECT-ANTHROPIC" not in names

    def test_migration_is_idempotent(self) -> None:
        """Calling migration twice must not create duplicate Chat Providers."""
        state = get_state()
        pc = ProviderConfig(
            provider="openai",
            model="gpt-4o",
            enabled=True,
            runtime_settings=RuntimeSettings(),
        )
        state.config.provider_configs.append(pc)
        state._migrate_chat_providers()
        count_after_first = len(state.config.chat_providers)
        state._migrate_chat_providers()
        count_after_second = len(state.config.chat_providers)
        assert count_after_first == count_after_second

    def test_migration_uses_direct_mode(self) -> None:
        """Migrated Chat Providers must have execution_mode='direct'."""
        state = get_state()
        pc = ProviderConfig(
            provider="openai",
            model="gpt-4o",
            enabled=True,
            runtime_settings=RuntimeSettings(),
        )
        state.config.provider_configs.append(pc)
        state._migrate_chat_providers()
        for cp in state.config.chat_providers:
            assert cp.execution_mode == "direct"

    def test_migration_propagates_runtime_settings(self) -> None:
        """Migrated Chat Providers should copy runtime_settings from ProviderConfig."""
        state = get_state()
        rt = RuntimeSettings(temperature=0.3, max_output_tokens=512)
        pc = ProviderConfig(
            provider="openai",
            model="gpt-4o",
            enabled=True,
            runtime_settings=rt,
        )
        state.config.provider_configs.append(pc)
        state._migrate_chat_providers()
        cp = state.config.chat_providers[0]
        assert cp.runtime_settings.temperature == 0.3
        assert cp.runtime_settings.max_output_tokens == 512

    def test_migration_name_format(self) -> None:
        """Migrated names must follow 'DIRECT-{PROVIDER_UPPERCASE}' pattern."""
        state = get_state()
        for key in ("openai", "anthropic"):
            pc = ProviderConfig(
                provider=key,
                model="some-model",
                enabled=True,
                runtime_settings=RuntimeSettings(),
            )
            state.config.provider_configs.append(pc)
        state._migrate_chat_providers()
        names = [cp.name for cp in state.config.chat_providers]
        for key in ("openai", "anthropic"):
            assert f"DIRECT-{key.upper()}" in names


# ---------------------------------------------------------------------------
# AC-6: ChatRequest model accepts chat_provider_id field
# ---------------------------------------------------------------------------


class TestChatRequestModel:
    def test_chat_request_accepts_chat_provider_id(self) -> None:
        req = ChatRequest(
            query="hello",
            content="some content",
            chat_provider_id="cp-123",
        )
        assert req.chat_provider_id == "cp-123"

    def test_chat_provider_id_is_optional(self) -> None:
        req = ChatRequest(query="hello", content="some content")
        assert req.chat_provider_id is None

    def test_chat_request_via_api_accepts_chat_provider_id(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """POST /api/chat with chat_provider_id must not be rejected as invalid input."""
        resp = client.post(
            "/api/chat",
            json={
                "query": "test",
                "content": "context",
                "mode": "direct",
                "chat_provider_id": created_provider["id"],
            },
        )
        # 202 = accepted; 404 = cp not found is acceptable here but not a 422
        assert resp.status_code in (202, 404), f"Unexpected status: {resp.status_code} {resp.text}"

    def test_chat_request_without_chat_provider_id_still_works(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={
                "query": "test",
                "content": "context",
                "mode": "direct",
            },
        )
        assert resp.status_code == 202, resp.text


# ---------------------------------------------------------------------------
# AC-7: ChatResponse model includes chat_provider_id field
# ---------------------------------------------------------------------------


class TestChatResponseModel:
    def test_chat_response_has_chat_provider_id_field(self) -> None:
        resp = ChatResponse(
            execution_id="exec-1",
            session_id="sess-1",
            status="running",
            chat_provider_id="cp-abc",
        )
        assert resp.chat_provider_id == "cp-abc"

    def test_chat_provider_id_is_optional_on_response(self) -> None:
        resp = ChatResponse(
            execution_id="exec-1",
            session_id="sess-1",
        )
        assert resp.chat_provider_id is None

    def test_api_response_echoes_chat_provider_id(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """The API response must echo back the chat_provider_id that was sent."""
        resp = client.post(
            "/api/chat",
            json={
                "query": "echo test",
                "content": "ctx",
                "mode": "direct",
                "chat_provider_id": created_provider["id"],
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "chat_provider_id" in data
        assert data["chat_provider_id"] == created_provider["id"]

    def test_api_response_chat_provider_id_null_when_not_sent(self, client: TestClient) -> None:
        resp = client.post(
            "/api/chat",
            json={"query": "test", "content": "ctx", "mode": "direct"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "chat_provider_id" in data
        assert data["chat_provider_id"] is None


# ---------------------------------------------------------------------------
# AC-8: SessionMessage model includes chat_provider_id and chat_provider_name
# ---------------------------------------------------------------------------


class TestSessionMessageModel:
    def test_session_message_has_chat_provider_id_field(self) -> None:
        msg = SessionMessage(
            id="msg-1",
            role="user",
            content="hello",
            timestamp=datetime.now(timezone.utc),
            chat_provider_id="cp-123",
        )
        assert msg.chat_provider_id == "cp-123"

    def test_session_message_has_chat_provider_name_field(self) -> None:
        msg = SessionMessage(
            id="msg-1",
            role="assistant",
            content="reply",
            timestamp=datetime.now(timezone.utc),
            chat_provider_name="MY-PROVIDER",
        )
        assert msg.chat_provider_name == "MY-PROVIDER"

    def test_session_message_both_fields_optional(self) -> None:
        msg = SessionMessage(
            id="msg-1",
            role="user",
            content="hi",
            timestamp=datetime.now(timezone.utc),
        )
        assert msg.chat_provider_id is None
        assert msg.chat_provider_name is None

    def test_session_message_fields_reflected_in_api(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """Messages stored with chat_provider_id must be retrievable via GET /api/sessions/{id}."""
        chat_resp = client.post(
            "/api/chat",
            json={
                "query": "hello",
                "content": "ctx",
                "mode": "direct",
                "chat_provider_id": created_provider["id"],
            },
        )
        assert chat_resp.status_code == 202
        session_id = chat_resp.json()["session_id"]

        detail_resp = client.get(f"/api/sessions/{session_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        messages = detail["messages"]
        assert len(messages) > 0
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) > 0
        assert user_msgs[0]["chat_provider_id"] == created_provider["id"]


# ---------------------------------------------------------------------------
# AC-9: SessionDetail model includes conversations dict field
# ---------------------------------------------------------------------------


class TestSessionDetailModel:
    def test_session_detail_has_conversations_field(self) -> None:
        detail = SessionDetail(
            id="s-1",
            name="Test Session",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert hasattr(detail, "conversations")
        assert isinstance(detail.conversations, dict)

    def test_session_detail_conversations_default_empty(self) -> None:
        detail = SessionDetail(
            id="s-1",
            name="Test Session",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert detail.conversations == {}

    def test_session_detail_conversations_keyed_by_provider_id(self) -> None:
        msg = SessionMessage(
            id="m-1",
            role="user",
            content="hi",
            timestamp=datetime.now(timezone.utc),
            chat_provider_id="cp-abc",
        )
        detail = SessionDetail(
            id="s-1",
            name="Test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            conversations={"cp-abc": [msg]},
        )
        assert "cp-abc" in detail.conversations
        assert detail.conversations["cp-abc"][0].chat_provider_id == "cp-abc"

    def test_api_session_detail_includes_conversations_key(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """GET /api/sessions/{id} response must include 'conversations' key."""
        chat_resp = client.post(
            "/api/chat",
            json={
                "query": "test",
                "content": "ctx",
                "mode": "direct",
                "chat_provider_id": created_provider["id"],
            },
        )
        assert chat_resp.status_code == 202
        session_id = chat_resp.json()["session_id"]

        detail_resp = client.get(f"/api/sessions/{session_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert "conversations" in detail
        assert isinstance(detail["conversations"], dict)

    def test_api_conversations_keyed_by_chat_provider_id(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        """conversations dict keys must be the chat_provider_id values."""
        chat_resp = client.post(
            "/api/chat",
            json={
                "query": "hello",
                "content": "ctx",
                "mode": "direct",
                "chat_provider_id": created_provider["id"],
            },
        )
        assert chat_resp.status_code == 202
        session_id = chat_resp.json()["session_id"]

        detail_resp = client.get(f"/api/sessions/{session_id}")
        assert detail_resp.status_code == 200
        conversations = detail_resp.json()["conversations"]
        assert created_provider["id"] in conversations


# ---------------------------------------------------------------------------
# GET /api/chat-providers/{id} — supplementary coverage
# ---------------------------------------------------------------------------


class TestGetChatProviderById:
    def test_get_by_id_returns_200(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        cp_id = created_provider["id"]
        resp = client.get(f"/api/chat-providers/{cp_id}")
        assert resp.status_code == 200

    def test_get_by_id_returns_correct_data(
        self, client: TestClient, created_provider: dict[str, Any]
    ) -> None:
        cp_id = created_provider["id"]
        resp = client.get(f"/api/chat-providers/{cp_id}")
        data = resp.json()
        assert data["id"] == cp_id
        assert data["name"] == created_provider["name"]

    def test_get_nonexistent_returns_404(self, client: TestClient) -> None:
        fake_id = str(uuid.uuid4())
        resp = client.get(f"/api/chat-providers/{fake_id}")
        assert resp.status_code == 404
