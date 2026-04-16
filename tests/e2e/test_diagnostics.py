"""E2E tests for the /api/diagnostics endpoint.

Covers the four checks (backend, provider, judge, storage) against the
real FastAPI app via TestClient. State is reset between tests via the
autouse ``_clean_state`` fixture in ``conftest.py``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rlmkit.server.dependencies import get_state
from rlmkit.server.models import ProviderConfig

pytestmark = [pytest.mark.e2e]


class TestDiagnosticsEndpoint:
    """GET /api/diagnostics."""

    def test_returns_200_with_all_four_checks(self, client: TestClient) -> None:
        resp = client.get("/api/diagnostics")
        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {"backend", "provider", "judge", "storage"}
        for key in ("backend", "provider", "judge", "storage"):
            assert data[key]["status"] in {"ok", "warn", "error"}
            assert isinstance(data[key]["message"], str)

    def test_backend_check_is_ok(self, client: TestClient) -> None:
        data = client.get("/api/diagnostics").json()
        assert data["backend"]["status"] == "ok"

    def test_provider_error_when_no_enabled_provider(self, client: TestClient) -> None:
        # Fresh state has no provider_configs with enabled=True.
        state = get_state()
        state.config.provider_configs = []
        data = client.get("/api/diagnostics").json()
        assert data["provider"]["status"] == "error"
        assert data["provider"]["fixUrl"] == "/settings"

    def test_provider_ok_when_one_provider_enabled(self, client: TestClient) -> None:
        state = get_state()
        state.config.provider_configs = [
            ProviderConfig(provider="openai", model="gpt-4o", enabled=True),
        ]
        data = client.get("/api/diagnostics").json()
        assert data["provider"]["status"] == "ok"
        assert data["provider"]["fixUrl"] is None

    def test_judge_warn_when_not_configured(self, client: TestClient) -> None:
        state = get_state()
        state.config.judge_chat_provider_id = None
        data = client.get("/api/diagnostics").json()
        assert data["judge"]["status"] == "warn"
        assert data["judge"]["fixUrl"] == "/settings"

    def test_judge_ok_when_configured(self, client: TestClient) -> None:
        state = get_state()
        state.config.judge_chat_provider_id = "cp-123"
        data = client.get("/api/diagnostics").json()
        assert data["judge"]["status"] == "ok"

    def test_storage_ok_on_healthy_telemetry_store(self, client: TestClient) -> None:
        data = client.get("/api/diagnostics").json()
        assert data["storage"]["status"] == "ok"

    def test_storage_error_when_telemetry_raises(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = get_state()

        def _boom() -> int:
            raise RuntimeError("disk full")

        monkeypatch.setattr(state.telemetry, "count_runs", _boom)
        data = client.get("/api/diagnostics").json()
        assert data["storage"]["status"] == "error"
        assert "RuntimeError" in data["storage"]["message"]
        assert data["storage"]["fixUrl"] == "/learn/troubleshoot"
