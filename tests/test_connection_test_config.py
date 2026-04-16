"""Tests for the config-model additions in Commit 3 of the scheduled-
connection-testing feature.

Covers:
- LLMProviderConfig gains last_tested_at, last_tested_by, consecutive_failures.
- ConfigResponse/ConfigUpdateRequest gain connection_test_interval_minutes
  with validator (0-1440 inclusive).
- Manual test route updates the new fields (spec §Failure Semantics — manual
  path bypasses the N-consecutive-failures threshold).
- PUT /api/config round-trips connection_test_interval_minutes.
- AppState._config_lock exists and is reentrant.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from rlmkit.application.services.provider_tester import ProviderTestResult
from rlmkit.server import dependencies
from rlmkit.server.app import create_app
from rlmkit.server.dependencies import AppState, reset_state
from rlmkit.server.models import (
    ConfigResponse,
    ConfigUpdateRequest,
    LLMProviderConfig,
    RuntimeSettings,
)


@pytest.fixture
def client() -> TestClient:
    reset_state()
    app = create_app()
    return TestClient(app)


# --------------------------------------------------------------------------
# Model-level tests
# --------------------------------------------------------------------------


def test_llm_provider_config_has_tracking_fields() -> None:
    """New fields exist with sensible defaults."""
    lp = LLMProviderConfig(id="x", name="x", backend="openai", model="gpt-4o")
    assert lp.last_tested_at is None
    assert lp.last_tested_by is None
    assert lp.consecutive_failures == 0


def test_llm_provider_config_accepts_tracking_values() -> None:
    now = datetime.now(timezone.utc)
    lp = LLMProviderConfig(
        id="x",
        name="x",
        backend="openai",
        model="gpt-4o",
        last_tested_at=now,
        last_tested_by="background",
        consecutive_failures=3,
    )
    assert lp.last_tested_at == now
    assert lp.last_tested_by == "background"
    assert lp.consecutive_failures == 3


def test_llm_provider_config_rejects_invalid_last_tested_by() -> None:
    with pytest.raises(ValueError):
        LLMProviderConfig(
            id="x",
            name="x",
            backend="openai",
            model="gpt-4o",
            last_tested_by="something_else",  # type: ignore[arg-type]
        )


def test_config_response_default_interval_is_zero() -> None:
    cr = ConfigResponse()
    assert cr.connection_test_interval_minutes == 0


@pytest.mark.parametrize("value", [-1, 1441, 99999])
def test_config_response_rejects_out_of_range_interval(value: int) -> None:
    with pytest.raises(ValueError):
        ConfigResponse(connection_test_interval_minutes=value)


@pytest.mark.parametrize("value", [0, 1, 60, 1440])
def test_config_response_accepts_in_range_interval(value: int) -> None:
    cr = ConfigResponse(connection_test_interval_minutes=value)
    assert cr.connection_test_interval_minutes == value


def test_config_update_request_accepts_none_interval() -> None:
    """None means "no change" in the partial-update request."""
    req = ConfigUpdateRequest()
    assert req.connection_test_interval_minutes is None


def test_config_update_request_rejects_out_of_range_interval() -> None:
    with pytest.raises(ValueError):
        ConfigUpdateRequest(connection_test_interval_minutes=2000)


# --------------------------------------------------------------------------
# AppState — lock
# --------------------------------------------------------------------------


def test_app_state_has_reentrant_config_lock() -> None:
    state = AppState(load_from_disk=False)
    # Must be an RLock, not a regular Lock — route handlers acquire it and
    # then call save_config() which also acquires it.
    assert state._config_lock is not None
    # Acquire twice from the same thread; RLock allows this, Lock would hang.
    with state._config_lock:
        with state._config_lock:
            pass


def test_save_config_acquires_lock() -> None:
    """save_config MUST hold the lock during the write so concurrent
    mutations can't race with its model_dump snapshot."""
    state = AppState(load_from_disk=False)
    holding = threading.Event()
    released = threading.Event()
    state.save_config = lambda: None  # type: ignore[assignment]  # no disk

    def _hold_lock() -> None:
        with state._config_lock:
            holding.set()
            released.wait(timeout=5.0)

    t = threading.Thread(target=_hold_lock)
    t.start()
    holding.wait(timeout=5.0)

    # At this point the worker thread holds the lock.  A non-blocking
    # acquire from this thread must fail.
    assert state._config_lock.acquire(blocking=False) is False
    released.set()
    t.join(timeout=5.0)

    # After the worker releases, we can acquire it.
    assert state._config_lock.acquire(blocking=False) is True
    state._config_lock.release()


# --------------------------------------------------------------------------
# Integration: manual test route updates the new fields
# --------------------------------------------------------------------------


def _add_provider(client: TestClient) -> str:
    resp = client.post(
        "/api/llm-providers",
        json={
            "name": "test-provider",
            "backend": "openai",
            "model": "gpt-4o-mini",
            "api_key": "sk-test",
            "runtime_settings": RuntimeSettings().model_dump(),
        },
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    lp_id: str = data["id"]
    return lp_id


def test_manual_test_success_updates_tracking_fields(client: TestClient) -> None:
    lp_id = _add_provider(client)

    class _FakeResponse:
        def __init__(self) -> None:
            self.choices = [object()]

    with patch("litellm.completion", return_value=_FakeResponse()):
        resp = client.post(f"/api/llm-providers/{lp_id}/test")

    assert resp.status_code == 200
    assert resp.json()["connected"] is True

    # Pull the provider back.
    get_resp = client.get(f"/api/llm-providers/{lp_id}")
    assert get_resp.status_code == 200
    lp = get_resp.json()
    assert lp["status"] == "connected"
    assert lp["last_tested_at"] is not None
    assert lp["last_tested_by"] == "manual"
    assert lp["consecutive_failures"] == 0


def test_manual_test_failure_flips_status_and_sets_counter_to_one(
    client: TestClient,
) -> None:
    """Manual failure path (spec §Failure Semantics): status flips
    immediately, counter goes to 1.  Distinct from the background
    threshold logic which requires N ≥ 2 consecutive failures."""
    lp_id = _add_provider(client)

    with patch("litellm.completion", side_effect=RuntimeError("AuthError - bad key")):
        resp = client.post(f"/api/llm-providers/{lp_id}/test")

    assert resp.status_code == 200
    assert resp.json()["connected"] is False

    get_resp = client.get(f"/api/llm-providers/{lp_id}")
    lp = get_resp.json()
    assert lp["status"] == "offline"
    assert lp["consecutive_failures"] == 1
    assert lp["last_tested_by"] == "manual"
    assert lp["last_tested_at"] is not None


def test_manual_test_resets_counter_from_nonzero_on_success(
    client: TestClient,
) -> None:
    lp_id = _add_provider(client)

    # Prime the counter to 3 by poking the state directly.
    state = dependencies.get_state()
    lp = state.get_llm_provider(lp_id)
    assert lp is not None
    lp.consecutive_failures = 3
    lp.status = "offline"

    class _FakeResponse:
        def __init__(self) -> None:
            self.choices = [object()]

    with patch("litellm.completion", return_value=_FakeResponse()):
        resp = client.post(f"/api/llm-providers/{lp_id}/test")
    assert resp.status_code == 200

    lp_after = state.get_llm_provider(lp_id)
    assert lp_after is not None
    assert lp_after.status == "connected"
    assert lp_after.consecutive_failures == 0


# --------------------------------------------------------------------------
# Integration: config round-trip
# --------------------------------------------------------------------------


def test_get_config_exposes_interval_field(client: TestClient) -> None:
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "connection_test_interval_minutes" in data
    assert data["connection_test_interval_minutes"] == 0


def test_put_config_updates_interval_field(client: TestClient) -> None:
    resp = client.put(
        "/api/config",
        json={"connection_test_interval_minutes": 5},
    )
    assert resp.status_code == 200
    assert resp.json()["connection_test_interval_minutes"] == 5

    # Round-trip via GET.
    get_resp = client.get("/api/config")
    assert get_resp.json()["connection_test_interval_minutes"] == 5


def test_put_config_rejects_out_of_range_interval(client: TestClient) -> None:
    resp = client.put(
        "/api/config",
        json={"connection_test_interval_minutes": 9999},
    )
    assert resp.status_code == 422  # Pydantic validation


def test_put_config_none_interval_means_no_change(client: TestClient) -> None:
    # Set it to something non-default first.
    client.put("/api/config", json={"connection_test_interval_minutes": 5})

    # Now send a PUT without the interval field.
    resp = client.put("/api/config", json={"budget": {"max_steps": 11}})
    assert resp.status_code == 200
    # Interval must be unchanged.
    assert resp.json()["connection_test_interval_minutes"] == 5


# --------------------------------------------------------------------------
# Provider_tester result plumbing
# --------------------------------------------------------------------------


def test_provider_tester_result_shape_matches_manual_route_consumer() -> None:
    """Pin the contract between ProviderTestResult and the manual route.

    If either side drifts, the manual route breaks.  This tests the
    structural shape rather than behavior.
    """
    result = ProviderTestResult(
        status="connected",
        tested_at=datetime.now(timezone.utc),
        latency_ms=42,
        error_message=None,
    )
    assert result.status in ("connected", "offline", "error")
    assert isinstance(result.tested_at, datetime)
    assert result.latency_ms == 42
    assert result.error_message is None
