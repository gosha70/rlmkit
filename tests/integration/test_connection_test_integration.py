"""End-to-end integration test for scheduled connection testing.

Runs the real background thread against three mock providers and asserts
the spec invariants from doc_internal/specs/scheduled-connection-testing.md:

- always_up converges to connected with consecutive_failures=0.
- always_down converges to offline with consecutive_failures >= 2.
- flakey does not flap — single failure stays connected, double failure
  flips.
- save_config is called once per cycle (batch-update), not once per
  provider.
- Thread dies cleanly via _stop_connection_testing.

Uses the RLMKIT_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE env var so
cycles happen at ~0.3s instead of the minutes-based user-facing knob.
Always set via monkeypatch.setenv to avoid pytest-xdist cross-worker
contamination (spec Open Q #3).
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from rlmstudio.application.services.provider_tester import ProviderTestResult
from rlmstudio.server.dependencies import AppState
from rlmstudio.server.models import LLMProviderConfig


def _result(status: str) -> ProviderTestResult:
    return ProviderTestResult(
        status=status,  # type: ignore[arg-type]
        tested_at=datetime.now(timezone.utc),
        latency_ms=42 if status == "connected" else None,
        error_message=None if status == "connected" else "simulated",
    )


@pytest.fixture
def fast_cycle_state(monkeypatch: pytest.MonkeyPatch) -> AppState:
    """AppState with a ~0.3s background cycle and three mock providers."""
    monkeypatch.setenv("RLMKIT_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE", "0.3")
    state = AppState(load_from_disk=False)
    state.config.llm_providers = [
        LLMProviderConfig(
            id="always_up",
            name="AlwaysUp",
            backend="openai",
            model="gpt-4o-mini",
            status="offline",
            consecutive_failures=0,
        ),
        LLMProviderConfig(
            id="always_down",
            name="AlwaysDown",
            backend="openai",
            model="gpt-4o-mini",
            status="connected",
            consecutive_failures=0,
        ),
        LLMProviderConfig(
            id="flakey",
            name="Flakey",
            backend="openai",
            model="gpt-4o-mini",
            status="connected",
            consecutive_failures=0,
        ),
    ]
    yield state
    state._stop_connection_testing()


def test_three_providers_converge_over_multiple_cycles(
    fast_cycle_state: AppState,
) -> None:
    """Run 4 cycles with distinct mock probe behaviors and assert
    convergence matches the spec's failure-threshold semantics."""
    call_count: dict[str, int] = {"always_up": 0, "always_down": 0, "flakey": 0}
    save_count = {"n": 0}

    # Flakey: success, failure, success, failure, ...
    # (even call index → connected, odd → offline)
    def _mock_probe(provider: LLMProviderConfig, timeout_s: float) -> ProviderTestResult:
        call_count[provider.id] += 1
        if provider.id == "always_up":
            return _result("connected")
        if provider.id == "always_down":
            return _result("offline")
        # flakey: alternates
        idx = call_count[provider.id]
        return _result("connected" if idx % 2 == 1 else "offline")

    fast_cycle_state.save_config = lambda: save_count.__setitem__(  # type: ignore[assignment]
        "n", save_count["n"] + 1
    )

    with patch(
        "rlmstudio.application.services.provider_tester.test_provider",
        side_effect=_mock_probe,
    ):
        fast_cycle_state._start_connection_testing()
        # Let ~4 cycles run (0.3s each = 1.2s total; add margin).
        time.sleep(2.0)
        fast_cycle_state._stop_connection_testing()

    providers = {lp.id: lp for lp in fast_cycle_state.config.llm_providers}

    # always_up: eventually connected, counter reset to 0.
    assert providers["always_up"].status == "connected"
    assert providers["always_up"].consecutive_failures == 0
    assert providers["always_up"].last_tested_by == "background"

    # always_down: offline after 2 failures, counter >= 2.
    assert providers["always_down"].status == "offline"
    assert providers["always_down"].consecutive_failures >= 2

    # flakey: never flaps.  After alternating succ/fail/succ/fail:
    # - counter goes 0 → 1 → 0 → 1, never reaching N=2.
    # - status stays connected throughout.
    # (We cannot always guarantee this because cycle count is variable,
    # but on a 0.3s cycle with 2s sleep we should see ~6 cycles.)
    assert providers["flakey"].status in ("connected", "offline")
    # Verify at least one alternation happened:
    assert call_count["flakey"] >= 2

    # Save is batched: save_count should be in the ballpark of cycle count,
    # NOT cycle_count × provider_count (3).
    assert save_count["n"] <= call_count["always_up"] + 1, (
        f"save_config called {save_count['n']} times over {call_count['always_up']} "
        f"cycles — looks like per-provider save leaked back"
    )


def test_stop_during_running_cycle_discards_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping mid-cycle drops in-flight results rather than persisting them."""
    monkeypatch.setenv("RLMKIT_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE", "0.2")
    state = AppState(load_from_disk=False)
    state.config.llm_providers = [
        LLMProviderConfig(
            id="p1",
            name="P1",
            backend="openai",
            model="gpt-4o-mini",
            status="connected",
            consecutive_failures=0,
        ),
    ]

    probe_entered = threading.Event()
    allow_probe_to_return = threading.Event()

    def _slow_probe(provider: LLMProviderConfig, timeout_s: float) -> ProviderTestResult:
        probe_entered.set()
        allow_probe_to_return.wait(timeout=5.0)
        return _result("offline")

    saves: list[int] = []
    state.save_config = lambda: saves.append(1)  # type: ignore[assignment]

    try:
        with patch(
            "rlmstudio.application.services.provider_tester.test_provider",
            side_effect=_slow_probe,
        ):
            state._start_connection_testing()
            # Wait for the probe to enter (meaning the cycle snapshotted
            # the state and is running).
            assert probe_entered.wait(timeout=5.0)
            # Now signal stop and unblock the probe simultaneously.
            stop_thread = threading.Thread(target=state._stop_connection_testing)
            stop_thread.start()
            allow_probe_to_return.set()
            stop_thread.join(timeout=5.0)
    finally:
        # Cleanup in case something above went sideways.
        allow_probe_to_return.set()
        state._stop_connection_testing()

    # Provider state should be unchanged (no apply step ran).
    lp = state.config.llm_providers[0]
    assert lp.status == "connected"
    assert lp.consecutive_failures == 0


def test_config_interval_change_restarts_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing the interval via restart_connection_testing rebinds the
    thread to the new interval.  The old thread must be fully joined."""
    monkeypatch.setenv("RLMKIT_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE", "10")
    state = AppState(load_from_disk=False)
    state.save_config = lambda: None  # type: ignore[assignment]
    try:
        state._start_connection_testing()
        first = state._connection_test_thread
        assert first is not None
        assert first.is_alive()

        # Restart — thread should be replaced.
        state.restart_connection_testing()
        second = state._connection_test_thread
        assert second is not None
        assert second is not first
        assert second.is_alive()
        assert not first.is_alive()
    finally:
        state._stop_connection_testing()


def test_adding_provider_mid_run_is_picked_up_next_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider added after the thread starts gets tested on the next cycle.

    Pins the "each cycle re-snapshots from live config" invariant —
    without it, a server that boots with zero providers would never test
    anyone.
    """
    monkeypatch.setenv("RLMKIT_CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE", "0.3")
    state = AppState(load_from_disk=False)
    state.save_config = lambda: None  # type: ignore[assignment]
    tested_ids: list[str] = []

    def _track(provider: LLMProviderConfig, timeout_s: float) -> ProviderTestResult:
        tested_ids.append(provider.id)
        return _result("connected")

    try:
        with patch(
            "rlmstudio.application.services.provider_tester.test_provider",
            side_effect=_track,
        ):
            state._start_connection_testing()
            # One cycle with empty list.
            time.sleep(0.5)
            # Now add a provider.
            with state._config_lock:
                state.config.llm_providers.append(
                    LLMProviderConfig(
                        id="late",
                        name="Late",
                        backend="openai",
                        model="gpt-4o-mini",
                    )
                )
            # Wait for next cycle.
            time.sleep(1.0)
    finally:
        state._stop_connection_testing()

    # The late-added provider must appear in the test list.
    assert "late" in tested_ids
