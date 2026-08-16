"""Unit tests for :mod:`rlmstudio.application.services.provider_tester`.

Per the spec (doc_internal/specs/scheduled-connection-testing.md §Prerequisite
refactor), these tests pin the timeout-enforcement contract that the
background connection-test thread relies on.  If the hang-timeout test ever
starts failing it means test_provider can block indefinitely — which would
leak worker threads in the caller's pool and stall the cycle loop.
"""

from __future__ import annotations

import socket
import threading
import time
from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest

from rlmstudio.application.services.provider_tester import (
    ProviderTestResult,
    test_provider,
)
from rlmstudio.server.models import LLMProviderConfig

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _make_provider(
    *,
    backend: str = "openai",
    model: str = "gpt-4o-mini",
    endpoint: str | None = None,
) -> LLMProviderConfig:
    return LLMProviderConfig(
        id="test-id",
        name="test-provider",
        backend=backend,
        model=model,
        endpoint=endpoint,
    )


class _FakeResponse:
    """Minimal stand-in for a litellm completion response."""

    def __init__(self, choices: list[Any]) -> None:
        self.choices = choices


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_test_provider_returns_result_on_success() -> None:
    """Happy path: litellm returns choices → status=connected."""
    provider = _make_provider()
    fake_response = _FakeResponse(choices=[object()])

    with patch("litellm.completion", return_value=fake_response):
        result = test_provider(provider, timeout_s=5.0)

    assert isinstance(result, ProviderTestResult)
    assert result.status == "connected"
    assert isinstance(result.tested_at, datetime)
    assert result.latency_ms is not None
    assert result.latency_ms >= 0
    assert result.error_message is None


def test_test_provider_returns_offline_on_empty_choices() -> None:
    """Empty choices is a graceful failure, not an error."""
    provider = _make_provider()
    fake_response = _FakeResponse(choices=[])

    with patch("litellm.completion", return_value=fake_response):
        result = test_provider(provider, timeout_s=5.0)

    assert result.status == "offline"
    assert result.error_message == "No response from model"


def test_test_provider_returns_offline_on_litellm_exception() -> None:
    """litellm raising → offline with sanitized error message."""
    provider = _make_provider()

    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("BadRequestError - invalid api key")

    with patch("litellm.completion", side_effect=_raise):
        result = test_provider(provider, timeout_s=5.0)

    assert result.status == "offline"
    assert result.latency_ms is None
    assert result.error_message is not None
    # litellm "ExceptionName - " prefix is stripped.
    assert "BadRequestError - " not in result.error_message
    assert "invalid api key" in result.error_message


def test_test_provider_never_raises_even_on_network_error() -> None:
    """Per contract, test_provider swallows all exceptions."""
    provider = _make_provider()

    with patch("litellm.completion", side_effect=ConnectionError("no route")):
        # Must not raise.
        result = test_provider(provider, timeout_s=5.0)

    assert result.status == "offline"
    assert result.error_message is not None


@pytest.mark.parametrize(
    ("fake_key", "pattern_label"),
    [
        ("sk-THISISSECRETTOKENABCDEF1234567890", "openai-style"),
        ("sk-ant-THISISSECRETTOKENABCDEF1234567890", "anthropic-style"),
        ("hf_THISISSECRETTOKENABCDEF1234567890", "hugging-face"),
        ("AIzaTHISISSECRETTOKENABCDEF1234567890", "google"),
    ],
)
def test_error_message_does_not_contain_api_key(
    fake_key: str,
    pattern_label: str,
) -> None:
    """Sanitized error messages must not leak API keys.

    The upstream exception string deliberately contains a fake key in a
    plausible echo pattern ("Invalid API key provided: sk-...").  The
    sanitizer must redact it.  Cap length alone is not sufficient — a
    short upstream error would otherwise leak the key unmodified.
    """
    provider = _make_provider()
    msg = f"AuthError - Invalid API key provided: {fake_key}"

    with patch("litellm.completion", side_effect=RuntimeError(msg)):
        result = test_provider(provider, timeout_s=5.0)

    assert result.error_message is not None
    # Strong assertion: the raw key MUST be absent.
    assert fake_key not in result.error_message, (
        f"sanitizer leaked a {pattern_label} API key into error_message"
    )
    # And the redaction marker should be present.
    assert "<redacted>" in result.error_message
    # Length cap still holds.
    assert len(result.error_message) <= 300


def test_error_message_redacts_bearer_token() -> None:
    """HTTP-layer errors sometimes echo the Authorization header verbatim."""
    provider = _make_provider()
    fake_token = "SECRETBEARER1234567890ABCDEFGHIJ"
    msg = f"httpx.ConnectError - Authorization: Bearer {fake_token}"

    with patch("litellm.completion", side_effect=RuntimeError(msg)):
        result = test_provider(provider, timeout_s=5.0)

    assert result.error_message is not None
    assert fake_token not in result.error_message
    assert "<redacted>" in result.error_message


def test_timeout_returns_offline_with_latency_none() -> None:
    """A litellm timeout exception produces offline with latency_ms=None."""
    provider = _make_provider()

    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise TimeoutError("Timeout - upstream did not respond")

    with patch("litellm.completion", side_effect=_raise):
        result = test_provider(provider, timeout_s=5.0)

    assert result.status == "offline"
    assert result.latency_ms is None


def test_test_provider_returns_within_timeout_even_when_server_hangs() -> None:
    """Pins the timeout-enforcement contract from the spec.

    Stand up a TCP listener that accepts connections and never responds.
    Call test_provider with timeout_s=2.0.  The function MUST return within
    4 seconds (timeout_s + overhead) regardless of server behavior.  If it
    hangs, worker threads in the caller's thread pool would leak forever.

    NOTE: this test is single-threaded by design.  Enforcement relies on
    litellm.completion(timeout=...) which is per-request and thread-safe.
    The commit 3.5 fix removed the previous process-global
    socket.setdefaulttimeout belt-and-braces because it races under
    ThreadPoolExecutor.
    """
    # Bind to an ephemeral port.  accept() runs in a daemon thread that
    # simply holds the socket open without writing any bytes.
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    port = listener.getsockname()[1]
    accepted_sockets: list[socket.socket] = []
    stop_event = threading.Event()

    def _accept_forever() -> None:
        listener.settimeout(0.5)
        while not stop_event.is_set():
            try:
                conn, _addr = listener.accept()
                accepted_sockets.append(conn)
            except TimeoutError:
                continue
            except OSError:
                break

    accept_thread = threading.Thread(target=_accept_forever, daemon=True)
    accept_thread.start()

    try:
        provider = _make_provider(
            backend="lmstudio",  # treated as local provider, no API key needed
            model="local-model",
            endpoint=f"http://127.0.0.1:{port}",
        )

        start = time.monotonic()
        # Use a short timeout for speed; any value < TCP_KEEPALIVE kernel
        # defaults would hang without the socket-level timeout we set.
        result = test_provider(provider, timeout_s=2.0)
        elapsed = time.monotonic() - start

        # Contract: return within timeout_s + small overhead.
        assert elapsed < 4.0, (
            f"test_provider took {elapsed:.2f}s with timeout_s=2.0 — "
            "timeout contract is not being enforced"
        )
        # An accepts-but-hangs server is an ordinary offline case.
        assert result.status == "offline"
    finally:
        stop_event.set()
        listener.close()
        accept_thread.join(timeout=2.0)
        for s in accepted_sockets:
            try:
                s.close()
            except OSError:
                pass


@pytest.mark.parametrize(
    ("exc_cls", "exc_args"),
    [
        (ValueError, ("simple error",)),
        (TimeoutError, ("timed out",)),
        (ConnectionError, ("refused",)),
    ],
)
def test_test_provider_classifies_litellm_exceptions_as_offline(
    exc_cls: type[Exception],
    exc_args: tuple[str, ...],
) -> None:
    """Any exception raised BY litellm.completion is an offline probe."""
    provider = _make_provider()
    with patch("litellm.completion", side_effect=exc_cls(*exc_args)):
        result = test_provider(provider, timeout_s=5.0)
    assert result.status == "offline"


def test_test_provider_does_not_mutate_process_global_socket_timeout() -> None:
    """Commit 3.5 removed the socket.setdefaulttimeout belt-and-braces
    (it races under ThreadPoolExecutor).  Pin the regression: probe calls
    must not touch the global socket timeout at all, so unrelated HTTP
    clients in the same process are unaffected."""
    original = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(None)
        provider = _make_provider()
        with patch("litellm.completion", return_value=_FakeResponse(choices=[object()])):
            test_provider(provider, timeout_s=2.0)
        assert socket.getdefaulttimeout() is None, (
            "test_provider is mutating the process-global socket timeout; "
            "this races under concurrent callers (ThreadPoolExecutor)"
        )
    finally:
        socket.setdefaulttimeout(original)
