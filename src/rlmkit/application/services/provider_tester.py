"""Shared LLM Provider connection-test logic.

Single source of truth for probing an `LLMProviderConfig` for connectivity.
Both the manual-test route (`POST /api/llm-providers/{id}/test`) and the
background connection-test thread call :func:`test_provider`.

Timeout contract (critical)
---------------------------
:func:`test_provider` MUST return within approximately ``timeout_s`` seconds
regardless of network behavior.  Python cannot cancel running threads from
outside, so callers rely on this guarantee: if the function ever hangs,
worker threads in the caller's pool leak forever.  Enforcement inside this
function:

* ``litellm.completion(timeout=timeout_s, ...)`` — bounds the HTTP call.
* Explicit socket timeout on the global default socket — bounds the TCP
  connect/read so an accepts-but-never-responds server does not stall.
* No retries.

Any violation of this contract is a bug in this module, not in the caller.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from rlmkit.server.models import LLMProviderConfig
from rlmkit.ui.data.providers_catalog import PROVIDERS_BY_KEY
from rlmkit.ui.services.secret_store import FileSecretStore, KeyringSecretStore

logger = logging.getLogger("rlmkit.application.provider_tester")

# Literal statuses — matches LLMProviderConfig.status vocabulary.
ProviderStatus = Literal["connected", "offline", "error"]

# Max characters of upstream error text we copy into `error_message`.
# Shorter than before (300 → 200) — frontend tooltip only needs the gist.
_ERROR_MESSAGE_MAX_LEN = 200

# Hard cap on test duration: timeout_s + this small overhead.  Used only as a
# diagnostic log threshold; the real bound is enforced by litellm's timeout
# kwarg plus the socket-level timeout set below.
_TIMEOUT_OVERHEAD_S = 2.0


@dataclass(frozen=True)
class ProviderTestResult:
    """Outcome of a single connection probe.

    Fields
    ------
    status:
        ``"connected"`` — probe reached the server and got a valid response.
        ``"offline"``   — reachable or not, but the probe failed gracefully
        (timeout, 4xx/5xx, auth error, no-choices response).  Normal failure.
        ``"error"``     — unhandled exception during probe.  Distinct from
        ``offline`` so the UI can surface a different affordance.
    tested_at:
        UTC timestamp of when the probe completed.
    latency_ms:
        Round-trip milliseconds for a successful probe.  ``None`` when the
        probe never reached the server (timeout, DNS failure, connection
        refused) or when the test errored out.
    error_message:
        Sanitized upstream error text, truncated to ~200 chars.  Never
        contains API keys, bearer tokens, or raw response bodies.
    """

    status: ProviderStatus
    tested_at: datetime
    latency_ms: int | None
    error_message: str | None


def _get_api_key(llm_provider_id: str, backend: str) -> str | None:
    """Retrieve the API key (instance-scoped first, then env var)."""
    key_name = f"llm_provider:{llm_provider_id}"
    store: KeyringSecretStore | FileSecretStore = (
        KeyringSecretStore() if KeyringSecretStore.is_available() else FileSecretStore()
    )
    raw_key: str | None = store.get(key_name)
    if raw_key:
        return raw_key
    entry = PROVIDERS_BY_KEY.get(backend)
    if entry and entry.env_var:
        return os.environ.get(entry.env_var)
    return None


def _sanitize_error_message(exc: BaseException) -> str:
    """Extract a short, safe description of an upstream failure.

    Strips the litellm "ExceptionName - " prefix (noise for the UI) and
    truncates to ``_ERROR_MESSAGE_MAX_LEN``.  Callers are responsible for
    never logging the raw exception object; use this helper instead.
    """
    msg = str(exc)
    if " - " in msg:
        msg = msg.split(" - ", 1)[1]
    # Hard cap — upstream may echo request bodies into error strings.
    return msg[:_ERROR_MESSAGE_MAX_LEN]


def _build_probe_params(
    provider: LLMProviderConfig,
    timeout_s: float,
) -> dict[str, object]:
    """Build the kwargs passed to ``litellm.completion``.

    Mirrors the logic previously inline in ``POST /api/llm-providers/{id}/test``.
    """
    # Local import so importing this module does not force litellm load
    # (keeps unit tests that don't probe fast and avoids circular imports).
    from rlmkit.server.routes.providers import _litellm_model_name

    litellm_model = _litellm_model_name(provider.backend, provider.model)
    params: dict[str, object] = {
        "model": litellm_model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5,
        "timeout": timeout_s,
        "drop_params": True,
        # Explicit: no retries.  Retries multiply the effective timeout and
        # can make the function overshoot its contract.
        "num_retries": 0,
    }

    entry = PROVIDERS_BY_KEY.get(provider.backend)
    if entry and entry.requires_api_key:
        api_key = _get_api_key(provider.id, provider.backend)
        if api_key:
            params["api_key"] = api_key

    endpoint = provider.endpoint
    if not endpoint and entry and entry.default_endpoint:
        endpoint = entry.default_endpoint
    if endpoint:
        params["api_base"] = endpoint

    return params


def test_provider(
    provider: LLMProviderConfig,
    timeout_s: float = 10.0,  # noqa: PT028  # not a pytest function — name is a domain verb
) -> ProviderTestResult:
    """Probe an LLM Provider for connectivity.

    Never raises.  All failure modes (timeout, auth, network, unknown) are
    collapsed into a :class:`ProviderTestResult`.  See module docstring for
    the timeout contract — violating it leaks threads.

    Parameters
    ----------
    provider:
        The provider config to probe.  Read fields: ``id``, ``backend``,
        ``model``, ``endpoint``.  API keys resolved via the usual precedence
        (instance-scoped secret → backend env var).
    timeout_s:
        Upper bound on probe duration.  Passed to ``litellm.completion`` and
        enforced at the socket level.  The function returns within
        ``timeout_s + ~2s`` on a well-behaved client library.

    Returns
    -------
    ProviderTestResult
        Always; never raises.
    """
    logger.debug("probe attempt provider_id=%s timeout_s=%s", provider.id, timeout_s)
    start = time.monotonic()

    # Socket-level timeout — belt-and-braces with litellm's own timeout.
    # Set locally so we do not disturb unrelated code paths, then restore.
    prior_socket_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_s)
    try:
        try:
            import litellm
        except ImportError as exc:
            # Should never happen in a configured RLMKit install, but do not
            # let it crash the caller.
            tested_at = datetime.now(timezone.utc)
            logger.error("litellm import failed: %s", exc)
            return ProviderTestResult(
                status="error",
                tested_at=tested_at,
                latency_ms=None,
                error_message="litellm not installed",
            )

        params = _build_probe_params(provider, timeout_s)
        try:
            response = litellm.completion(**params)
        except Exception as exc:  # noqa: BLE001 — any litellm failure = offline probe result
            tested_at = datetime.now(timezone.utc)
            elapsed = time.monotonic() - start
            msg = _sanitize_error_message(exc)
            logger.debug(
                "probe result provider_id=%s status=offline latency_ms=None error=%s elapsed=%.2fs",
                provider.id,
                msg,
                elapsed,
            )
            if elapsed > timeout_s + _TIMEOUT_OVERHEAD_S:
                logger.warning(
                    "test_provider overshot timeout_s=%.1fs by %.1fs — "
                    "contract violation, check timeout enforcement in the probe",
                    timeout_s,
                    elapsed - timeout_s,
                )
            return ProviderTestResult(
                status="offline",
                tested_at=tested_at,
                latency_ms=None,
                error_message=msg,
            )

        # Success path
        tested_at = datetime.now(timezone.utc)
        latency_ms = int((time.monotonic() - start) * 1000)
        if not getattr(response, "choices", None):
            logger.debug(
                "probe result provider_id=%s status=offline (empty choices)",
                provider.id,
            )
            return ProviderTestResult(
                status="offline",
                tested_at=tested_at,
                latency_ms=latency_ms,
                error_message="No response from model",
            )
        logger.debug(
            "probe result provider_id=%s status=connected latency_ms=%d",
            provider.id,
            latency_ms,
        )
        return ProviderTestResult(
            status="connected",
            tested_at=tested_at,
            latency_ms=latency_ms,
            error_message=None,
        )
    except Exception as exc:  # noqa: BLE001 — never raise from this function
        # Truly unexpected (non-litellm) exception.  Distinct "error" status.
        tested_at = datetime.now(timezone.utc)
        logger.exception(
            "unhandled error probing provider_id=%s — classified as error status",
            provider.id,
        )
        return ProviderTestResult(
            status="error",
            tested_at=tested_at,
            latency_ms=None,
            error_message=_sanitize_error_message(exc),
        )
    finally:
        socket.setdefaulttimeout(prior_socket_timeout)


# Pytest auto-collects any callable whose name matches ``python_functions``
# ("test_*") from the test-module namespace.  Because :func:`test_provider`
# is a domain verb that tests import directly, mark it as non-test so pytest
# skips it instead of treating it as a test case missing fixtures.
test_provider.__test__ = False  # type: ignore[attr-defined]
