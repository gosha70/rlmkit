"""Diagnostics endpoint — powers the Learn tab persistent strip.

Returns four lightweight health checks (backend, provider, judge,
storage) so the client can render a status row on every Learn page.

Check semantics match the pitch's red-dot badge policy:
``error`` states are hard blockers (no enabled provider, storage
unwritable); ``warn`` is a soft issue (judge missing); ``ok`` passes.
Only ``error`` drives the sidebar badge — see ``pitch-learn-tab.md``
§Diagnostics red-dot badge.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import DiagnosticCheck, DiagnosticsResponse

router = APIRouter()


@router.get("/api/diagnostics")
async def get_diagnostics(
    state: AppState = Depends(get_state),  # noqa: B008
) -> DiagnosticsResponse:
    """Run the four Learn-tab diagnostic checks and return their status.

    Target: sub-500 ms. All checks read in-memory config or issue a
    single lightweight SQLite query; no network calls or LLM pings.
    """
    return DiagnosticsResponse(
        backend=_check_backend(),
        provider=_check_provider(state),
        judge=_check_judge(state),
        storage=_check_storage(state),
    )


def _check_backend() -> DiagnosticCheck:
    # If this code runs, the backend is up — the endpoint itself is the probe.
    return DiagnosticCheck(status="ok", message="Backend reachable")


def _check_provider(state: AppState) -> DiagnosticCheck:
    enabled = [pc for pc in state.config.provider_configs if pc.enabled]
    if enabled:
        return DiagnosticCheck(
            status="ok",
            message=f"{len(enabled)} enabled provider(s)",
        )
    return DiagnosticCheck(
        status="error",
        message="No enabled LLM provider",
        fixUrl="/settings",
    )


def _check_judge(state: AppState) -> DiagnosticCheck:
    if state.config.judge_chat_provider_id:
        return DiagnosticCheck(status="ok", message="Judge configured")
    # Judge missing is a soft warning, not an error — see pitch §Resolved
    # Decisions #6. It does not drive the red-dot sidebar badge.
    return DiagnosticCheck(
        status="warn",
        message="Judge not configured",
        fixUrl="/settings",
    )


def _check_storage(state: AppState) -> DiagnosticCheck:
    try:
        state.telemetry.count_runs()
    except Exception as exc:  # noqa: BLE001 — any failure means storage is unhealthy
        return DiagnosticCheck(
            status="error",
            message=f"Storage unavailable: {exc.__class__.__name__}",
            fixUrl="/learn/troubleshoot",
        )
    return DiagnosticCheck(status="ok", message="Storage writable")
