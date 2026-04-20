"""Replay endpoint — Learn tab Concepts §C trace-backed deep-link.

``GET /api/replays/{execution_id}`` loads a canonical
:class:`TraceResponse` (via the shared ``load_canonical_trace``
helper from the traces route) and converts it to a
:class:`LearnReplay` through the pure service function in
:mod:`rlmkit.application.services.trace_to_replay`.

Returns a bare ``LearnReplay`` (no wrapper object) to mirror the
other single-resource endpoints on the Learn surface
(``/api/traces/{id}``, ``/api/docs/{slug}``). See
``doc_internal/specs/learn-tab/NEXT.md`` §3a.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from rlmkit.application.services.trace_to_replay import trace_to_replay
from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import LearnReplay
from rlmkit.server.routes.traces import load_canonical_trace

router = APIRouter()


@router.get("/api/replays/{execution_id}")
async def get_replay(
    execution_id: str,
    state: AppState = Depends(get_state),  # noqa: B008
) -> LearnReplay:
    """Return the LearnReplay for a given execution id.

    Raises 404 with the standard error envelope when the execution
    is unknown (not in the live ``state.executions`` dict and not
    in the telemetry store). The route is a thin adapter: all
    conversion logic lives in the service function.
    """
    trace = load_canonical_trace(execution_id, state)
    if trace is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    return trace_to_replay(trace)
