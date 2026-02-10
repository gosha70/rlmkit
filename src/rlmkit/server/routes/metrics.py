"""Metrics endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from rlmkit.server.dependencies import AppState, get_state
from rlmkit.server.models import (
    MetricsResponse,
    MetricsSummary,
    ModeSummary,
    ProviderSummary,
    TimelineEntry,
)

router = APIRouter()


@router.get("/api/metrics/{session_id}")
async def get_metrics(
    session_id: str,
    state: AppState = Depends(get_state),
) -> MetricsResponse:
    """Get aggregated metrics for a session."""
    session = state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    total_tokens = 0
    total_cost = 0.0
    total_latency = 0.0
    query_count = 0
    by_mode: dict[str, dict] = {}
    timeline = []

    for msg in session.messages:
        if msg["role"] != "assistant":
            continue
        m = msg.get("metrics")
        if not m:
            continue

        query_count += 1
        tokens = m.get("total_tokens", 0)
        cost = m.get("cost_usd", 0.0)
        latency = m.get("elapsed_seconds", 0.0)
        mode = msg.get("mode_used", "unknown")

        total_tokens += tokens
        total_cost += cost
        total_latency += latency

        if mode not in by_mode:
            by_mode[mode] = {"queries": 0, "total_tokens": 0, "total_cost_usd": 0.0, "latencies": []}
        by_mode[mode]["queries"] += 1
        by_mode[mode]["total_tokens"] += tokens
        by_mode[mode]["total_cost_usd"] += cost
        by_mode[mode]["latencies"].append(latency)

        timeline.append(
            TimelineEntry(
                timestamp=datetime.fromisoformat(msg["timestamp"]),
                tokens=tokens,
                cost_usd=cost,
                latency_seconds=latency,
                mode=mode,
            )
        )

    avg_latency = total_latency / query_count if query_count else 0.0

    mode_summaries = {}
    for mode, data in by_mode.items():
        lats = data.pop("latencies")
        avg_lat = sum(lats) / len(lats) if lats else 0.0
        mode_summaries[mode] = ModeSummary(
            queries=data["queries"],
            total_tokens=data["total_tokens"],
            total_cost_usd=data["total_cost_usd"],
            avg_latency_seconds=round(avg_lat, 2),
        )

    return MetricsResponse(
        session_id=session_id,
        summary=MetricsSummary(
            total_queries=query_count,
            total_tokens=total_tokens,
            total_cost_usd=round(total_cost, 4),
            avg_latency_seconds=round(avg_latency, 2),
        ),
        by_mode=mode_summaries,
        timeline=timeline,
    )
