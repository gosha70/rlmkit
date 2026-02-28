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
    state: AppState = Depends(get_state),  # noqa: B008
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
    by_provider: dict[str, dict] = {}
    by_chat_provider: dict[str, dict] = {}
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
        provider = msg.get("provider", "unknown")

        total_tokens += tokens
        total_cost += cost
        total_latency += latency

        if mode not in by_mode:
            by_mode[mode] = {
                "queries": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "latencies": [],
            }
        by_mode[mode]["queries"] += 1
        by_mode[mode]["total_tokens"] += tokens
        by_mode[mode]["total_cost_usd"] += cost
        by_mode[mode]["latencies"].append(latency)

        if provider not in by_provider:
            by_provider[provider] = {
                "queries": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "latencies": [],
            }
        by_provider[provider]["queries"] += 1
        by_provider[provider]["total_tokens"] += tokens
        by_provider[provider]["total_cost_usd"] += cost
        by_provider[provider]["latencies"].append(latency)

        # Aggregate by Chat Provider (if present)
        cp_name = msg.get("chat_provider_name")
        if cp_name:
            if cp_name not in by_chat_provider:
                by_chat_provider[cp_name] = {
                    "queries": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "latencies": [],
                }
            by_chat_provider[cp_name]["queries"] += 1
            by_chat_provider[cp_name]["total_tokens"] += tokens
            by_chat_provider[cp_name]["total_cost_usd"] += cost
            by_chat_provider[cp_name]["latencies"].append(latency)

        timeline.append(
            TimelineEntry(
                timestamp=datetime.fromisoformat(msg["timestamp"]),
                tokens=tokens,
                cost_usd=cost,
                latency_seconds=latency,
                mode=mode,
                provider=provider,
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

    provider_summaries = {}
    for p, d in by_provider.items():
        lats = d.pop("latencies")
        avg_lat = sum(lats) / len(lats) if lats else 0.0
        provider_summaries[p] = ProviderSummary(
            queries=d["queries"],
            total_tokens=d["total_tokens"],
            total_cost_usd=d["total_cost_usd"],
            avg_latency_seconds=round(avg_lat, 2),
        )

    chat_provider_summaries = {}
    for cp_key, cpd in by_chat_provider.items():
        lats = cpd.pop("latencies")
        avg_lat = sum(lats) / len(lats) if lats else 0.0
        chat_provider_summaries[cp_key] = ProviderSummary(
            queries=cpd["queries"],
            total_tokens=cpd["total_tokens"],
            total_cost_usd=cpd["total_cost_usd"],
            avg_latency_seconds=round(avg_lat, 2),
        )

    # Token savings: compare RLM tokens vs Direct tokens (lower is better)
    rlm_tokens = mode_summaries.get("rlm", ModeSummary()).total_tokens
    direct_tokens = mode_summaries.get("direct", ModeSummary()).total_tokens
    if direct_tokens > 0 and rlm_tokens > 0:
        savings = round((1 - rlm_tokens / direct_tokens) * 100, 1)
    else:
        savings = 0.0

    return MetricsResponse(
        session_id=session_id,
        summary=MetricsSummary(
            total_queries=query_count,
            total_tokens=total_tokens,
            total_cost_usd=round(total_cost, 4),
            avg_latency_seconds=round(avg_latency, 2),
            avg_token_savings_percent=savings,
        ),
        by_mode=mode_summaries,
        by_provider=provider_summaries,
        by_chat_provider=chat_provider_summaries,
        timeline=timeline,
    )
