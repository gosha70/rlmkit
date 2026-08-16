"""AC-16 — CompareMatrixSlotResponse carries the six per-slot perf fields."""

from __future__ import annotations

from rlmstudio.application.dto import RunResultDTO
from rlmstudio.server.routes.compare_matrix import (
    CompareMatrixSlotResponse,
    _slot_perf_aggregates,
)


def test_slot_response_defaults_when_trace_is_empty():
    aggs = _slot_perf_aggregates(None)
    slot = CompareMatrixSlotResponse(
        slot_id="s",
        label="s",
        mode="direct",
        provider="p",
        model="m",
        chat_provider_id="cp",
        execution_id="e",
        success=False,
        answer="",
        **aggs,
    )
    assert slot.total_prompt_tokens == 0
    assert slot.total_completion_tokens == 0
    assert slot.total_cached_tokens == 0
    assert slot.total_decode_ms == 0
    assert slot.median_ttft_ms is None
    assert slot.cache_hit_rate == 0.0


def test_slot_aggregates_from_populated_trace():
    result = RunResultDTO(
        answer="ok",
        mode_used="direct",
        success=True,
        input_tokens=300,
        output_tokens=40,
        trace=[
            {
                "step": 1,
                "role": "assistant",
                "input_tokens": 150,
                "output_tokens": 20,
                "ttft_ms": 100,
                "decode_ms": 50,
                "cached_tokens": 90,
                "cache_write_tokens": 0,
                "elapsed_seconds": 0.2,
            },
            {
                "step": 2,
                "role": "assistant",
                "input_tokens": 150,
                "output_tokens": 20,
                "ttft_ms": 140,
                "decode_ms": 60,
                "cached_tokens": 90,
                "cache_write_tokens": 0,
                "elapsed_seconds": 0.25,
            },
        ],
    )
    aggs = _slot_perf_aggregates(result.trace)
    slot = CompareMatrixSlotResponse(
        slot_id="s",
        label="s",
        mode="direct",
        provider="p",
        model="m",
        chat_provider_id="cp",
        execution_id="e",
        success=True,
        answer="ok",
        input_tokens=300,
        output_tokens=40,
        **aggs,
    )
    assert slot.total_prompt_tokens == 300
    assert slot.total_completion_tokens == 40
    assert slot.total_cached_tokens == 180
    assert slot.total_decode_ms == 110
    # Median of [100, 140] — we take vals[len//2] = vals[1] = 140.
    assert slot.median_ttft_ms == 140
    # 180 / 300 = 0.6
    assert slot.cache_hit_rate == 0.6
