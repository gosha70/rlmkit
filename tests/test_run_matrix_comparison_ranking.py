"""Phase 5 — AC-6 tests for the three new backend ranking metrics."""

from __future__ import annotations

from rlmstudio.application.dto import RunResultDTO
from rlmstudio.application.use_cases.run_matrix_comparison import (
    MatrixSlotResultDTO,
    RunMatrixComparisonUseCase,
)


def _slot(
    *,
    slot_id: str,
    success: bool = True,
    ttft_ms: int | None = None,
    decode_ms: int = 0,
    completion_tokens: int = 0,
    prompt_tokens: int = 0,
    cached_tokens: int = 0,
) -> MatrixSlotResultDTO:
    trace = []
    if success:
        # Single assistant entry carrying the raw-DTO telemetry keys.
        trace.append(
            {
                "step": 1,
                "role": "assistant",
                "ttft_ms": ttft_ms,
                "decode_ms": decode_ms,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
                "cache_write_tokens": 0,
                "elapsed_seconds": 1.0,
                "model": "test",
            }
        )
    result = RunResultDTO(
        answer="ok" if success else "",
        mode_used="direct",
        success=success,
        error=None if success else "boom",
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_cost=0.0,
        elapsed_time=1.0,
        trace=trace,
    )
    return MatrixSlotResultDTO(
        slot_id=slot_id,
        label=slot_id,
        mode="direct",
        provider="test",
        model="test",
        result=result,
    )


class TestTTFTRanking:
    def test_ascending_ttft(self):
        slots = [
            _slot(slot_id="a", ttft_ms=300),
            _slot(slot_id="b", ttft_ms=100),
            _slot(slot_id="c", ttft_ms=200),
        ]
        ranking = RunMatrixComparisonUseCase._rank(slots, "ttft")
        assert ranking == [1, 2, 0]  # 100, 200, 300

    def test_slots_without_ttft_sort_last_among_successes(self):
        slots = [
            _slot(slot_id="no-ttft", ttft_ms=None),
            _slot(slot_id="fast", ttft_ms=50),
        ]
        ranking = RunMatrixComparisonUseCase._rank(slots, "ttft")
        assert ranking == [1, 0]

    def test_failed_slots_always_last(self):
        slots = [
            _slot(slot_id="failed", success=False),
            _slot(slot_id="fast", ttft_ms=50),
        ]
        ranking = RunMatrixComparisonUseCase._rank(slots, "ttft")
        assert ranking == [1, 0]


class TestDecodeTokensPerSecRanking:
    def test_descending_decode_tps(self):
        # slot a: 100 tokens / (100 ms = 0.1 s) = 1000 tok/s
        # slot b: 100 tokens / (1000 ms = 1.0 s) = 100 tok/s
        slots = [
            _slot(slot_id="a", decode_ms=100, completion_tokens=100),
            _slot(slot_id="b", decode_ms=1000, completion_tokens=100),
        ]
        ranking = RunMatrixComparisonUseCase._rank(slots, "decode_tokens_per_sec")
        assert ranking == [0, 1]  # fast first

    def test_zero_decode_sorts_last(self):
        slots = [
            _slot(slot_id="no-decode", decode_ms=0, completion_tokens=50),
            _slot(slot_id="fast", decode_ms=100, completion_tokens=100),
        ]
        ranking = RunMatrixComparisonUseCase._rank(slots, "decode_tokens_per_sec")
        assert ranking == [1, 0]


class TestCacheHitRateRanking:
    def test_descending_cache_hit_rate(self):
        slots = [
            _slot(slot_id="low", prompt_tokens=100, cached_tokens=10),
            _slot(slot_id="high", prompt_tokens=100, cached_tokens=80),
        ]
        ranking = RunMatrixComparisonUseCase._rank(slots, "cache_hit_rate")
        assert ranking == [1, 0]

    def test_no_prompt_tokens_sorts_last(self):
        slots = [
            _slot(slot_id="empty", prompt_tokens=0, cached_tokens=0),
            _slot(slot_id="real", prompt_tokens=100, cached_tokens=50),
        ]
        ranking = RunMatrixComparisonUseCase._rank(slots, "cache_hit_rate")
        assert ranking == [1, 0]
