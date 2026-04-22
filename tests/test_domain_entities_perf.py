"""Phase 2 tests for performance-telemetry fields on domain entities.

Covers AC-4 (ExecutionTrace.cache_hit_rate + median_ttft_ms derived
properties) and AC-23 (TraceStep.from_dict classmethod).
"""

from __future__ import annotations

from rlmkit.domain.entities import ExecutionTrace, TraceStep


class TestTraceStepNewFields:
    def test_defaults_preserve_legacy_shape(self):
        step = TraceStep(index=0, action_type="inspect")
        assert step.prompt_tokens == 0
        assert step.completion_tokens == 0
        assert step.ttft_ms is None
        assert step.decode_ms == 0
        assert step.cached_tokens == 0
        assert step.cache_write_tokens == 0

    def test_to_dict_includes_new_fields(self):
        step = TraceStep(
            index=2,
            action_type="final",
            prompt_tokens=100,
            completion_tokens=25,
            ttft_ms=120,
            decode_ms=45,
            cached_tokens=80,
            cache_write_tokens=5,
        )
        d = step.to_dict()
        assert d["prompt_tokens"] == 100
        assert d["completion_tokens"] == 25
        assert d["ttft_ms"] == 120
        assert d["decode_ms"] == 45
        assert d["cached_tokens"] == 80
        assert d["cache_write_tokens"] == 5


class TestTraceStepFromDict:
    """AC-23 — from_dict classmethod."""

    def test_round_trip(self):
        original = TraceStep(
            index=1,
            action_type="subcall",
            code="print('x')",
            output="x",
            tokens_used=15,
            timestamp=42.0,
            recursion_depth=1,
            cost=0.01,
            duration=0.5,
            model="gpt-4o",
            prompt_tokens=10,
            completion_tokens=5,
            ttft_ms=50,
            decode_ms=20,
            cached_tokens=3,
            cache_write_tokens=0,
        )
        rehydrated = TraceStep.from_dict(original.to_dict())
        assert rehydrated == original

    def test_tolerates_missing_keys(self):
        step = TraceStep.from_dict({"index": 0, "action_type": "inspect"})
        assert step.index == 0
        assert step.action_type == "inspect"
        assert step.prompt_tokens == 0
        assert step.ttft_ms is None

    def test_tolerates_extra_keys(self):
        step = TraceStep.from_dict(
            {
                "index": 0,
                "action_type": "inspect",
                "unknown_field": "ignored",
                "random_nested": {"a": 1},
            }
        )
        assert step.index == 0

    def test_unknown_action_type_falls_back_to_inspect(self):
        step = TraceStep.from_dict({"index": 0, "action_type": "bogus_action"})
        assert step.action_type == "inspect"

    def test_legacy_dict_without_new_fields(self):
        # Simulate a pre-Phase-2 serialized trace.
        legacy = {
            "index": 0,
            "action_type": "final",
            "tokens_used": 50,
            "duration": 1.5,
            "model": "gpt-4o",
        }
        step = TraceStep.from_dict(legacy)
        assert step.tokens_used == 50
        assert step.prompt_tokens == 0
        assert step.completion_tokens == 0
        assert step.ttft_ms is None


class TestExecutionTraceDerivedProps:
    """AC-4 — cache_hit_rate + median_ttft_ms + token totals."""

    def test_total_prompt_and_completion_tokens(self):
        trace = ExecutionTrace(
            steps=[
                TraceStep(index=0, action_type="inspect", prompt_tokens=100, completion_tokens=10),
                TraceStep(index=1, action_type="final", prompt_tokens=200, completion_tokens=20),
            ]
        )
        assert trace.total_prompt_tokens == 300
        assert trace.total_completion_tokens == 30

    def test_cache_hit_rate_computes_sum_over_sum(self):
        trace = ExecutionTrace(
            steps=[
                TraceStep(index=0, action_type="inspect", prompt_tokens=100, cached_tokens=40),
                TraceStep(index=1, action_type="final", prompt_tokens=100, cached_tokens=60),
            ]
        )
        # 100 cached / 200 prompt = 0.5
        assert trace.cache_hit_rate == 0.5

    def test_cache_hit_rate_zero_when_no_prompt_tokens(self):
        trace = ExecutionTrace(steps=[TraceStep(index=0, action_type="inspect")])
        assert trace.cache_hit_rate == 0.0

    def test_cache_hit_rate_capped_at_one(self):
        # Defensive: provider reports cached > prompt in some edge cases.
        trace = ExecutionTrace(
            steps=[
                TraceStep(index=0, action_type="inspect", prompt_tokens=100, cached_tokens=150),
            ]
        )
        assert trace.cache_hit_rate == 1.0

    def test_median_ttft_ms_odd_count(self):
        trace = ExecutionTrace(
            steps=[
                TraceStep(index=0, action_type="inspect", ttft_ms=50),
                TraceStep(index=1, action_type="inspect", ttft_ms=100),
                TraceStep(index=2, action_type="final", ttft_ms=200),
            ]
        )
        assert trace.median_ttft_ms == 100

    def test_median_ttft_ms_even_count_upper(self):
        # Spec's implementation picks vals[len//2] — for len=2 that's index 1
        # (the upper of the two middles). Acceptable and documented.
        trace = ExecutionTrace(
            steps=[
                TraceStep(index=0, action_type="inspect", ttft_ms=50),
                TraceStep(index=1, action_type="final", ttft_ms=100),
            ]
        )
        assert trace.median_ttft_ms == 100

    def test_median_ttft_ms_none_when_no_values(self):
        trace = ExecutionTrace(
            steps=[TraceStep(index=0, action_type="inspect")]  # ttft_ms=None
        )
        assert trace.median_ttft_ms is None

    def test_median_ttft_ms_ignores_none_values(self):
        trace = ExecutionTrace(
            steps=[
                TraceStep(index=0, action_type="inspect", ttft_ms=None),
                TraceStep(index=1, action_type="final", ttft_ms=42),
            ]
        )
        assert trace.median_ttft_ms == 42

    def test_to_dict_exposes_all_derived_props(self):
        trace = ExecutionTrace(
            steps=[
                TraceStep(
                    index=0,
                    action_type="final",
                    prompt_tokens=80,
                    completion_tokens=20,
                    cached_tokens=40,
                    ttft_ms=100,
                )
            ]
        )
        d = trace.to_dict()
        assert d["total_prompt_tokens"] == 80
        assert d["total_completion_tokens"] == 20
        assert d["cache_hit_rate"] == 0.5
        assert d["median_ttft_ms"] == 100
