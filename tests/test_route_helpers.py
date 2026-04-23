"""AC-24, AC-26, AC-13b — route-layer _translate_raw_trace_entry + _materialize_trace.

Covers the raw-DTO to `TraceStep` translation (AC-26), the
`_materialize_trace` tolerant input handling (AC-24), and the RAG
two-step post-canonicalization shape (AC-13b).
"""

from __future__ import annotations

from rlmkit.server.routes._helpers import (
    _materialize_trace,
    _translate_raw_trace_entry,
)


def _assistant_entry(**overrides):
    base = {
        "step": 1,
        "role": "assistant",
        "content": "the answer",
        "input_tokens": 10,
        "output_tokens": 5,
        "elapsed_seconds": 0.5,
        "model": "gpt-4o",
        "ttft_ms": 120,
        "decode_ms": 45,
        "cached_tokens": 6,
        "cache_write_tokens": 1,
    }
    base.update(overrides)
    return base


def _execution_entry(**overrides):
    base = {
        "step": 1,
        "role": "execution",
        "content": "stdout: hi",
        "code": "print('hi')",
        "elapsed_seconds": 0.1,
    }
    base.update(overrides)
    return base


class TestTranslateRawTraceEntry:
    """AC-26 — raw-DTO → TraceStep translation."""

    def test_translate_assistant_entry(self):
        step = _translate_raw_trace_entry(_assistant_entry(), is_last=False, run_success=True)
        assert step.action_type == "inspect"
        assert step.raw_response == "the answer"
        assert step.output is None  # assistant content routes to raw_response
        assert step.prompt_tokens == 10
        assert step.completion_tokens == 5
        assert step.tokens_used == 15
        assert step.duration == 0.5
        assert step.model == "gpt-4o"
        assert step.ttft_ms == 120
        assert step.decode_ms == 45
        assert step.cached_tokens == 6
        assert step.cache_write_tokens == 1

    def test_translate_execution_entry(self):
        step = _translate_raw_trace_entry(_execution_entry(), is_last=False, run_success=True)
        assert step.action_type == "subcall"
        assert step.output == "stdout: hi"  # execution content routes to output
        assert step.raw_response is None
        assert step.code == "print('hi')"

    def test_translate_rag_retrieval_entry(self):
        entry = {"step": 0, "role": "rag_retrieval", "content": "chunk text"}
        step = _translate_raw_trace_entry(entry, is_last=False, run_success=True)
        # Unknown role → inspect by the canonicalizer's default.
        assert step.action_type == "inspect"
        # rag_retrieval routes content to output, same as execution.
        assert step.output == "chunk text"
        assert step.raw_response is None

    def test_translate_is_last_promotes_to_final_on_success(self):
        step = _translate_raw_trace_entry(_assistant_entry(), is_last=True, run_success=True)
        assert step.action_type == "final"

    def test_translate_failed_terminal_keeps_role_action_type(self):
        """Regression guard for v1.5 prose bug: failed terminal steps
        must NOT be promoted to `error` — the canonicalizer has no
        such branch."""
        step = _translate_raw_trace_entry(_assistant_entry(), is_last=True, run_success=False)
        assert step.action_type == "inspect"

    def test_translate_preserves_new_telemetry_keys(self):
        step = _translate_raw_trace_entry(
            _assistant_entry(ttft_ms=200, decode_ms=80, cached_tokens=50, cache_write_tokens=3),
            is_last=False,
            run_success=True,
        )
        assert step.ttft_ms == 200
        assert step.decode_ms == 80
        assert step.cached_tokens == 50
        assert step.cache_write_tokens == 3


class TestMaterializeTrace:
    """AC-24 — tolerant wrapper."""

    def test_none_input_returns_none(self):
        assert _materialize_trace(None, run_success=True) is None

    def test_empty_input_returns_none(self):
        assert _materialize_trace([], run_success=True) is None

    def test_all_non_dict_input_returns_none(self):
        assert _materialize_trace([None, 42, "str"], run_success=True) is None

    def test_skips_non_dict_entries(self):
        trace = _materialize_trace([_assistant_entry(), None, _execution_entry()], run_success=True)
        assert trace is not None
        assert len(trace.steps) == 2

    def test_populates_all_steps(self):
        entries = [_assistant_entry(), _execution_entry(), _assistant_entry()]
        trace = _materialize_trace(entries, run_success=True)
        assert trace is not None
        assert len(trace.steps) == 3
        # Last step promoted to `final` because run_success=True.
        assert trace.steps[-1].action_type == "final"


class TestRagMaterializedShape:
    """AC-13b — the route-materialized RAG trace pair is [inspect, final]."""

    def test_materialize_rag_trace_yields_inspect_then_final(self):
        rag_raw = [
            {"step": 0, "role": "rag_retrieval", "chunks_retrieved": 3},
            _assistant_entry(step=1),
        ]
        trace = _materialize_trace(rag_raw, run_success=True)
        assert trace is not None
        assert [s.action_type for s in trace.steps] == ["inspect", "final"]
