"""Phase 2 — AC-13a. RAG use case returns two-entry trace at the DTO boundary.

Entry 0: rag_retrieval (retrieval-side metrics only).
Entry 1: assistant (LLM completion — four new telemetry keys populated).

Route-layer materialization (AC-13b) is tested in Phase 4 alongside
`_materialize_trace`. This test lives at the DTO boundary per the
spec's v1.5 split.
"""

from __future__ import annotations

from rlmkit.application.dto import LLMResponseDTO, RunConfigDTO
from rlmkit.application.sandbox_vars import (
    TRACE_KEY_ELAPSED_SECONDS,
    TRACE_KEY_INPUT_TOKENS,
    TRACE_KEY_OUTPUT_TOKENS,
    TRACE_KEY_ROLE,
)
from rlmkit.application.use_cases.run_rag import RunRAGUseCase


class _FakeLLM:
    active_model = "fake-model"

    def complete(self, messages):
        return LLMResponseDTO(
            content="the answer",
            model="fake-model",
            input_tokens=42,
            output_tokens=7,
            ttft_ms=110,
            decode_ms=35,
            cached_tokens=30,
            cache_write_tokens=2,
        )

    def get_pricing(self):
        return {"input_cost_per_1m": 1.0, "output_cost_per_1m": 2.0}


class _FakeEmbedder:
    total_tokens = 20
    total_cost = 0.01
    dimension = 8

    def embed(self, text):
        return [0.0] * 8

    def embed_batch(self, texts):
        return [[0.0] * 8 for _ in texts]


class _FakeStorage:
    def add_chunks(self, *, collection, chunks, embeddings):
        self._chunks = chunks

    def search_chunks(self, *, collection, query_embedding, top_k):
        return [(0.9, f"id-{i}", f"chunk {i}") for i in range(min(top_k, 2))]


def _run() -> list[dict]:
    uc = RunRAGUseCase(_FakeLLM(), _FakeEmbedder(), _FakeStorage())
    result = uc.execute(
        content="A document about testing.",
        query="What is this about?",
        config=RunConfigDTO(mode="rag", extra={"top_k": 2, "chunk_size": 100}),
    )
    assert result.success is True
    return list(result.trace)


class TestRagDtoTwoStepShape:
    def test_trace_has_exactly_two_entries(self):
        trace = _run()
        assert len(trace) == 2

    def test_entry_zero_is_rag_retrieval(self):
        trace = _run()
        assert trace[0][TRACE_KEY_ROLE] == "rag_retrieval"
        # Retrieval-side fields present.
        assert trace[0]["chunks_retrieved"] == 2
        assert "scores" in trace[0]
        assert trace[0]["embedding_tokens"] == 20

    def test_entry_one_is_assistant_with_llm_telemetry(self):
        trace = _run()
        assistant = trace[1]
        assert assistant[TRACE_KEY_ROLE] == "assistant"
        assert assistant[TRACE_KEY_INPUT_TOKENS] == 42
        assert assistant[TRACE_KEY_OUTPUT_TOKENS] == 7
        assert assistant[TRACE_KEY_ELAPSED_SECONDS] >= 0
        # The four new plain-string telemetry keys per AC-13a.
        assert assistant["ttft_ms"] == 110
        assert assistant["decode_ms"] == 35
        assert assistant["cached_tokens"] == 30
        assert assistant["cache_write_tokens"] == 2
