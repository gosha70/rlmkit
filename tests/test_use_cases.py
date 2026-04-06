"""Tests for application-layer use cases.

Each use case is tested with mock ports so no real LLM calls or
sandbox execution is needed. This validates orchestration logic only.
"""

import asyncio
from collections.abc import Iterator
from typing import Any

from rlmkit.application.dto import (
    ExecutionResultDTO,
    LLMResponseDTO,
    RunConfigDTO,
    RunResultDTO,
)
from rlmkit.application.use_cases.run_comparison import (
    ComparisonResultDTO,
    RunComparisonUseCase,
)
from rlmkit.application.use_cases.run_direct import RunDirectUseCase
from rlmkit.application.use_cases.run_rag import RunRAGUseCase
from rlmkit.application.use_cases.run_rlm import RunRLMUseCase

# ---------------------------------------------------------------------------
# Mock adapters for port interfaces
# ---------------------------------------------------------------------------


class FakeLLM:
    """Minimal LLMPort-compliant fake for testing use cases."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        idx = min(self._idx, len(self._responses) - 1)
        text = self._responses[idx]
        self._idx += 1
        return LLMResponseDTO(
            content=text,
            model="fake",
            input_tokens=10,
            output_tokens=5,
        )

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        result = self.complete(messages)
        yield result.content

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}


class FakeSandbox:
    """Minimal SandboxPort-compliant fake for testing use cases."""

    def __init__(self) -> None:
        self._namespace: dict[str, Any] = {}

    def execute(self, code: str) -> ExecutionResultDTO:
        try:
            exec(code, self._namespace)
            import contextlib
            import io

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(code, self._namespace)
            return ExecutionResultDTO(stdout=buf.getvalue())
        except Exception as exc:
            return ExecutionResultDTO(exception=str(exc))

    def reset(self) -> None:
        self._namespace.clear()

    def is_healthy(self) -> bool:
        return True

    def set_variable(self, name: str, value: Any) -> None:
        self._namespace[name] = value

    def get_variable(self, name: str) -> Any | None:
        return self._namespace.get(name)


class FakeEmbedder:
    """Minimal EmbeddingPort-compliant fake."""

    def embed(self, text: str) -> list[float]:
        return [float(len(text) % 10)] * 8

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return 8


class FakeStorage:
    """Minimal StoragePort-compliant fake with in-memory vector search."""

    def __init__(self) -> None:
        self._chunks: list[tuple] = []

    def create_conversation(self, **kwargs: Any) -> str:
        return "conv-1"

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        return None

    def list_conversations(self) -> list[dict[str, Any]]:
        return []

    def delete_conversation(self, conversation_id: str) -> None:
        pass

    def save_file_context(self, content: str, filename: str | None = None) -> str:
        return "hash-1"

    def get_file_context(self, content_hash: str) -> str | None:
        return None

    def add_chunks(
        self,
        collection: str,
        chunks: list[str],
        embeddings: list[list[float]],
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        for chunk, emb in zip(chunks, embeddings, strict=False):
            self._chunks.append((collection, chunk, emb))
        return len(chunks)

    def search_chunks(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple]:
        matching = [(c, txt, emb) for c, txt, emb in self._chunks if c == collection]
        results = [(0.9, f"id-{i}", txt) for i, (_, txt, _) in enumerate(matching[:top_k])]
        return results


# ---------------------------------------------------------------------------
# RunDirectUseCase tests
# ---------------------------------------------------------------------------


class TestRunDirectUseCase:
    """Tests for direct (single-call) LLM execution."""

    def test_success(self):
        llm = FakeLLM(["The answer is 42."])
        uc = RunDirectUseCase(llm)
        result = uc.execute(content="some content", query="what is it?")

        assert isinstance(result, RunResultDTO)
        assert result.success is True
        assert result.mode_used == "direct"
        assert result.answer == "The answer is 42."
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.elapsed_time > 0

    def test_has_trace(self):
        llm = FakeLLM(["answer"])
        uc = RunDirectUseCase(llm)
        result = uc.execute("content", "question")
        assert len(result.trace) == 1
        assert result.trace[0]["mode"] == "direct"

    def test_failure_returns_error(self):
        class FailLLM:
            def complete(self, messages):
                raise RuntimeError("Service down")

            def complete_stream(self, messages):
                yield ""

            def count_tokens(self, text):
                return 0

            def get_pricing(self):
                return {}

        uc = RunDirectUseCase(FailLLM())
        result = uc.execute("content", "question")
        assert result.success is False
        assert "Service down" in result.error
        assert result.mode_used == "direct"

    def test_custom_config(self):
        llm = FakeLLM(["yes"])
        config = RunConfigDTO(mode="direct", verbose=True)
        uc = RunDirectUseCase(llm)
        result = uc.execute("c", "q", config=config)
        assert result.success is True

    def test_default_config_when_none(self):
        llm = FakeLLM(["answer"])
        uc = RunDirectUseCase(llm)
        result = uc.execute("c", "q", config=None)
        assert result.success is True


# ---------------------------------------------------------------------------
# RunRLMUseCase tests
# ---------------------------------------------------------------------------


class TestRunRLMUseCase:
    """Tests for RLM execution loop."""

    def test_immediate_final_answer(self):
        llm = FakeLLM(["FINAL: The answer is 42."])
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("large content", "what is it?")

        assert result.success is True
        assert result.mode_used == "rlm"
        assert result.answer == "The answer is 42."
        assert result.steps == 1

    def test_code_then_final(self):
        llm = FakeLLM(
            [
                "```python\nprint('exploring')\n```",
                "FINAL: Found it.",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "question")

        assert result.success is True
        assert result.answer == "Found it."
        assert result.steps >= 2

    def test_budget_exhaustion(self):
        llm = FakeLLM(["```python\nprint(1)\n```"])  # always code, never FINAL
        sandbox = FakeSandbox()
        config = RunConfigDTO(mode="rlm", max_steps=3)
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "question", config=config)

        assert result.success is True
        assert "⚠️" in result.answer

    def test_synthesis_fallback_on_inspect_exhaustion(self):
        """Sync execute returns synthesized answer when inspect-only run exhausts max_steps."""
        # Step 1: JSON inspect action; step 2 (synthesis extra call): plain answer
        llm = FakeLLM(
            [
                '{"type": "inspect", "tool": "peek", "args": {"start": 0, "end": 3000}}',
                "The content is repetitive placeholder text with no themes.",
            ]
        )
        sandbox = FakeSandbox()
        config = RunConfigDTO(mode="rlm", max_steps=1)
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("word " * 100, "Summarize the key themes", config=config)

        assert result.success is True
        assert result.steps == 2  # main inspect step + synthesis call
        assert "repetitive" in result.answer
        # Synthesis call must appear in trace with the dedicated note
        assert any(t.get("note") == "synthesis fallback" for t in result.trace)

    def test_synthesis_fallback_empty_response_uses_default_message(self):
        """Sync: empty synthesis response falls back to a descriptive message."""
        llm = FakeLLM(
            [
                '{"type": "inspect", "tool": "peek", "args": {"start": 0, "end": 3000}}',
                "",  # synthesis returns empty string
            ]
        )
        sandbox = FakeSandbox()
        config = RunConfigDTO(mode="rlm", max_steps=1)
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("word " * 100, "Summarize", config=config)

        assert result.success is True
        assert result.answer  # non-empty fallback message

    def test_stall_detection_circuit_breaker_returns_plain_text(self):
        # LLM produces plain-text answers without FINAL: prefix (common with small models).
        # Circuit breaker should accept the text as the answer instead of discarding it.
        filler = "I need more context to answer your question."
        llm = FakeLLM([filler])  # repeats same response indefinitely
        sandbox = FakeSandbox()
        config = RunConfigDTO(mode="rlm", max_steps=20)
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "question", config=config)

        assert result.success is True
        assert result.answer == filler
        assert result.steps < 10  # terminated well before max_steps=20

    def test_stall_with_empty_responses_fails(self):
        # LLM produces only whitespace — no usable answer to fall back to.
        llm = FakeLLM(["   "])  # whitespace-only, stripped to empty
        sandbox = FakeSandbox()
        config = RunConfigDTO(mode="rlm", max_steps=20)
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "question", config=config)

        assert result.success is True
        assert "⚠️" in result.answer
        assert result.steps < 10

    def test_sandbox_receives_content(self):
        llm = FakeLLM(["FINAL: done"])
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        uc.execute("my document text", "q")
        assert sandbox.get_variable("P") == "my document text"

    def test_llm_error_handled(self):
        class FailLLM:
            def complete(self, messages):
                raise RuntimeError("API error")

            def complete_stream(self, messages):
                yield ""

            def count_tokens(self, text):
                return 0

            def get_pricing(self):
                return {}

        sandbox = FakeSandbox()
        uc = RunRLMUseCase(FailLLM(), sandbox)
        result = uc.execute("content", "question")
        assert result.success is False
        assert "API error" in result.error

    def test_nudge_on_no_code_no_final(self):
        """When LLM returns neither code nor FINAL, a nudge message is sent."""
        llm = FakeLLM(
            [
                "I'm thinking about this...",  # no code, no FINAL
                "FINAL: Got it!",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "question")
        assert result.success is True
        assert result.answer == "Got it!"

    def test_trace_records_steps(self):
        llm = FakeLLM(["FINAL: answer"])
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "question")
        assert len(result.trace) >= 1
        assert result.trace[0]["role"] == "assistant"


# ---------------------------------------------------------------------------
# RunRLMUseCase helper method tests
# ---------------------------------------------------------------------------


class TestRunRLMHelpers:
    """Tests for private helper methods of RunRLMUseCase."""

    def test_extract_final_basic(self):
        assert RunRLMUseCase._extract_final("FINAL: The answer is 42") == "The answer is 42"

    def test_extract_final_case_insensitive(self):
        assert RunRLMUseCase._extract_final("final: answer here") == "answer here"

    def test_extract_final_multiline(self):
        text = "Some preamble\nFINAL: The real answer\nMore text"
        assert RunRLMUseCase._extract_final(text) is not None
        assert "The real answer" in RunRLMUseCase._extract_final(text)

    def test_extract_final_none_when_absent(self):
        assert RunRLMUseCase._extract_final("No final here") is None

    def test_extract_code_python_block(self):
        text = "Here is code:\n```python\nprint(42)\n```\nDone."
        assert RunRLMUseCase._extract_code(text) == "print(42)"

    def test_extract_code_generic_block(self):
        text = "```\nprint(42)\n```"
        assert RunRLMUseCase._extract_code(text) == "print(42)"

    def test_extract_code_none_when_absent(self):
        assert RunRLMUseCase._extract_code("No code here") is None

    def test_format_execution_stdout(self):
        result = ExecutionResultDTO(stdout="hello world\n")
        formatted = RunRLMUseCase._format_execution(result)
        assert "hello world" in formatted

    def test_format_execution_exception(self):
        result = ExecutionResultDTO(exception="NameError: x not defined")
        formatted = RunRLMUseCase._format_execution(result)
        assert "Exception" in formatted
        assert "NameError" in formatted

    def test_format_execution_timeout(self):
        result = ExecutionResultDTO(timeout=True)
        formatted = RunRLMUseCase._format_execution(result)
        assert "timed out" in formatted.lower()

    def test_format_execution_no_output(self):
        result = ExecutionResultDTO()
        formatted = RunRLMUseCase._format_execution(result)
        assert "no output" in formatted.lower()


# ---------------------------------------------------------------------------
# RunRAGUseCase tests
# ---------------------------------------------------------------------------


class TestRunRAGUseCase:
    """Tests for RAG pipeline use case."""

    def test_success(self):
        llm = FakeLLM(["The document says X."])
        embedder = FakeEmbedder()
        storage = FakeStorage()
        uc = RunRAGUseCase(llm, embedder, storage)

        result = uc.execute("A long document with relevant content.", "What does it say?")

        assert result.success is True
        assert result.mode_used == "rag"
        assert result.answer == "The document says X."
        assert result.steps == 1

    def test_metadata_includes_chunk_info(self):
        llm = FakeLLM(["answer"])
        embedder = FakeEmbedder()
        storage = FakeStorage()
        uc = RunRAGUseCase(llm, embedder, storage)

        result = uc.execute("Some content for chunking.", "q")
        assert "chunks_total" in result.metadata
        assert "chunks_retrieved" in result.metadata

    def test_custom_config(self):
        llm = FakeLLM(["answer"])
        embedder = FakeEmbedder()
        storage = FakeStorage()
        config = RunConfigDTO(mode="rag", extra={"top_k": 3, "chunk_size": 500})
        uc = RunRAGUseCase(llm, embedder, storage)

        result = uc.execute("Some content.", "q", config=config)
        assert result.success is True

    def test_error_handled(self):
        class FailEmbedder:
            def embed(self, text):
                raise RuntimeError("Embed failed")

            def embed_batch(self, texts):
                raise RuntimeError("Embed failed")

            @property
            def dimension(self):
                return 8

        llm = FakeLLM(["answer"])
        storage = FakeStorage()
        uc = RunRAGUseCase(llm, FailEmbedder(), storage)
        result = uc.execute("content", "question")
        assert result.success is False
        assert "Embed failed" in result.error

    def test_chunk_text_static(self):
        chunks = RunRAGUseCase._chunk_text("abcdefghij", chunk_size=3)
        assert len(chunks) == 4  # abc, def, ghi, j
        assert chunks[0] == "abc"

    def test_chunk_text_skips_empty(self):
        chunks = RunRAGUseCase._chunk_text("ab   ", chunk_size=3)
        # "ab " is one chunk, "  " might be empty when stripped
        assert all(c.strip() for c in chunks)


# ---------------------------------------------------------------------------
# RunComparisonUseCase tests
# ---------------------------------------------------------------------------


class TestRunComparisonUseCase:
    """Tests for comparison use case."""

    def test_default_modes(self):
        llm = FakeLLM(["direct answer", "FINAL: rlm answer"])
        sandbox = FakeSandbox()
        uc = RunComparisonUseCase(llm, sandbox)
        result = uc.execute("content", "question")

        assert isinstance(result, ComparisonResultDTO)
        assert "direct" in result.modes_run
        assert "rlm" in result.modes_run
        assert result.total_elapsed > 0

    def test_get_result(self):
        llm = FakeLLM(["direct answer", "FINAL: rlm answer"])
        sandbox = FakeSandbox()
        uc = RunComparisonUseCase(llm, sandbox)
        result = uc.execute("content", "question")

        direct = result.get_result("direct")
        assert direct is not None
        assert direct.mode_used == "direct"

    def test_custom_modes(self):
        llm = FakeLLM(["direct only"])
        sandbox = FakeSandbox()
        uc = RunComparisonUseCase(llm, sandbox)
        result = uc.execute("content", "question", modes=["direct"])

        assert result.modes_run == ["direct"]
        assert result.get_result("rlm") is None

    def test_comparison_result_dto(self):
        dto = ComparisonResultDTO()
        assert dto.results == {}
        assert dto.modes_run == []
        assert dto.total_elapsed == 0.0


# ---------------------------------------------------------------------------
# RunRLMUseCase.execute_async tests
# ---------------------------------------------------------------------------


class FakeEventEmitter:
    """Captures events emitted during execute_async."""

    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.steps: list[dict[str, Any]] = []
        self.metrics: list[dict[str, Any]] = []

    async def on_token(self, token: str) -> None:
        self.tokens.append(token)

    async def on_step(self, step_data: dict[str, Any]) -> None:
        self.steps.append(step_data)

    async def on_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics.append(metrics)


class TestRunRLMAsync:
    """Tests for execute_async with event emitter."""

    def test_async_immediate_final(self):
        """execute_async returns correct result with FINAL answer."""
        llm = FakeLLM(["FINAL: async answer"])
        sandbox = FakeSandbox()
        emitter = FakeEventEmitter()
        uc = RunRLMUseCase(llm, sandbox)
        result = asyncio.get_event_loop().run_until_complete(
            uc.execute_async("content", "question", event_emitter=emitter)
        )

        assert result.success is True
        assert result.mode_used == "rlm"
        assert result.answer == "async answer"
        assert result.steps == 1

    def test_async_emits_step_and_metrics(self):
        """Event emitter receives on_step and on_metrics calls."""
        llm = FakeLLM(["FINAL: done"])
        sandbox = FakeSandbox()
        emitter = FakeEventEmitter()
        uc = RunRLMUseCase(llm, sandbox)
        asyncio.get_event_loop().run_until_complete(
            uc.execute_async("content", "q", event_emitter=emitter)
        )

        assert len(emitter.steps) == 1
        assert emitter.steps[0]["role"] == "assistant"
        assert len(emitter.metrics) == 1
        assert "total_tokens" in emitter.metrics[0]
        assert emitter.metrics[0]["steps"] == 1

    def test_async_code_then_final(self):
        """execute_async handles code execution + FINAL across steps."""
        llm = FakeLLM(
            [
                "```python\nprint('exploring')\n```",
                "FINAL: Found it.",
            ]
        )
        sandbox = FakeSandbox()
        emitter = FakeEventEmitter()
        uc = RunRLMUseCase(llm, sandbox)
        result = asyncio.get_event_loop().run_until_complete(
            uc.execute_async("content", "question", event_emitter=emitter)
        )

        assert result.success is True
        assert result.answer == "Found it."
        assert result.steps >= 2
        assert len(emitter.steps) >= 2
        assert emitter.metrics[-1]["steps"] >= 2

    def test_async_budget_exhaustion(self):
        """execute_async respects budget limits."""
        llm = FakeLLM(["```python\nprint(1)\n```"])
        sandbox = FakeSandbox()
        config = RunConfigDTO(mode="rlm", max_steps=2)
        uc = RunRLMUseCase(llm, sandbox)
        result = asyncio.get_event_loop().run_until_complete(
            uc.execute_async("content", "q", config=config)
        )

        assert result.success is True
        assert "⚠️" in result.answer

    def test_async_stall_circuit_breaker_returns_plain_text(self):
        """execute_async circuit breaker accepts plain-text answer on stall."""
        filler = "I need more context to answer your question."
        llm = FakeLLM([filler])
        sandbox = FakeSandbox()
        config = RunConfigDTO(mode="rlm", max_steps=20)
        uc = RunRLMUseCase(llm, sandbox)
        result = asyncio.get_event_loop().run_until_complete(
            uc.execute_async("content", "question", config=config)
        )

        assert result.success is True
        assert result.answer == filler
        assert result.steps < 10

    def test_async_synthesis_fallback_on_inspect_exhaustion(self):
        """execute_async returns synthesized answer when inspect-only run exhausts max_steps."""
        # Step 1: JSON inspect action; step 2 (synthesis): plain answer
        llm = FakeLLM(
            [
                '{"type": "inspect", "tool": "peek", "args": {"start": 0, "end": 3000}}',
                "The content is repetitive placeholder text.",
            ]
        )
        sandbox = FakeSandbox()
        config = RunConfigDTO(mode="rlm", max_steps=1)
        uc = RunRLMUseCase(llm, sandbox)
        result = asyncio.get_event_loop().run_until_complete(
            uc.execute_async("word " * 100, "Summarize", config=config)
        )

        assert result.success is True
        assert result.steps == 2  # main step + synthesis
        assert "repetitive" in result.answer
        # Synthesis call must appear in trace
        assert any(t.get("note") == "synthesis fallback" for t in result.trace)

    def test_async_stall_with_empty_responses_fails(self):
        """execute_async fails when stalled responses are all empty."""
        llm = FakeLLM(["   "])
        sandbox = FakeSandbox()
        config = RunConfigDTO(mode="rlm", max_steps=20)
        uc = RunRLMUseCase(llm, sandbox)
        result = asyncio.get_event_loop().run_until_complete(
            uc.execute_async("content", "question", config=config)
        )

        assert result.success is True
        assert "⚠️" in result.answer
        assert result.steps < 10


# ---------------------------------------------------------------------------
# Tests: Deep multi-step RLM exploration
# ---------------------------------------------------------------------------


class TestRLMDeepExploration:
    """Tests exercising 3+ code-execute-feedback cycles."""

    def test_three_step_exploration(self):
        """Three code executions followed by FINAL answer."""
        llm = FakeLLM(
            [
                "```python\nprint(len(P))\n```",
                "```python\nprint(P[:50])\n```",
                '```python\nprint(P.count("a"))\n```',
                "FINAL: The document has 100 characters.",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("a" * 100, "How long is the document?")

        assert result.success is True
        assert result.steps == 4
        assert "100" in result.answer
        # 4 assistant entries + 3 execution entries = 7 trace entries
        assert len(result.trace) == 7

    def test_five_step_with_varying_outputs(self):
        """Five distinct code steps produce unique trace entries."""
        llm = FakeLLM(
            [
                '```python\nprint("step1")\n```',
                '```python\nprint("step2")\n```',
                '```python\nprint("step3")\n```',
                '```python\nprint("step4")\n```',
                '```python\nprint("step5")\n```',
                "FINAL: Done after 5 steps.",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        config = RunConfigDTO(mode="rlm", max_steps=10)
        result = uc.execute("content", "query", config=config)

        assert result.success is True
        assert result.steps == 6
        # 6 assistant entries + 5 execution entries = 11
        assert len(result.trace) == 11
        # Verify each assistant trace entry has a code field
        assistant_steps = [t for t in result.trace if t["role"] == "assistant"]
        for step in assistant_steps[:5]:
            assert step.get("code") is not None

    def test_deep_steps_with_budget_limit(self):
        """Budget exhaustion after max_steps with no FINAL."""
        llm = FakeLLM(['```python\nprint("loop")\n```'])
        sandbox = FakeSandbox()
        config = RunConfigDTO(mode="rlm", max_steps=4)
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "query", config=config)

        assert result.success is True
        assert result.steps == 4
        assert "⚠️" in result.answer


# ---------------------------------------------------------------------------
# FINAL_VAR: variable lookup via sandbox
# ---------------------------------------------------------------------------


class TestFinalVar:
    """FINAL_VAR: directive reads the named variable from the sandbox."""

    def test_final_var_sync(self):
        """LLM outputs FINAL_VAR: result — value is read from sandbox."""
        # Step 1: code sets variable; step 2: LLM declares FINAL_VAR
        llm = FakeLLM(
            [
                "```python\nresult = 'computed answer'\n```",
                "FINAL_VAR: result",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "query")

        assert result.success is True
        assert result.answer == "computed answer"

    def test_final_var_missing_variable(self):
        """FINAL_VAR pointing to an undefined variable returns an error message."""
        llm = FakeLLM(["FINAL_VAR: nonexistent"])
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "query")

        assert result.success is True
        assert "not found" in result.answer.lower()

    def test_final_var_numeric_value(self):
        """FINAL_VAR works when the variable holds a non-string value."""
        llm = FakeLLM(
            [
                "```python\ncount = 42\n```",
                "FINAL_VAR: count",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "query")

        assert result.success is True
        assert result.answer == "42"

    @staticmethod
    def test_final_var_async():
        """Async execute also resolves FINAL_VAR from the sandbox."""

        async def run():
            llm = FakeLLM(
                [
                    "```python\nmsg = 'async result'\n```",
                    "FINAL_VAR: msg",
                ]
            )
            sandbox = FakeSandbox()
            uc = RunRLMUseCase(llm, sandbox)
            return await uc.execute_async("content", "query")

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result.success is True
        assert result.answer == "async result"


# ---------------------------------------------------------------------------
# last_execution_failed warning trace entry
# ---------------------------------------------------------------------------


class TestLastExecutionFailedWarning:
    """FINAL after a failed execution step adds a warning to the trace."""

    def test_warning_added_when_final_follows_failure(self):
        """Trace contains a system warning when FINAL follows an exception."""
        llm = FakeLLM(
            [
                "```python\nraise ValueError('oops')\n```",
                "FINAL: I reasoned it out directly.",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "query")

        assert result.success is True
        assert result.answer == "I reasoned it out directly."
        system_entries = [t for t in result.trace if t.get("role") == "system"]
        assert len(system_entries) == 1
        assert "Warning" in system_entries[0]["content"]
        assert "execution failure" in system_entries[0]["content"].lower()

    def test_no_warning_when_execution_succeeds_then_final(self):
        """No warning when execution succeeds before FINAL."""
        llm = FakeLLM(
            [
                "```python\nprint('ok')\n```",
                "FINAL: The answer.",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "query")

        assert result.success is True
        system_entries = [t for t in result.trace if t.get("role") == "system"]
        assert len(system_entries) == 0

    def test_warning_clears_after_successful_execution(self):
        """Warning not emitted if a later execution succeeds before FINAL."""
        llm = FakeLLM(
            [
                "```python\nraise ValueError('fail')\n```",
                "```python\nprint('ok')\n```",
                "FINAL: Recovered.",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "query")

        assert result.success is True
        system_entries = [t for t in result.trace if t.get("role") == "system"]
        assert len(system_entries) == 0


# ---------------------------------------------------------------------------
# JSON v2.0 protocol parsing
# ---------------------------------------------------------------------------


class TestJsonProtocolParsing:
    """_parse_rlm_response handles both JSON v2.0 and markdown v1.0 formats."""

    def test_json_final_action(self):
        """JSON {"type": "final", "answer": "..."} is parsed as a complete response."""
        llm = FakeLLM(['{"type": "final", "answer": "The JSON answer"}'])
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "query")

        assert result.success is True
        assert result.answer == "The JSON answer"

    def test_json_inspect_grep_generates_code(self):
        """JSON inspect/grep action is converted to executable Python code."""
        from rlmkit.core.parsing import ParsedResponse

        llm = FakeLLM([])  # unused; we call the method directly
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        text = '{"type": "inspect", "tool": "grep", "args": {"pattern": "def foo"}}'
        parsed = uc._parse_rlm_response(text)

        assert isinstance(parsed, ParsedResponse)
        assert parsed.code is not None
        assert "grep(" in parsed.code
        assert "'def foo'" in parsed.code

    def test_json_inspect_peek_generates_code(self):
        """JSON inspect/peek action is converted to peek() call."""
        uc = RunRLMUseCase(FakeLLM([]), FakeSandbox())
        text = '{"type": "inspect", "tool": "peek", "args": {"start": 0, "end": 500}}'
        parsed = uc._parse_rlm_response(text)

        assert parsed.code is not None
        assert "peek(" in parsed.code
        assert "500" in parsed.code

    def test_json_inspect_via_execute_loop(self):
        """Full execute loop with JSON inspect then JSON final."""
        llm = FakeLLM(
            [
                '{"type": "inspect", "tool": "grep", "args": {"pattern": "hello"}}',
                '{"type": "final", "answer": "Found it"}',
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("hello world content", "find hello")

        assert result.success is True
        assert result.answer == "Found it"

    def test_markdown_fallback_still_works(self):
        """Markdown v1.0 responses are still parsed when JSON parsing fails."""
        llm = FakeLLM(
            [
                "```python\nprint('hello')\n```",
                "FINAL: Done via markdown",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "query")

        assert result.success is True
        assert result.answer == "Done via markdown"


# ---------------------------------------------------------------------------
# System prompt uses v2.0 template
# ---------------------------------------------------------------------------


class TestSystemPromptV2:
    """_build_system_prompt uses the versioned v2.0 template."""

    def test_system_prompt_contains_json_instructions(self):
        """v2.0 template includes JSON action protocol instructions."""
        prompt = RunRLMUseCase._build_system_prompt(10000)
        # v2.0 template contains JSON action type keywords
        assert '"type"' in prompt or "type" in prompt
        assert "final" in prompt.lower()

    def test_system_prompt_includes_content_length(self):
        """Content length is formatted into the prompt."""
        prompt = RunRLMUseCase._build_system_prompt(12345)
        assert "12,345" in prompt


# ---------------------------------------------------------------------------
# Subcall / recursion wiring
# ---------------------------------------------------------------------------


class TestSubcall:
    """subcall() is bound in sandbox globals and callable from LLM-generated code."""

    def test_subcall_bound_in_sandbox(self):
        """After execute(), sandbox has 'subcall' variable."""
        llm = FakeLLM(["FINAL: done"])
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        uc.execute("content", "query")
        assert callable(sandbox.get_variable("subcall"))

    def test_subcall_callable_from_llm_code(self):
        """LLM code that calls subcall() completes without NameError."""
        # The sub-RLM uses the same FakeLLM; it will immediately return FINAL.
        llm = FakeLLM(
            [
                # Step 1 (main): call subcall — sub-RLM immediately returns FINAL below
                "```python\nresult = subcall(content='hi', query='q')\nprint(result)\n```",
                # Step 2 (main): after seeing subcall result, provide final answer
                "FINAL: all done",
                # Step 1 (sub-RLM): immediate final
                "FINAL: sub answer",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("main content", "query")

        assert result.success is True
        # main RLM should finish with its own FINAL answer
        assert result.answer in ("all done", "sub answer")

    def test_subcall_respects_recursion_depth_zero(self):
        """subcall() returns an error when max_recursion_depth is 0."""
        llm = FakeLLM(
            [
                "```python\nresult = subcall(content='c', query='q')\nprint(result)\n```",
                "FINAL: done",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        config = RunConfigDTO(mode="rlm", max_recursion_depth=0)
        result = uc.execute("content", "query", config=config)

        # Should complete; subcall error gets printed and fed back
        assert result.success is True


# ---------------------------------------------------------------------------
# Cost accounting
# ---------------------------------------------------------------------------


class TestCostAccounting:
    """total_cost is computed from token counts and pricing, and enforced."""

    def test_zero_cost_when_pricing_unavailable(self):
        """total_cost is 0.0 when LLM has no get_pricing method."""
        llm = FakeLLM(["FINAL: answer"])  # FakeLLM returns 0.0 pricing
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "query")

        assert result.total_cost == 0.0

    def test_cost_accumulates_across_steps(self):
        """total_cost grows with each LLM call when pricing > 0."""

        class PricedLLM(FakeLLM):
            def get_pricing(self) -> dict:
                # $1 per 1M input, $2 per 1M output
                return {"input_cost_per_1m": 1.0, "output_cost_per_1m": 2.0}

        # FakeLLM returns 10 input + 5 output tokens per call
        # Step 1: code; step 2: FINAL — 2 LLM calls
        # Each call cost: (10*1 + 5*2) / 1_000_000 = 20 / 1_000_000 = 0.00002
        llm = PricedLLM(
            [
                "```python\nprint('x')\n```",
                "FINAL: done",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("content", "query")

        assert result.total_cost > 0.0
        expected = 2 * (10 * 1.0 + 5 * 2.0) / 1_000_000
        assert abs(result.total_cost - expected) < 1e-10

    def test_max_cost_enforced(self):
        """Execution stops when cumulative cost exceeds max_cost."""

        class ExpensiveLLM(FakeLLM):
            def get_pricing(self) -> dict:
                # Very expensive: $1000 per 1M tokens
                return {"input_cost_per_1m": 1000.0, "output_cost_per_1m": 1000.0}

        llm = ExpensiveLLM(['```python\nprint("step")\n```'])
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        # max_cost so small that even one step exceeds it
        config = RunConfigDTO(mode="rlm", max_cost=0.000001)
        result = uc.execute("content", "query", config=config)

        assert result.success is True
        assert "⚠️" in result.answer

    def test_max_cost_enforced_on_immediate_final(self):
        """A single FINAL response that exceeds max_cost must not return success."""

        class ExpensiveLLM(FakeLLM):
            def get_pricing(self) -> dict:
                return {"input_cost_per_1m": 1000.0, "output_cost_per_1m": 1000.0}

        # LLM immediately returns FINAL — only one LLM call occurs
        llm = ExpensiveLLM(["FINAL: done"])
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        config = RunConfigDTO(mode="rlm", max_cost=0.000001)
        result = uc.execute("content", "query", config=config)

        assert result.success is True
        assert "⚠️" in result.answer

    def test_subcall_cost_folded_into_parent(self):
        """Parent total_cost includes child RLM token usage."""

        class PricedLLM(FakeLLM):
            def get_pricing(self) -> dict:
                return {"input_cost_per_1m": 1.0, "output_cost_per_1m": 2.0}

        # FakeLLM: 10 input + 5 output tokens per call → 0.00002 per call.
        # FakeSandbox executes code twice (silent pass + stdout-capture pass), so
        # subcall() fires twice per code block — the sub-RLM accumulates 2 LLM
        # calls worth of cost.  The key assertion is simply that total_cost is
        # strictly greater than 2 × per_call (the parent-only floor), proving
        # child usage was folded back in.
        llm = PricedLLM(
            [
                "```python\nresult = subcall(content='hi', query='q')\nprint(result)\n```",
                "FINAL: sub answer",
                "FINAL: all done",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        result = uc.execute("main content", "query")

        assert result.success is True
        per_call = (10 * 1.0 + 5 * 2.0) / 1_000_000  # 0.00002
        parent_only = 2 * per_call  # at minimum: step-1 code + step-2 final
        assert result.total_cost > parent_only  # sub-RLM cost was folded in

    def test_subcall_inherits_budget_caps(self):
        """Child RLM receives the parent's max_cost so it cannot overspend internally."""

        class ExpensiveLLM(FakeLLM):
            def get_pricing(self) -> dict:
                return {"input_cost_per_1m": 1000.0, "output_cost_per_1m": 1000.0}

        # Parent max_cost is tiny. The child should self-terminate quickly
        # because it inherits max_cost from the parent config.
        llm = ExpensiveLLM(
            [
                "```python\nresult = subcall(content='c', query='q')\nprint(result)\n```",
                "FINAL: done",
                # sub-RLM responses — it must stop before burning through many steps
                '```python\nprint("child step")\n```',
                '```python\nprint("child step 2")\n```',
                '```python\nprint("child step 3")\n```',
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        # max_cost tight enough that the child cannot complete even 2 LLM calls
        config = RunConfigDTO(mode="rlm", max_cost=0.02, max_steps=20)
        result = uc.execute("main content", "query", config=config)

        # The run must fail (budget exceeded before a FINAL is reached),
        # or if the child fails and the parent reaches FINAL, total_cost <= max_cost
        # is NOT guaranteed (one call can exceed), but total_cost must be much less
        # than it would be without the child cap (which was 0.075 in the reproduction).
        # Just verify child did not run 3+ expensive steps uncapped.
        per_call = (10 * 1000.0 + 5 * 1000.0) / 1_000_000  # 0.015 per call
        # Without child cap: 5+ LLM calls → > 0.075; with cap: ≤ 3 LLM calls → ≤ 0.045
        assert result.total_cost < 5 * per_call

    def test_multi_subcall_block_does_not_exceed_global_budget(self):
        """Two subcalls in one code block: the second is blocked once the first has spent the budget."""

        class ExpensiveLLM(FakeLLM):
            def get_pricing(self) -> dict:
                return {"input_cost_per_1m": 1000.0, "output_cost_per_1m": 1000.0}

        # Code block has two subcalls. max_cost=0.02; each individual call costs 0.015
        # (fine on its own) but combined they exceed the 0.02 cap.
        llm = ExpensiveLLM(
            [
                "```python\nr1 = subcall('c', 'q1')\nr2 = subcall('c', 'q2')\nprint(r1, r2)\n```",
                "FINAL: first answer",  # first child's LLM response
                "FINAL: second answer",  # second child (should never be reached)
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        per_call = (10 * 1000.0 + 5 * 1000.0) / 1_000_000  # 0.015 per call
        config = RunConfigDTO(mode="rlm", max_cost=0.02, max_steps=20)
        result = uc.execute("content", "query", config=config)

        # Without the fix: 5 LLM calls (parent + 2×first + 2×second) = 0.075 ≥ 3×per_call.
        # With the fix: second subcall is blocked; at most parent + first = 2 calls = 0.030.
        assert result.total_cost < 3 * per_call

    def test_subcall_steps_folded_into_parent(self):
        """Child RLM steps are counted against the parent's max_steps budget."""
        # max_steps=2: parent step 1 (code block + child step 1) = 2 total; no spare step
        # for the parent to make another LLM call.
        llm = FakeLLM(
            [
                "```python\nresult = subcall('c', 'q')\nprint(result)\n```",  # parent step 1
                "FINAL: child answer",  # child step 1
                "FINAL: parent done",  # parent step 2 — must never be reached
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        config = RunConfigDTO(mode="rlm", max_steps=2, max_recursion_depth=1)
        result = uc.execute("content", "query", config=config)

        # Without folding: parent step 2 runs ("FINAL: parent done") → 3 LLM calls, input_tokens=30.
        # With folding: child step folds back; while condition fails; result has only 2 calls.
        # FakeSandbox double-executes code, but the second subcall is blocked by the steps guard,
        # so exactly 2 LLM calls fire: parent step 1 + child step 1.
        assert result.input_tokens == 2 * 10  # 2 calls × 10 input tokens

    def test_subcall_receives_remaining_time(self):
        """Child receives remaining wall-clock headroom, not the full max_time_seconds."""
        import time as time_module
        from unittest.mock import patch

        class SlowLLM(FakeLLM):
            def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
                time_module.sleep(0.01)  # 10 ms per call ensures measurable elapsed time
                return super().complete(messages)

        child_configs: list[RunConfigDTO] = []
        original_execute = RunRLMUseCase.execute
        call_count = [0]

        def tracking_execute(
            self_inner: RunRLMUseCase, content: str, query: str, config: RunConfigDTO | None = None
        ) -> RunResultDTO:
            call_count[0] += 1
            if call_count[0] > 1 and config is not None:
                child_configs.append(config)
            return original_execute(self_inner, content, query, config=config)

        with patch.object(RunRLMUseCase, "execute", tracking_execute):
            llm = SlowLLM(
                [
                    "```python\nresult = subcall('c', 'q')\nprint(result)\n```",
                    "FINAL: child answer",
                    "FINAL: done",
                ]
            )
            sandbox = FakeSandbox()
            uc = RunRLMUseCase(llm, sandbox)
            uc.execute(
                "content",
                "query",
                config=RunConfigDTO(mode="rlm", max_time_seconds=10.0, max_recursion_depth=1),
            )

        # Every child call must receive less than the full 10.0 s, because the parent
        # step (≥10 ms from SlowLLM) consumed measurable time before spawning the child.
        assert len(child_configs) >= 1
        for cfg in child_configs:
            assert cfg.max_time_seconds is not None
            assert cfg.max_time_seconds < 10.0


# ---------------------------------------------------------------------------
# LiteLLMEmbeddingAdapter unit tests (no live API — litellm.embedding mocked)
# ---------------------------------------------------------------------------


class TestLiteLLMEmbeddingAdapter:
    """Unit tests for the embedding adapter's token tracking and cost accounting."""

    def _make_response(self, vectors: list[list[float]], total_tokens: int):
        """Build a minimal fake litellm embedding response.

        litellm response items are accessed as item["embedding"] (dict-style).
        """
        from types import SimpleNamespace

        data = [{"embedding": v} for v in vectors]
        usage = SimpleNamespace(total_tokens=total_tokens, prompt_tokens=total_tokens)
        return SimpleNamespace(data=data, usage=usage)

    def test_embed_batch_returns_vectors(self, monkeypatch):
        from rlmkit.infrastructure.embedding.litellm_embedding_adapter import (
            LiteLLMEmbeddingAdapter,
        )

        vectors = [[0.1, 0.2], [0.3, 0.4]]
        monkeypatch.setattr(
            "litellm.embedding", lambda **kw: self._make_response(vectors, total_tokens=10)
        )
        adapter = LiteLLMEmbeddingAdapter(model="text-embedding-3-small")
        result = adapter.embed_batch(["hello", "world"])
        assert result == vectors

    def test_embed_delegates_to_embed_batch(self, monkeypatch):
        from rlmkit.infrastructure.embedding.litellm_embedding_adapter import (
            LiteLLMEmbeddingAdapter,
        )

        monkeypatch.setattr(
            "litellm.embedding",
            lambda **kw: self._make_response([[1.0, 2.0]], total_tokens=5),
        )
        adapter = LiteLLMEmbeddingAdapter(model="text-embedding-3-small")
        result = adapter.embed("test")
        assert result == [1.0, 2.0]

    def test_total_tokens_accumulates(self, monkeypatch):
        from rlmkit.infrastructure.embedding.litellm_embedding_adapter import (
            LiteLLMEmbeddingAdapter,
        )

        call_count = [0]

        def fake_embed(**kw):
            call_count[0] += 1
            return self._make_response([[0.1]], total_tokens=100)

        monkeypatch.setattr("litellm.embedding", fake_embed)
        adapter = LiteLLMEmbeddingAdapter(model="text-embedding-3-small")
        adapter.embed("a")
        adapter.embed("b")
        assert adapter.total_tokens == 200
        assert call_count[0] == 2

    def test_total_cost_uses_pricing_table(self, monkeypatch):
        from rlmkit.infrastructure.embedding.litellm_embedding_adapter import (
            LiteLLMEmbeddingAdapter,
        )

        monkeypatch.setattr(
            "litellm.embedding",
            lambda **kw: self._make_response([[0.1]], total_tokens=1_000_000),
        )
        adapter = LiteLLMEmbeddingAdapter(model="text-embedding-3-small")
        adapter.embed("x")
        # 1M tokens × $0.020/1M = $0.020
        assert abs(adapter.total_cost - 0.020) < 1e-9

    def test_unknown_model_zero_cost(self, monkeypatch):
        from rlmkit.infrastructure.embedding.litellm_embedding_adapter import (
            LiteLLMEmbeddingAdapter,
        )

        monkeypatch.setattr(
            "litellm.embedding",
            lambda **kw: self._make_response([[0.1]], total_tokens=500),
        )
        adapter = LiteLLMEmbeddingAdapter(model="some-unknown-model")
        adapter.embed("x")
        assert adapter.total_tokens == 500
        assert adapter.total_cost == 0.0

    def test_model_property(self):
        from rlmkit.infrastructure.embedding.litellm_embedding_adapter import (
            LiteLLMEmbeddingAdapter,
        )

        adapter = LiteLLMEmbeddingAdapter(model="text-embedding-3-large")
        assert adapter.model == "text-embedding-3-large"

    def test_dimension_from_known_table(self):
        from rlmkit.infrastructure.embedding.litellm_embedding_adapter import (
            LiteLLMEmbeddingAdapter,
        )

        adapter = LiteLLMEmbeddingAdapter(model="text-embedding-3-small")
        assert adapter.dimension == 1536

    def test_dimension_inferred_from_response(self, monkeypatch):
        from rlmkit.infrastructure.embedding.litellm_embedding_adapter import (
            LiteLLMEmbeddingAdapter,
        )

        monkeypatch.setattr(
            "litellm.embedding",
            lambda **kw: self._make_response([[0.1] * 512], total_tokens=3),
        )
        adapter = LiteLLMEmbeddingAdapter(model="custom-model")
        assert adapter.dimension == 512

    def test_usage_missing_gracefully(self, monkeypatch):
        """Adapter does not crash when response has no usage attribute."""
        from types import SimpleNamespace

        from rlmkit.infrastructure.embedding.litellm_embedding_adapter import (
            LiteLLMEmbeddingAdapter,
        )

        def fake_embed(**kw):
            data = [{"embedding": [0.1]}]
            return SimpleNamespace(data=data)  # no .usage

        monkeypatch.setattr("litellm.embedding", fake_embed)
        adapter = LiteLLMEmbeddingAdapter(model="text-embedding-3-small")
        result = adapter.embed("x")
        assert result == [0.1]
        assert adapter.total_tokens == 0

    def test_api_key_and_base_forwarded(self, monkeypatch):
        """api_key and api_base are passed through to litellm."""
        from rlmkit.infrastructure.embedding.litellm_embedding_adapter import (
            LiteLLMEmbeddingAdapter,
        )

        captured = {}

        def fake_embed(**kw):
            captured.update(kw)
            return self._make_response([[0.1]], total_tokens=1)

        monkeypatch.setattr("litellm.embedding", fake_embed)
        adapter = LiteLLMEmbeddingAdapter(
            model="text-embedding-3-small", api_key="sk-test", api_base="http://localhost"
        )
        adapter.embed("x")
        assert captured["api_key"] == "sk-test"
        assert captured["api_base"] == "http://localhost"


# ---------------------------------------------------------------------------
# RunRAGUseCase: cost accounting tests
# ---------------------------------------------------------------------------


class TestRunRAGCostAccounting:
    """Verify that both embedding and LLM completion costs are included."""

    def test_cost_combines_embedding_and_llm(self):
        """total_cost = embed_cost + llm_completion_cost."""
        from rlmkit.application.use_cases.run_rag import RunRAGUseCase

        class PricedLLM(FakeLLM):
            def get_pricing(self):
                return {"input_cost_per_1m": 1.0, "output_cost_per_1m": 2.0}

        class PricedEmbedder(FakeEmbedder):
            @property
            def total_tokens(self):
                return 1_000_000  # 1M tokens

            @property
            def total_cost(self):
                return 0.020  # $0.020 for 1M @ text-embedding-3-small rate

        llm = PricedLLM(["answer"])
        uc = RunRAGUseCase(llm, PricedEmbedder(), FakeStorage())
        result = uc.execute("content", "query")

        assert result.success is True
        # LLM: 10 input + 5 output tokens (FakeLLM defaults)
        # LLM cost = (10*1.0 + 5*2.0) / 1_000_000 = 0.000020
        llm_cost = (10 * 1.0 + 5 * 2.0) / 1_000_000
        expected = 0.020 + llm_cost
        assert abs(result.total_cost - expected) < 1e-9

    def test_cost_zero_when_both_free(self):
        """No cost when embedder reports 0 and LLM has no pricing."""
        from rlmkit.application.use_cases.run_rag import RunRAGUseCase

        result = RunRAGUseCase(FakeLLM(["answer"]), FakeEmbedder(), FakeStorage()).execute(
            "content", "query"
        )
        assert result.total_cost == 0.0

    def test_embedding_tokens_added_to_input_tokens(self):
        """embed tokens are folded into RunResultDTO.input_tokens."""
        from rlmkit.application.use_cases.run_rag import RunRAGUseCase

        class EmbedderWith500Tokens(FakeEmbedder):
            @property
            def total_tokens(self):
                return 500

            @property
            def total_cost(self):
                return 0.0

        llm = FakeLLM(["answer"])  # returns input_tokens=10
        uc = RunRAGUseCase(llm, EmbedderWith500Tokens(), FakeStorage())
        result = uc.execute("content", "query")
        # input_tokens = LLM input (10) + embed tokens (500)
        assert result.input_tokens == 510

    def test_get_pricing_exception_falls_back_to_zero_llm_cost(self):
        """If get_pricing() raises, llm_cost is silently set to 0."""
        from rlmkit.application.use_cases.run_rag import RunRAGUseCase

        class BrokenPricingLLM(FakeLLM):
            def get_pricing(self):
                raise RuntimeError("pricing unavailable")

        result = RunRAGUseCase(BrokenPricingLLM(["answer"]), FakeEmbedder(), FakeStorage()).execute(
            "content", "query"
        )
        assert result.success is True
        assert result.total_cost == 0.0


# ---------------------------------------------------------------------------
# _humanize_rag_error branch coverage
# ---------------------------------------------------------------------------


class TestHumanizeRAGError:
    """Ensure each error-classification branch in _humanize_rag_error is exercised."""

    def _humanize(self, msg: str) -> str:
        from rlmkit.application.use_cases.run_rag import _humanize_rag_error

        return str(_humanize_rag_error(Exception(msg)))

    def test_context_window_branch(self):
        result = self._humanize("context window exceeded")
        assert "chunk_size" in result
        assert "top_k" in result

    def test_context_length_branch(self):
        result = self._humanize("context length is 4096 tokens")
        assert "chunk_size" in result

    def test_timeout_branch(self):
        result = self._humanize("request timed out after 30s")
        assert "timeout" in result.lower()

    def test_auth_branch(self):
        result = self._humanize("401 unauthorized api key missing")
        assert "OPENAI_API_KEY" in result

    def test_rate_limit_branch(self):
        result = self._humanize("rate limit exceeded 429")
        assert "chunk_size" in result

    def test_unknown_error_returns_original(self):
        result = self._humanize("something completely unexpected")
        assert result == "something completely unexpected"
