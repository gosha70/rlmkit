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

        assert result.success is False
        assert "exceeded" in result.error.lower() or "budget" in result.error.lower()

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

        assert result.success is False
        assert "exceeded" in result.error.lower() or "budget" in result.error.lower()


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

        assert result.success is False
        assert result.steps == 4
        assert "exceeded" in result.error.lower() or "steps" in result.error.lower()
