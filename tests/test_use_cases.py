"""Tests for application-layer use cases.

Each use case is tested with mock ports so no real LLM calls or
sandbox execution is needed. This validates orchestration logic only.
"""

import pytest
from typing import Any, Dict, Iterator, List, Optional

from rlmkit.application.dto import (
    ExecutionResultDTO,
    LLMResponseDTO,
    RunConfigDTO,
    RunResultDTO,
)
from rlmkit.application.use_cases.run_direct import RunDirectUseCase
from rlmkit.application.use_cases.run_rlm import RunRLMUseCase
from rlmkit.application.use_cases.run_rag import RunRAGUseCase
from rlmkit.application.use_cases.run_comparison import (
    ComparisonResultDTO,
    RunComparisonUseCase,
)


# ---------------------------------------------------------------------------
# Mock adapters for port interfaces
# ---------------------------------------------------------------------------


class FakeLLM:
    """Minimal LLMPort-compliant fake for testing use cases."""

    def __init__(self, responses: List[str]) -> None:
        self._responses = responses
        self._idx = 0

    def complete(self, messages: List[Dict[str, str]]) -> LLMResponseDTO:
        idx = min(self._idx, len(self._responses) - 1)
        text = self._responses[idx]
        self._idx += 1
        return LLMResponseDTO(
            content=text,
            model="fake",
            input_tokens=10,
            output_tokens=5,
        )

    def complete_stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        result = self.complete(messages)
        yield result.content

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_pricing(self) -> Dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}


class FakeSandbox:
    """Minimal SandboxPort-compliant fake for testing use cases."""

    def __init__(self) -> None:
        self._namespace: Dict[str, Any] = {}

    def execute(self, code: str) -> ExecutionResultDTO:
        try:
            exec(code, self._namespace)
            import io, contextlib
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

    def get_variable(self, name: str) -> Optional[Any]:
        return self._namespace.get(name)


class FakeEmbedder:
    """Minimal EmbeddingPort-compliant fake."""

    def embed(self, text: str) -> List[float]:
        return [float(len(text) % 10)] * 8

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return 8


class FakeStorage:
    """Minimal StoragePort-compliant fake with in-memory vector search."""

    def __init__(self) -> None:
        self._chunks: List[tuple] = []

    def create_conversation(self, **kwargs: Any) -> str:
        return "conv-1"

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return None

    def list_conversations(self) -> List[Dict[str, Any]]:
        return []

    def delete_conversation(self, conversation_id: str) -> None:
        pass

    def save_file_context(self, content: str, filename: Optional[str] = None) -> str:
        return "hash-1"

    def get_file_context(self, content_hash: str) -> Optional[str]:
        return None

    def add_chunks(
        self,
        collection: str,
        chunks: List[str],
        embeddings: List[List[float]],
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        for chunk, emb in zip(chunks, embeddings):
            self._chunks.append((collection, chunk, emb))
        return len(chunks)

    def search_chunks(
        self,
        collection: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[tuple]:
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
        llm = FakeLLM([
            "```python\nprint('exploring')\n```",
            "FINAL: Found it.",
        ])
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
        llm = FakeLLM([
            "I'm thinking about this...",  # no code, no FINAL
            "FINAL: Got it!",
        ])
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
