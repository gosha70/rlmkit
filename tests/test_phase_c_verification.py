"""Phase C verification tests: RLM mode execute_async with event emitter.

Covers Bug #8 from the outstanding bugs handoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from rlmkit.application.dto import LLMResponseDTO, RunConfigDTO

# ---------------------------------------------------------------------------
# Fake adapters for unit testing (no real LLM / sandbox)
# ---------------------------------------------------------------------------


class FakeLLM:
    """Returns canned responses in sequence."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.active_model = "fake-model"

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        text = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return LLMResponseDTO(
            content=text,
            model="fake-model",
            input_tokens=50,
            output_tokens=30,
        )

    async def complete_async(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        return self.complete(messages)


@dataclass
class FakeExecResult:
    stdout: str = ""
    stderr: str = ""
    exception: str | None = None
    timeout: bool = False


class FakeSandbox:
    """In-memory sandbox that can execute simple expressions."""

    def __init__(self) -> None:
        self._vars: dict[str, Any] = {}

    def set_variable(self, name: str, value: Any) -> None:
        self._vars[name] = value

    def execute(self, code: str) -> FakeExecResult:
        try:
            ns = dict(self._vars)
            exec(code, ns)
            # Capture any print output
            return FakeExecResult(stdout="(executed)")
        except Exception as exc:
            return FakeExecResult(exception=str(exc))


@dataclass
class CollectedEvents:
    tokens: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)


class FakeEventEmitter:
    """Captures emitted events for assertion."""

    def __init__(self) -> None:
        self.events = CollectedEvents()

    async def on_token(self, token: str) -> None:
        self.events.tokens.append(token)

    async def on_step(self, step_data: dict[str, Any]) -> None:
        self.events.steps.append(step_data)

    async def on_metrics(self, metrics: dict[str, Any]) -> None:
        self.events.metrics.append(metrics)


# ---------------------------------------------------------------------------
# Bug #8: RLM mode execute / execute_async
# ---------------------------------------------------------------------------


class TestRLMExecute:
    def test_sync_execute_returns_final_answer(self) -> None:
        """RLM sync execute finds FINAL: answer."""
        from rlmkit.application.use_cases.run_rlm import RunRLMUseCase

        llm = FakeLLM(["FINAL: The answer is 42"])
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)

        result = uc.execute("test content", "What is the answer?")

        assert result.success is True
        assert result.mode_used == "rlm"
        assert "42" in result.answer
        assert result.steps == 1

    def test_sync_execute_runs_code_then_final(self) -> None:
        """RLM sync execute runs code, then finds FINAL on step 2."""
        from rlmkit.application.use_cases.run_rlm import RunRLMUseCase

        llm = FakeLLM(
            [
                "Let me explore.\n```python\nx = len(P)\n```",
                "FINAL: The document has some characters",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)

        result = uc.execute("hello world", "How long?")

        assert result.success is True
        assert result.steps == 2
        assert len(result.trace) >= 2

    def test_sync_execute_respects_budget(self) -> None:
        """RLM stops when budget is exceeded."""
        from rlmkit.application.use_cases.run_rlm import RunRLMUseCase

        # LLM always returns code, never FINAL
        llm = FakeLLM(["```python\nx = 1\n```"] * 20)
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)

        config = RunConfigDTO(mode="rlm", max_steps=3)
        result = uc.execute("content", "query", config)

        # Budget exhaustion is a known config limit — treated as complete with a warning,
        # not an error, so Auto-Judge can evaluate the result.
        assert result.success is True
        assert "⚠️" in result.answer
        assert result.steps <= 3


class TestRLMExecuteAsync:
    @pytest.mark.asyncio
    async def test_async_execute_with_emitter(self) -> None:
        """execute_async emits step and metrics events."""
        from rlmkit.application.use_cases.run_rlm import RunRLMUseCase

        llm = FakeLLM(["FINAL: async answer"])
        sandbox = FakeSandbox()
        emitter = FakeEventEmitter()
        uc = RunRLMUseCase(llm, sandbox)

        result = await uc.execute_async("test content", "What?", event_emitter=emitter)

        assert result.success is True
        assert "async answer" in result.answer
        assert len(emitter.events.steps) >= 1
        assert len(emitter.events.metrics) >= 1
        assert emitter.events.metrics[0]["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_async_execute_code_then_final(self) -> None:
        """execute_async runs code exploration then returns final."""
        from rlmkit.application.use_cases.run_rlm import RunRLMUseCase

        llm = FakeLLM(
            [
                "```python\nresult = P[:10]\n```",
                "FINAL: First 10 chars found",
            ]
        )
        sandbox = FakeSandbox()
        emitter = FakeEventEmitter()
        uc = RunRLMUseCase(llm, sandbox)

        result = await uc.execute_async(
            "hello world data",
            "What are the first chars?",
            event_emitter=emitter,
        )

        assert result.success is True
        assert result.steps == 2
        # Should have step events for both LLM calls
        assert len(emitter.events.steps) >= 2

    @pytest.mark.asyncio
    async def test_async_execute_without_emitter(self) -> None:
        """execute_async works without event_emitter (REST path)."""
        from rlmkit.application.use_cases.run_rlm import RunRLMUseCase

        llm = FakeLLM(["FINAL: no emitter answer"])
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)

        result = await uc.execute_async("content", "query")

        assert result.success is True
        assert "no emitter" in result.answer

    @pytest.mark.asyncio
    async def test_async_budget_exceeded(self) -> None:
        """execute_async returns warning (not error) when budget exceeded."""
        from rlmkit.application.use_cases.run_rlm import RunRLMUseCase

        llm = FakeLLM(["```python\nx = 1\n```"] * 10)
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)

        config = RunConfigDTO(mode="rlm", max_steps=2)
        result = await uc.execute_async("content", "query", config)

        assert result.success is True
        assert "⚠️" in result.answer
