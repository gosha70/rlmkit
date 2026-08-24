"""Tests for Cycle 1 cool-down fixes: MAJOR-1, MAJOR-2, MAJOR-3, MINOR-5."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rlmstudio.application.dto import ExecutionResultDTO, LLMResponseDTO, RunConfigDTO
from rlmstudio.application.use_cases.run_rlm import RunRLMUseCase

# ---------------------------------------------------------------------------
# MAJOR-1: restricted_sandbox.py imports SecurityViolationError from domain
# ---------------------------------------------------------------------------


class TestMajor1DomainImport:
    """Verify restricted_sandbox uses domain exceptions, not core.errors."""

    def test_restricted_sandbox_imports_domain_exception(self):
        """restricted_sandbox should import SecurityViolationError from domain."""
        import rlmstudio.infrastructure.sandbox.restricted_sandbox as mod

        # The module should reference SecurityViolationError from domain
        from rlmstudio.domain.exceptions import SecurityViolationError

        assert hasattr(mod, "SecurityViolationError")
        assert mod.SecurityViolationError is SecurityViolationError

    def test_restricted_sandbox_does_not_import_core_errors(self):
        """restricted_sandbox should NOT import from rlmstudio.core.errors."""
        import inspect

        import rlmstudio.infrastructure.sandbox.restricted_sandbox as mod

        source = inspect.getsource(mod)
        assert "from rlmstudio.core.errors" not in source


# ---------------------------------------------------------------------------
# MAJOR-2: Docker sandbox adapter wraps DockerExecutor as SandboxPort
# ---------------------------------------------------------------------------


class TestMajor2DockerSandboxAdapter:
    """Verify DockerSandboxAdapter implements SandboxPort protocol."""

    def test_adapter_exists(self):
        from rlmstudio.infrastructure.sandbox.docker_sandbox_adapter import (
            DockerSandboxAdapter,
        )

        assert DockerSandboxAdapter is not None

    def test_adapter_exported_from_package(self):
        from rlmstudio.infrastructure.sandbox import DockerSandboxAdapter

        assert DockerSandboxAdapter is not None

    @patch("rlmstudio.envs.sandbox.DockerExecutor")
    def test_execute_returns_execution_result_dto(self, MockExecutor):
        mock_instance = MagicMock()
        mock_instance.execute.return_value = {
            "result": True,
            "output": "hello world",
            "error": None,
        }
        MockExecutor.return_value = mock_instance

        from rlmstudio.infrastructure.sandbox.docker_sandbox_adapter import (
            DockerSandboxAdapter,
        )

        adapter = DockerSandboxAdapter.__new__(DockerSandboxAdapter)
        adapter._executor = mock_instance
        adapter._namespace = {}

        result = adapter.execute("print('hello world')")
        assert isinstance(result, ExecutionResultDTO)
        assert result.stdout == "hello world"
        assert result.exception is None

    @patch("rlmstudio.envs.sandbox.DockerExecutor")
    def test_execute_handles_error(self, MockExecutor):
        mock_instance = MagicMock()
        mock_instance.execute.return_value = {
            "result": False,
            "output": "",
            "error": "NameError: name 'x' is not defined",
        }
        MockExecutor.return_value = mock_instance

        from rlmstudio.infrastructure.sandbox.docker_sandbox_adapter import (
            DockerSandboxAdapter,
        )

        adapter = DockerSandboxAdapter.__new__(DockerSandboxAdapter)
        adapter._executor = mock_instance
        adapter._namespace = {}

        result = adapter.execute("print(x)")
        assert not result.success
        assert "NameError" in result.exception

    @patch("rlmstudio.envs.sandbox.DockerExecutor")
    def test_execute_handles_timeout(self, MockExecutor):
        mock_instance = MagicMock()
        mock_instance.execute.return_value = {
            "result": False,
            "output": "",
            "error": "Execution timed out after 30 seconds",
        }
        MockExecutor.return_value = mock_instance

        from rlmstudio.infrastructure.sandbox.docker_sandbox_adapter import (
            DockerSandboxAdapter,
        )

        adapter = DockerSandboxAdapter.__new__(DockerSandboxAdapter)
        adapter._executor = mock_instance
        adapter._namespace = {}

        result = adapter.execute("while True: pass")
        assert result.timeout is True

    @patch("rlmstudio.envs.sandbox.DockerExecutor")
    def test_set_get_variable(self, MockExecutor):
        from rlmstudio.infrastructure.sandbox.docker_sandbox_adapter import (
            DockerSandboxAdapter,
        )

        adapter = DockerSandboxAdapter.__new__(DockerSandboxAdapter)
        adapter._executor = MagicMock()
        adapter._namespace = {}

        adapter.set_variable("x", 42)
        assert adapter.get_variable("x") == 42
        assert adapter.get_variable("nonexistent") is None

    @patch("rlmstudio.envs.sandbox.DockerExecutor")
    def test_reset_clears_namespace(self, MockExecutor):
        from rlmstudio.infrastructure.sandbox.docker_sandbox_adapter import (
            DockerSandboxAdapter,
        )

        adapter = DockerSandboxAdapter.__new__(DockerSandboxAdapter)
        adapter._executor = MagicMock()
        adapter._namespace = {"x": 1}

        adapter.reset()
        assert adapter._namespace == {}

    def test_sandbox_factory_docker_type(self):
        """sandbox_factory should return DockerSandboxAdapter for 'docker' type."""
        from rlmstudio.infrastructure.sandbox.sandbox_factory import create_sandbox

        with patch("rlmstudio.envs.sandbox.DockerExecutor") as MockExecutor:
            MockExecutor.is_available.return_value = True
            MockExecutor.return_value = MagicMock()

            sandbox = create_sandbox(sandbox_type="docker")
            from rlmstudio.infrastructure.sandbox.docker_sandbox_adapter import (
                DockerSandboxAdapter,
            )

            assert isinstance(sandbox, DockerSandboxAdapter)


# ---------------------------------------------------------------------------
# MAJOR-3: Async port methods
# ---------------------------------------------------------------------------


class TestMajor3AsyncPorts:
    """Verify async methods exist on port protocols and LiteLLM adapter."""

    def test_llm_port_has_complete_async(self):
        import inspect

        from rlmstudio.application.ports.llm_port import LLMPort

        assert hasattr(LLMPort, "complete_async")
        assert inspect.iscoroutinefunction(LLMPort.complete_async)

    def test_llm_port_has_complete_stream_async(self):
        import inspect

        from rlmstudio.application.ports.llm_port import LLMPort

        assert hasattr(LLMPort, "complete_stream_async")
        assert inspect.isasyncgenfunction(LLMPort.complete_stream_async)

    def test_sandbox_port_has_execute_async(self):
        import inspect

        from rlmstudio.application.ports.sandbox_port import SandboxPort

        assert hasattr(SandboxPort, "execute_async")
        assert inspect.iscoroutinefunction(SandboxPort.execute_async)

    def test_litellm_adapter_has_complete_async(self):
        import inspect

        from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter

        assert hasattr(LiteLLMAdapter, "complete_async")
        assert inspect.iscoroutinefunction(LiteLLMAdapter.complete_async)

    def test_litellm_adapter_has_complete_stream_async(self):
        import inspect

        from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter

        assert hasattr(LiteLLMAdapter, "complete_stream_async")
        assert inspect.isasyncgenfunction(LiteLLMAdapter.complete_stream_async)

    @pytest.mark.asyncio
    async def test_litellm_complete_async_calls_acompletion(self):
        """LiteLLMAdapter.complete_async should call litellm.acompletion.

        Phase 1 routes complete_async through streaming under the hood,
        so the mocked acompletion return must be a chunk iterator
        carrying usage on the terminal chunk.
        """
        from types import SimpleNamespace

        from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter

        adapter = LiteLLMAdapter(model="gpt-4o")

        content_chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="async answer", role=None),
                    finish_reason=None,
                    index=0,
                )
            ],
            model="gpt-4o",
            usage=None,
        )
        terminal_chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, role=None),
                    finish_reason="stop",
                    index=0,
                )
            ],
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        async def mock_async_iter():
            yield content_chunk
            yield terminal_chunk

        with patch("litellm.acompletion", return_value=mock_async_iter()) as mock_ac:
            result = await adapter.complete_async([{"role": "user", "content": "hello"}])
            mock_ac.assert_called_once()
            assert isinstance(result, LLMResponseDTO)
            assert result.content == "async answer"
            assert result.input_tokens == 10
            assert result.output_tokens == 5
            # stream-under-the-hood is the default; TTFT is measured
            assert result.ttft_ms is not None

    @pytest.mark.asyncio
    async def test_litellm_complete_stream_async_yields_chunks(self):
        """LiteLLMAdapter.complete_stream_async yields StreamChunk events.

        Non-final chunks carry text deltas; the final chunk has
        is_final=True and a populated response DTO.
        """
        from types import SimpleNamespace

        from rlmstudio.application.dto import StreamChunk
        from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter

        adapter = LiteLLMAdapter(model="gpt-4o")

        def _content_chunk(text: str):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=text, role=None),
                        finish_reason=None,
                        index=0,
                    )
                ],
                model="gpt-4o",
                usage=None,
            )

        terminal_chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, role=None),
                    finish_reason="stop",
                    index=0,
                )
            ],
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

        async def mock_async_iter():
            yield _content_chunk("Hello")
            yield _content_chunk(" World")
            yield terminal_chunk

        with patch("litellm.acompletion", return_value=mock_async_iter()):
            collected: list[StreamChunk] = []
            async for chunk in adapter.complete_stream_async([{"role": "user", "content": "hi"}]):
                collected.append(chunk)

            # Two content StreamChunks + one terminal StreamChunk.
            assert [c.delta for c in collected if not c.is_final] == ["Hello", " World"]
            assert collected[-1].is_final is True
            assert collected[-1].response is not None
            assert collected[-1].response.input_tokens == 3
            assert collected[-1].response.output_tokens == 2
            assert collected[-1].response.ttft_ms is not None


# ---------------------------------------------------------------------------
# MINOR-5: Two-model switching wired into RunRLMUseCase
# ---------------------------------------------------------------------------


class FakeTwoModelLLM:
    """LLM fake that tracks model-switching calls."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0
        self._active_model = "root-model"
        self.model_history: list[str] = []

    def complete(self, messages: list[dict[str, str]]) -> LLMResponseDTO:
        self.model_history.append(self._active_model)
        idx = min(self._idx, len(self._responses) - 1)
        text = self._responses[idx]
        self._idx += 1
        return LLMResponseDTO(
            content=text, model=self._active_model, input_tokens=10, output_tokens=5
        )

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield self.complete(messages).content

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_pricing(self) -> dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

    def use_root_model(self) -> None:
        self._active_model = "root-model"

    def use_recursive_model(self) -> None:
        self._active_model = "recursive-model"


class FakeSandbox:
    """Minimal SandboxPort-compliant fake."""

    def __init__(self) -> None:
        self._namespace: dict[str, Any] = {}

    def execute(self, code: str) -> ExecutionResultDTO:
        return ExecutionResultDTO(stdout="ok")

    def reset(self) -> None:
        self._namespace.clear()

    def is_healthy(self) -> bool:
        return True

    def set_variable(self, name: str, value: Any) -> None:
        self._namespace[name] = value

    def get_variable(self, name: str) -> Any | None:
        return self._namespace.get(name)


class TestMinor5TwoModelSwitching:
    """Verify RunRLMUseCase switches between root and recursive models."""

    def test_first_step_uses_root_model(self):
        """Step 1 should use root model."""
        llm = FakeTwoModelLLM(["FINAL: answer"])
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        uc.execute("content", "question")

        assert llm.model_history[0] == "root-model"

    def test_subsequent_steps_use_recursive_model(self):
        """Steps after step 1 should use recursive model."""
        llm = FakeTwoModelLLM(
            [
                "```python\nprint('exploring')\n```",
                "```python\nprint('more')\n```",
                "FINAL: done",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        uc.execute("content", "question")

        # Step 1 should be root, steps 2+ should be recursive
        assert llm.model_history[0] == "root-model"
        for model in llm.model_history[1:]:
            assert model == "recursive-model"

    def test_root_model_restored_after_final(self):
        """After finding FINAL answer, root model should be restored."""
        llm = FakeTwoModelLLM(
            [
                "```python\nprint('exploring')\n```",
                "FINAL: answer",
            ]
        )
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        uc.execute("content", "question")

        assert llm._active_model == "root-model"

    def test_root_model_restored_on_budget_exceeded(self):
        """Root model should be restored even on budget exhaustion."""
        llm = FakeTwoModelLLM(["```python\nprint(1)\n```"])
        sandbox = FakeSandbox()
        config = RunConfigDTO(mode="rlm", max_steps=2)
        uc = RunRLMUseCase(llm, sandbox)
        uc.execute("content", "question", config=config)

        assert llm._active_model == "root-model"

    def test_works_with_llm_without_model_switching(self):
        """Should work fine with LLMs that lack use_root/recursive_model."""

        class SimpleLLM:
            def complete(self, messages):
                return LLMResponseDTO(
                    content="FINAL: answer", model="simple", input_tokens=5, output_tokens=5
                )

            def complete_stream(self, messages):
                yield "FINAL: answer"

            def count_tokens(self, text):
                return 1

            def get_pricing(self):
                return {}

        sandbox = FakeSandbox()
        uc = RunRLMUseCase(SimpleLLM(), sandbox)
        result = uc.execute("content", "question")
        assert result.success is True
