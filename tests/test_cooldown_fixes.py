"""Tests for Cycle 1 cool-down fixes: MAJOR-1, MAJOR-2, MAJOR-3, MINOR-5."""

import asyncio
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from rlmkit.application.dto import ExecutionResultDTO, LLMResponseDTO, RunConfigDTO
from rlmkit.application.use_cases.run_rlm import RunRLMUseCase


# ---------------------------------------------------------------------------
# MAJOR-1: restricted_sandbox.py imports SecurityViolationError from domain
# ---------------------------------------------------------------------------


class TestMajor1DomainImport:
    """Verify restricted_sandbox uses domain exceptions, not core.errors."""

    def test_restricted_sandbox_imports_domain_exception(self):
        """restricted_sandbox should import SecurityViolationError from domain."""
        import rlmkit.infrastructure.sandbox.restricted_sandbox as mod

        # The module should reference SecurityViolationError from domain
        from rlmkit.domain.exceptions import SecurityViolationError

        assert hasattr(mod, "SecurityViolationError")
        assert mod.SecurityViolationError is SecurityViolationError

    def test_restricted_sandbox_does_not_import_core_errors(self):
        """restricted_sandbox should NOT import from rlmkit.core.errors."""
        import inspect
        import rlmkit.infrastructure.sandbox.restricted_sandbox as mod

        source = inspect.getsource(mod)
        assert "from rlmkit.core.errors" not in source


# ---------------------------------------------------------------------------
# MAJOR-2: Docker sandbox adapter wraps DockerExecutor as SandboxPort
# ---------------------------------------------------------------------------


class TestMajor2DockerSandboxAdapter:
    """Verify DockerSandboxAdapter implements SandboxPort protocol."""

    def test_adapter_exists(self):
        from rlmkit.infrastructure.sandbox.docker_sandbox_adapter import (
            DockerSandboxAdapter,
        )

        assert DockerSandboxAdapter is not None

    def test_adapter_exported_from_package(self):
        from rlmkit.infrastructure.sandbox import DockerSandboxAdapter

        assert DockerSandboxAdapter is not None

    @patch("rlmkit.envs.sandbox.DockerExecutor")
    def test_execute_returns_execution_result_dto(self, MockExecutor):
        mock_instance = MagicMock()
        mock_instance.execute.return_value = {
            "result": True,
            "output": "hello world",
            "error": None,
        }
        MockExecutor.return_value = mock_instance

        from rlmkit.infrastructure.sandbox.docker_sandbox_adapter import (
            DockerSandboxAdapter,
        )

        adapter = DockerSandboxAdapter.__new__(DockerSandboxAdapter)
        adapter._executor = mock_instance
        adapter._namespace = {}

        result = adapter.execute("print('hello world')")
        assert isinstance(result, ExecutionResultDTO)
        assert result.stdout == "hello world"
        assert result.exception is None

    @patch("rlmkit.envs.sandbox.DockerExecutor")
    def test_execute_handles_error(self, MockExecutor):
        mock_instance = MagicMock()
        mock_instance.execute.return_value = {
            "result": False,
            "output": "",
            "error": "NameError: name 'x' is not defined",
        }
        MockExecutor.return_value = mock_instance

        from rlmkit.infrastructure.sandbox.docker_sandbox_adapter import (
            DockerSandboxAdapter,
        )

        adapter = DockerSandboxAdapter.__new__(DockerSandboxAdapter)
        adapter._executor = mock_instance
        adapter._namespace = {}

        result = adapter.execute("print(x)")
        assert not result.success
        assert "NameError" in result.exception

    @patch("rlmkit.envs.sandbox.DockerExecutor")
    def test_execute_handles_timeout(self, MockExecutor):
        mock_instance = MagicMock()
        mock_instance.execute.return_value = {
            "result": False,
            "output": "",
            "error": "Execution timed out after 30 seconds",
        }
        MockExecutor.return_value = mock_instance

        from rlmkit.infrastructure.sandbox.docker_sandbox_adapter import (
            DockerSandboxAdapter,
        )

        adapter = DockerSandboxAdapter.__new__(DockerSandboxAdapter)
        adapter._executor = mock_instance
        adapter._namespace = {}

        result = adapter.execute("while True: pass")
        assert result.timeout is True

    @patch("rlmkit.envs.sandbox.DockerExecutor")
    def test_set_get_variable(self, MockExecutor):
        from rlmkit.infrastructure.sandbox.docker_sandbox_adapter import (
            DockerSandboxAdapter,
        )

        adapter = DockerSandboxAdapter.__new__(DockerSandboxAdapter)
        adapter._executor = MagicMock()
        adapter._namespace = {}

        adapter.set_variable("x", 42)
        assert adapter.get_variable("x") == 42
        assert adapter.get_variable("nonexistent") is None

    @patch("rlmkit.envs.sandbox.DockerExecutor")
    def test_reset_clears_namespace(self, MockExecutor):
        from rlmkit.infrastructure.sandbox.docker_sandbox_adapter import (
            DockerSandboxAdapter,
        )

        adapter = DockerSandboxAdapter.__new__(DockerSandboxAdapter)
        adapter._executor = MagicMock()
        adapter._namespace = {"x": 1}

        adapter.reset()
        assert adapter._namespace == {}

    def test_sandbox_factory_docker_type(self):
        """sandbox_factory should return DockerSandboxAdapter for 'docker' type."""
        from rlmkit.infrastructure.sandbox.sandbox_factory import create_sandbox

        with patch("rlmkit.envs.sandbox.DockerExecutor") as MockExecutor:
            MockExecutor.is_available.return_value = True
            MockExecutor.return_value = MagicMock()

            sandbox = create_sandbox(sandbox_type="docker")
            from rlmkit.infrastructure.sandbox.docker_sandbox_adapter import (
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
        from rlmkit.application.ports.llm_port import LLMPort

        assert hasattr(LLMPort, "complete_async")
        assert inspect.iscoroutinefunction(LLMPort.complete_async)

    def test_llm_port_has_complete_stream_async(self):
        import inspect
        from rlmkit.application.ports.llm_port import LLMPort

        assert hasattr(LLMPort, "complete_stream_async")
        assert inspect.isasyncgenfunction(LLMPort.complete_stream_async)

    def test_sandbox_port_has_execute_async(self):
        import inspect
        from rlmkit.application.ports.sandbox_port import SandboxPort

        assert hasattr(SandboxPort, "execute_async")
        assert inspect.iscoroutinefunction(SandboxPort.execute_async)

    def test_litellm_adapter_has_complete_async(self):
        import inspect
        from rlmkit.infrastructure.llm.litellm_adapter import LiteLLMAdapter

        assert hasattr(LiteLLMAdapter, "complete_async")
        assert inspect.iscoroutinefunction(LiteLLMAdapter.complete_async)

    def test_litellm_adapter_has_complete_stream_async(self):
        import inspect
        from rlmkit.infrastructure.llm.litellm_adapter import LiteLLMAdapter

        assert hasattr(LiteLLMAdapter, "complete_stream_async")
        assert inspect.isasyncgenfunction(LiteLLMAdapter.complete_stream_async)

    @pytest.mark.asyncio
    async def test_litellm_complete_async_calls_acompletion(self):
        """LiteLLMAdapter.complete_async should call litellm.acompletion."""
        from rlmkit.infrastructure.llm.litellm_adapter import LiteLLMAdapter

        adapter = LiteLLMAdapter(model="gpt-4o")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "async answer"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        with patch("litellm.acompletion", return_value=mock_response) as mock_ac:
            result = await adapter.complete_async(
                [{"role": "user", "content": "hello"}]
            )
            mock_ac.assert_called_once()
            assert isinstance(result, LLMResponseDTO)
            assert result.content == "async answer"
            assert result.input_tokens == 10
            assert result.output_tokens == 5

    @pytest.mark.asyncio
    async def test_litellm_complete_stream_async_yields_chunks(self):
        """LiteLLMAdapter.complete_stream_async should yield text chunks."""
        from rlmkit.infrastructure.llm.litellm_adapter import LiteLLMAdapter

        adapter = LiteLLMAdapter(model="gpt-4o")

        # Build mock async iterator
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = " World"

        async def mock_async_iter():
            for c in [chunk1, chunk2]:
                yield c

        with patch("litellm.acompletion", return_value=mock_async_iter()):
            collected = []
            async for text in adapter.complete_stream_async(
                [{"role": "user", "content": "hi"}]
            ):
                collected.append(text)

            assert collected == ["Hello", " World"]


# ---------------------------------------------------------------------------
# MINOR-5: Two-model switching wired into RunRLMUseCase
# ---------------------------------------------------------------------------


class FakeTwoModelLLM:
    """LLM fake that tracks model-switching calls."""

    def __init__(self, responses: List[str]) -> None:
        self._responses = responses
        self._idx = 0
        self._active_model = "root-model"
        self.model_history: List[str] = []

    def complete(self, messages: List[Dict[str, str]]) -> LLMResponseDTO:
        self.model_history.append(self._active_model)
        idx = min(self._idx, len(self._responses) - 1)
        text = self._responses[idx]
        self._idx += 1
        return LLMResponseDTO(content=text, model=self._active_model,
                               input_tokens=10, output_tokens=5)

    def complete_stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        yield self.complete(messages).content

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def get_pricing(self) -> Dict[str, float]:
        return {"input_cost_per_1m": 0.0, "output_cost_per_1m": 0.0}

    def use_root_model(self) -> None:
        self._active_model = "root-model"

    def use_recursive_model(self) -> None:
        self._active_model = "recursive-model"


class FakeSandbox:
    """Minimal SandboxPort-compliant fake."""

    def __init__(self) -> None:
        self._namespace: Dict[str, Any] = {}

    def execute(self, code: str) -> ExecutionResultDTO:
        return ExecutionResultDTO(stdout="ok")

    def reset(self) -> None:
        self._namespace.clear()

    def is_healthy(self) -> bool:
        return True

    def set_variable(self, name: str, value: Any) -> None:
        self._namespace[name] = value

    def get_variable(self, name: str) -> Optional[Any]:
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
        llm = FakeTwoModelLLM([
            "```python\nprint('exploring')\n```",
            "```python\nprint('more')\n```",
            "FINAL: done",
        ])
        sandbox = FakeSandbox()
        uc = RunRLMUseCase(llm, sandbox)
        uc.execute("content", "question")

        # Step 1 should be root, steps 2+ should be recursive
        assert llm.model_history[0] == "root-model"
        for model in llm.model_history[1:]:
            assert model == "recursive-model"

    def test_root_model_restored_after_final(self):
        """After finding FINAL answer, root model should be restored."""
        llm = FakeTwoModelLLM([
            "```python\nprint('exploring')\n```",
            "FINAL: answer",
        ])
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
                return LLMResponseDTO(content="FINAL: answer", model="simple",
                                       input_tokens=5, output_tokens=5)
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
