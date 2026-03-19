"""Tests for the unified interact() API (src/rlmkit/api.py).

Uses monkeypatching to avoid real LLM calls while exercising the full
dispatch logic: auto mode selection, provider resolution, and use-case
wiring.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from rlmkit.api import (
    InteractResult,
    _auto_detect_provider,
    _determine_auto_mode,
    _estimate_tokens,
    _resolve_model,
    complete,
    complete_async,
    interact,
    interact_async,
)
from rlmkit.application.dto import RunResultDTO

# ---------------------------------------------------------------------------
# _estimate_tokens / _determine_auto_mode
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_short_text(self):
        assert _estimate_tokens("abcd") == 1

    def test_empty_text(self):
        assert _estimate_tokens("") == 1  # max(1, 0)

    def test_longer_text(self):
        assert _estimate_tokens("a" * 100) == 25


class TestDetermineAutoMode:
    def test_short_selects_direct(self):
        assert _determine_auto_mode("a" * 1000) == "direct"

    def test_medium_selects_rlm(self):
        # RAG tier removed — medium content goes to rlm
        assert _determine_auto_mode("a" * 40_000) == "rlm"

    def test_large_selects_rlm(self):
        assert _determine_auto_mode("a" * 500_000) == "rlm"


# ---------------------------------------------------------------------------
# _auto_detect_provider
# ---------------------------------------------------------------------------


class TestAutoDetectProvider:
    def test_openai_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        assert _auto_detect_provider() == "openai"

    def test_anthropic_from_env(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        assert _auto_detect_provider() == "anthropic"

    def test_none_when_no_env(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        assert _auto_detect_provider() is None


# ---------------------------------------------------------------------------
# _resolve_model
# ---------------------------------------------------------------------------


class TestResolveModel:
    def test_default_openai(self):
        assert _resolve_model("openai", None) == "gpt-4o"

    def test_anthropic_prefix(self):
        result = _resolve_model("anthropic", "claude-3-haiku")
        assert result == "anthropic/claude-3-haiku"

    def test_ollama_prefix(self):
        result = _resolve_model("ollama", "llama3")
        assert result == "ollama/llama3"

    def test_already_prefixed(self):
        result = _resolve_model("openai", "openai/gpt-4o")
        assert result == "openai/gpt-4o"


# ---------------------------------------------------------------------------
# interact() — mocked LLM
# ---------------------------------------------------------------------------

_FAKE_RESULT = RunResultDTO(
    answer="Test answer",
    mode_used="direct",
    success=True,
    steps=1,
    input_tokens=10,
    output_tokens=5,
    total_cost=0.001,
    elapsed_time=0.5,
    trace=[{"step": 0, "role": "assistant", "content": "Test answer"}],
)


class TestInteract:
    def test_validation_empty_content(self):
        with pytest.raises(ValueError, match="content cannot be empty"):
            interact("", "question")

    def test_validation_empty_query(self):
        with pytest.raises(ValueError, match="query cannot be empty"):
            interact("content", "")

    def test_no_provider_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        with pytest.raises(ValueError, match="No LLM provider configured"):
            interact("content", "question")

    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_direct_mode(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute.return_value = _FAKE_RESULT
        result = interact("content", "question", mode="direct", provider="openai")

        assert isinstance(result, InteractResult)
        assert result.answer == "Test answer"
        assert result.mode_used == "direct"
        assert result.metrics["total_tokens"] == 15
        mock_uc_cls.return_value.execute.assert_called_once()

    @patch("rlmkit.api.RunRLMUseCase")
    @patch("rlmkit.api.create_sandbox")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_rlm_mode(self, mock_adapter_cls, mock_sandbox_fn, mock_uc_cls):
        rlm_result = RunResultDTO(
            answer="RLM answer",
            mode_used="rlm",
            success=True,
            steps=3,
            input_tokens=50,
            output_tokens=20,
        )
        mock_uc_cls.return_value.execute.return_value = rlm_result
        result = interact("content", "question", mode="rlm", provider="openai")

        assert result.mode_used == "rlm"
        assert result.answer == "RLM answer"
        mock_sandbox_fn.assert_called_once()

    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_auto_mode_selects_direct_for_short(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute.return_value = _FAKE_RESULT
        result = interact("short", "question", mode="auto", provider="openai")
        assert result.mode_used == "direct"

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            interact("content", "question", mode="invalid", provider="openai")  # type: ignore[arg-type]

    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_metrics_populated(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute.return_value = _FAKE_RESULT
        result = interact("content", "question", mode="direct", provider="openai")

        assert "total_tokens" in result.metrics
        assert "input_tokens" in result.metrics
        assert "output_tokens" in result.metrics
        assert "total_cost" in result.metrics
        assert "execution_time" in result.metrics
        assert "llm_calls" in result.metrics

    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_raw_result_is_run_result_dto(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute.return_value = _FAKE_RESULT
        result = interact("content", "question", mode="direct", provider="openai")

        assert isinstance(result.raw_result, RunResultDTO)
        assert result.answer == result.raw_result.answer

    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_verbose_prints(self, mock_adapter_cls, mock_uc_cls, capsys):
        mock_uc_cls.return_value.execute.return_value = _FAKE_RESULT
        interact("content", "question", mode="auto", provider="openai", verbose=True)

        captured = capsys.readouterr()
        assert "Auto Mode" in captured.out
        assert "Setup" in captured.out
        assert "Complete" in captured.out

    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_auto_detect_from_env(self, mock_adapter_cls, mock_uc_cls, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        mock_uc_cls.return_value.execute.return_value = _FAKE_RESULT

        result = interact("content", "question", mode="direct")
        assert result.answer == "Test answer"

    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_interact_result_str(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute.return_value = _FAKE_RESULT
        result = interact("content", "question", mode="direct", provider="openai")
        assert str(result) == "Test answer"

    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_interact_result_to_dict(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute.return_value = _FAKE_RESULT
        result = interact("content", "question", mode="direct", provider="openai")
        d = result.to_dict()
        assert d["answer"] == "Test answer"
        assert d["mode_used"] == "direct"
        assert d["has_trace"] is True


# ---------------------------------------------------------------------------
# complete() convenience wrapper
# ---------------------------------------------------------------------------


class TestComplete:
    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_returns_string(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute.return_value = _FAKE_RESULT
        answer = complete("content", "question", provider="openai", mode="direct")
        assert answer == "Test answer"
        assert isinstance(answer, str)


# ---------------------------------------------------------------------------
# Default model updates
# ---------------------------------------------------------------------------


class TestDefaultModels:
    def test_default_anthropic_model(self):
        result = _resolve_model("anthropic", None)
        assert result == "anthropic/claude-sonnet-4-5-20250514"


# ---------------------------------------------------------------------------
# api_base and timeout parameters
# ---------------------------------------------------------------------------


class TestApiBaseAndTimeout:
    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_api_base_passed_to_adapter(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute.return_value = _FAKE_RESULT
        interact(
            "content",
            "question",
            mode="direct",
            provider="ollama",
            api_base="http://localhost:11434",
        )
        _, kwargs = mock_adapter_cls.call_args
        assert kwargs["api_base"] == "http://localhost:11434"

    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_timeout_passed_to_adapter(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute.return_value = _FAKE_RESULT
        interact(
            "content",
            "question",
            mode="direct",
            provider="openai",
            timeout=30.0,
        )
        _, kwargs = mock_adapter_cls.call_args
        assert kwargs["timeout"] == 30.0

    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_default_timeout(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute.return_value = _FAKE_RESULT
        interact("content", "question", mode="direct", provider="openai")
        _, kwargs = mock_adapter_cls.call_args
        assert kwargs["timeout"] == 120.0


# ---------------------------------------------------------------------------
# interact_async / complete_async
# ---------------------------------------------------------------------------


class TestInteractAsync:
    @pytest.mark.asyncio
    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    async def test_async_direct_mode(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute_async = AsyncMock(return_value=_FAKE_RESULT)
        result = await interact_async(
            "content",
            "question",
            mode="direct",
            provider="openai",
        )
        assert isinstance(result, InteractResult)
        assert result.answer == "Test answer"
        assert result.mode_used == "direct"
        mock_uc_cls.return_value.execute_async.assert_called_once()

    @pytest.mark.asyncio
    @patch("rlmkit.api.RunRLMUseCase")
    @patch("rlmkit.api.create_sandbox")
    @patch("rlmkit.api.LiteLLMAdapter")
    async def test_async_rlm_mode(self, mock_adapter_cls, mock_sandbox_fn, mock_uc_cls):
        rlm_result = RunResultDTO(
            answer="Async RLM",
            mode_used="rlm",
            success=True,
            steps=2,
            input_tokens=40,
            output_tokens=15,
        )
        mock_uc_cls.return_value.execute_async = AsyncMock(return_value=rlm_result)
        result = await interact_async(
            "content",
            "question",
            mode="rlm",
            provider="openai",
        )
        assert result.mode_used == "rlm"
        assert result.answer == "Async RLM"
        mock_sandbox_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_validation(self):
        with pytest.raises(ValueError, match="content cannot be empty"):
            await interact_async("", "question")

    @pytest.mark.asyncio
    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    async def test_async_api_base_and_timeout(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute_async = AsyncMock(return_value=_FAKE_RESULT)
        await interact_async(
            "content",
            "question",
            mode="direct",
            provider="ollama",
            api_base="http://myhost:8080",
            timeout=15.0,
        )
        _, kwargs = mock_adapter_cls.call_args
        assert kwargs["api_base"] == "http://myhost:8080"
        assert kwargs["timeout"] == 15.0


class TestCompleteAsync:
    @pytest.mark.asyncio
    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    async def test_returns_string(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute_async = AsyncMock(return_value=_FAKE_RESULT)
        answer = await complete_async(
            "content",
            "question",
            provider="openai",
            mode="direct",
        )
        assert answer == "Test answer"
        assert isinstance(answer, str)


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------


class TestCompareMode:
    @patch("rlmkit.api.RunComparisonUseCase")
    @patch("rlmkit.api.create_sandbox")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_compare_mode_returns_result(self, mock_adapter_cls, mock_sandbox_fn, mock_uc_cls):
        from rlmkit.application.use_cases.run_comparison import ComparisonResultDTO

        cmp = ComparisonResultDTO(
            results={
                "direct": RunResultDTO(answer="Direct answer", mode_used="direct", success=True),
                "rlm": RunResultDTO(
                    answer="RLM answer",
                    mode_used="rlm",
                    success=True,
                    input_tokens=30,
                    output_tokens=20,
                ),
            },
            total_elapsed=1.5,
        )
        mock_uc_cls.return_value.execute.return_value = cmp
        result = interact("content", "question", mode="compare", provider="openai")

        assert result.mode_used == "compare"
        assert result.answer == "RLM answer"  # prefers rlm
        assert "comparison" in result.metrics
        assert "direct" in result.metrics["comparison"]
        assert "rlm" in result.metrics["comparison"]
        mock_sandbox_fn.assert_called_once()

    @patch("rlmkit.api.RunComparisonUseCase")
    @patch("rlmkit.api.create_sandbox")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_compare_aggregates_tokens(self, mock_adapter_cls, mock_sandbox_fn, mock_uc_cls):
        from rlmkit.application.use_cases.run_comparison import ComparisonResultDTO

        cmp = ComparisonResultDTO(
            results={
                "direct": RunResultDTO(
                    answer="A",
                    mode_used="direct",
                    success=True,
                    input_tokens=5,
                    output_tokens=5,
                    total_cost=0.01,
                ),
                "rlm": RunResultDTO(
                    answer="B",
                    mode_used="rlm",
                    success=True,
                    input_tokens=20,
                    output_tokens=20,
                    total_cost=0.04,
                ),
            },
            total_elapsed=2.0,
        )
        mock_uc_cls.return_value.execute.return_value = cmp
        result = interact("content", "question", mode="compare", provider="openai")

        assert result.metrics["total_tokens"] == 50
        assert result.metrics["total_cost"] == 0.05


# ---------------------------------------------------------------------------
# Two-model params (root_model / recursive_model)
# ---------------------------------------------------------------------------


class TestTwoModelParams:
    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_root_model_passed_to_adapter(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute.return_value = _FAKE_RESULT
        interact(
            "content",
            "question",
            mode="direct",
            provider="openai",
            root_model="gpt-4o",
            recursive_model="gpt-4o-mini",
        )
        _, kwargs = mock_adapter_cls.call_args
        assert kwargs["root_model"] == "gpt-4o"
        assert kwargs["recursive_model"] == "gpt-4o-mini"

    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_none_by_default(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute.return_value = _FAKE_RESULT
        interact("content", "question", mode="direct", provider="openai")
        _, kwargs = mock_adapter_cls.call_args
        assert kwargs["root_model"] is None
        assert kwargs["recursive_model"] is None


# ---------------------------------------------------------------------------
# RLMKitClient deprecation
# ---------------------------------------------------------------------------


class TestRLMKitClientDeprecation:
    def test_emits_deprecation_warning(self):
        import warnings

        from rlmkit.public import RLMKitClient

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RLMKitClient(provider="mock")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()


# ---------------------------------------------------------------------------
# Explicit rag mode still works (falls back to direct)
# ---------------------------------------------------------------------------


class TestExplicitRagMode:
    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_rag_mode_returns_result(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute.return_value = _FAKE_RESULT
        result = interact("content", "question", mode="rag", provider="openai")
        assert result.mode_used == "rag"
        assert result.answer == "Test answer"


# ---------------------------------------------------------------------------
# Compare mode: failed rlm falls back to successful direct
# ---------------------------------------------------------------------------


class TestCompareFailedRlmFallback:
    @patch("rlmkit.api.RunComparisonUseCase")
    @patch("rlmkit.api.create_sandbox")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_prefers_successful_direct_over_failed_rlm(
        self, mock_adapter_cls, mock_sandbox_fn, mock_uc_cls
    ):
        from rlmkit.application.use_cases.run_comparison import ComparisonResultDTO

        cmp = ComparisonResultDTO(
            results={
                "direct": RunResultDTO(
                    answer="Good direct answer", mode_used="direct", success=True
                ),
                "rlm": RunResultDTO(
                    answer="", mode_used="rlm", success=False, error="Budget exceeded"
                ),
            },
            total_elapsed=3.0,
        )
        mock_uc_cls.return_value.execute.return_value = cmp
        result = interact("content", "question", mode="compare", provider="openai")

        assert result.answer == "Good direct answer"
        assert result.mode_used == "compare"


# ---------------------------------------------------------------------------
# PublicInteractResult deprecation alias
# ---------------------------------------------------------------------------


class TestPublicInteractResultAlias:
    def test_import_emits_deprecation(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from rlmkit.public.types import PublicInteractResult  # noqa: F811

            assert len(w) >= 1
            assert any(
                issubclass(x.category, DeprecationWarning)
                and "PublicInteractResult" in str(x.message)
                for x in w
            )
            # It should resolve to InteractResult
            from rlmkit.api import InteractResult

            assert PublicInteractResult is InteractResult


# ---------------------------------------------------------------------------
# Two-model provider normalization
# ---------------------------------------------------------------------------


class TestTwoModelNormalization:
    @patch("rlmkit.api.RunDirectUseCase")
    @patch("rlmkit.api.LiteLLMAdapter")
    def test_anthropic_root_model_gets_prefix(self, mock_adapter_cls, mock_uc_cls):
        mock_uc_cls.return_value.execute.return_value = _FAKE_RESULT
        interact(
            "content",
            "question",
            mode="direct",
            provider="anthropic",
            root_model="claude-sonnet-4-5-20250514",
            recursive_model="claude-haiku-4-5-20251001",
        )
        _, kwargs = mock_adapter_cls.call_args
        assert kwargs["root_model"] == "anthropic/claude-sonnet-4-5-20250514"
        assert kwargs["recursive_model"] == "anthropic/claude-haiku-4-5-20251001"
