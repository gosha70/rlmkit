"""Tests for the unified interact() API (src/rlmkit/api.py).

Uses monkeypatching to avoid real LLM calls while exercising the full
dispatch logic: auto mode selection, provider resolution, and use-case
wiring.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rlmkit.api import (
    InteractResult,
    _auto_detect_provider,
    _determine_auto_mode,
    _estimate_tokens,
    _resolve_model,
    complete,
    interact,
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

    def test_medium_selects_rag(self):
        assert _determine_auto_mode("a" * 40_000) == "rag"

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
