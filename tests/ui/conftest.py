"""Shared fixtures for ui test infrastructure."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rlmstudio.llm.base import LLMResponse
from rlmstudio.ui.services import ExecutionMetrics, Response
from rlmstudio.ui.services.models import LLMProviderConfig


@pytest.fixture
def fake_provider_config() -> LLMProviderConfig:
    return LLMProviderConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-test",
        input_cost_per_1k_tokens=0.01,
        output_cost_per_1k_tokens=0.03,
    )


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock LLM client that returns realistic non-zero responses."""
    client = MagicMock()
    client.complete.return_value = "FINAL: Mock answer"
    client.complete_with_metadata.return_value = LLMResponse(
        content="FINAL: Mock answer",
        model="mock",
        input_tokens=50,
        output_tokens=30,
        finish_reason="stop",
    )
    return client


def make_fake_rlm_result() -> dict:
    return {
        "response": Response(content="RLM mock answer", stop_reason="stop"),
        "metrics": ExecutionMetrics(
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
            cost_usd=0.05,
            cost_breakdown={"input": 0.03, "output": 0.02},
            execution_time_seconds=1.5,
            steps_taken=3,
            memory_used_mb=40.0,
            memory_peak_mb=55.0,
            success=True,
            execution_type="rlm",
        ),
        "trace": [{"step": 1, "action": "FINAL", "input_tokens": 200, "output_tokens": 100}],
    }


def make_fake_direct_result() -> dict:
    return {
        "response": Response(content="Direct mock answer", stop_reason="stop"),
        "metrics": ExecutionMetrics(
            input_tokens=50,
            output_tokens=30,
            total_tokens=80,
            cost_usd=0.01,
            cost_breakdown={"input": 0.006, "output": 0.004},
            execution_time_seconds=0.4,
            steps_taken=0,
            memory_used_mb=10.0,
            memory_peak_mb=12.0,
            success=True,
            execution_type="direct",
        ),
    }


@pytest.fixture
def mock_execute_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch _execute_rlm and _execute_direct on ChatManager to return fake results."""
    from rlmstudio.ui.services.chat_manager import ChatManager

    async def fake_rlm(self: object, *args: object, **kwargs: object) -> dict:
        return make_fake_rlm_result()

    async def fake_direct(self: object, *args: object, **kwargs: object) -> dict:
        return make_fake_direct_result()

    monkeypatch.setattr(ChatManager, "_execute_rlm", fake_rlm)
    monkeypatch.setattr(ChatManager, "_execute_direct", fake_direct)
