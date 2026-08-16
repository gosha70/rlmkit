"""AC-19 — structural conformance of all five LLM adapters to the
updated LLMPort.complete_stream_async(...) -> AsyncIterator[StreamChunk]
signature.

Each adapter is driven with a minimal fake provider. Every adapter
must yield a terminal StreamChunk(is_final=True, response=<DTO>).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rlmstudio.application.dto import LLMResponseDTO, StreamChunk
from rlmstudio.infrastructure.llm.anthropic_adapter import AnthropicAdapter
from rlmstudio.infrastructure.llm.litellm_adapter import LiteLLMAdapter
from rlmstudio.infrastructure.llm.mock_adapter import MockLLMAdapter
from rlmstudio.infrastructure.llm.ollama_adapter import OllamaAdapter
from rlmstudio.infrastructure.llm.openai_adapter import OpenAIAdapter


class _StubClient:
    """Minimal legacy-style client that supports complete_with_metadata."""

    model = "stub-model"

    def complete_with_metadata(self, messages):
        return SimpleNamespace(
            content="hello",
            model="stub-model",
            input_tokens=9,
            output_tokens=4,
            finish_reason="stop",
        )

    # Some adapters fall back to .complete(messages) -> str if
    # complete_with_metadata is absent; both paths are covered
    # elsewhere. Keep the attribute so hasattr() checks succeed.


async def _collect(adapter, messages) -> list[StreamChunk]:
    result: list[StreamChunk] = []
    async for chunk in adapter.complete_stream_async(messages):
        result.append(chunk)
    return result


@pytest.mark.asyncio
async def test_mock_adapter_terminal_chunk_carries_dto():
    adapter = MockLLMAdapter(responses=["hi"])
    chunks = await _collect(adapter, [{"role": "user", "content": "?"}])

    assert chunks[-1].is_final is True
    assert chunks[-1].response is not None
    assert isinstance(chunks[-1].response, LLMResponseDTO)
    assert chunks[-1].response.content == "hi"
    # Mock adapter ships synthetic but populated telemetry.
    assert chunks[-1].response.ttft_ms == 1


@pytest.mark.asyncio
async def test_openai_adapter_terminal_chunk_carries_dto():
    adapter = OpenAIAdapter(client=_StubClient())
    chunks = await _collect(adapter, [{"role": "user", "content": "?"}])

    assert chunks[-1].is_final is True
    assert chunks[-1].response is not None
    assert isinstance(chunks[-1].response, LLMResponseDTO)
    assert chunks[-1].response.content == "hello"
    assert chunks[-1].response.input_tokens == 9
    assert chunks[-1].response.output_tokens == 4
    # Legacy adapter has no streaming backend -> ttft_ms is None.
    assert chunks[-1].response.ttft_ms is None


@pytest.mark.asyncio
async def test_anthropic_adapter_terminal_chunk_carries_dto():
    adapter = AnthropicAdapter(client=_StubClient())
    chunks = await _collect(adapter, [{"role": "user", "content": "?"}])

    assert chunks[-1].is_final is True
    assert chunks[-1].response is not None
    assert chunks[-1].response.content == "hello"
    assert chunks[-1].response.ttft_ms is None


@pytest.mark.asyncio
async def test_ollama_adapter_terminal_chunk_carries_dto():
    adapter = OllamaAdapter(client=_StubClient())
    chunks = await _collect(adapter, [{"role": "user", "content": "?"}])

    assert chunks[-1].is_final is True
    assert chunks[-1].response is not None
    assert chunks[-1].response.content == "hello"
    assert chunks[-1].response.ttft_ms is None


@pytest.mark.asyncio
async def test_litellm_adapter_terminal_chunk_carries_dto():
    content_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="hi", role=None),
                finish_reason=None,
                index=0,
            )
        ],
        model="gpt-4o",
        usage=None,
    )
    terminal = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None, role=None),
                finish_reason="stop",
                index=0,
            )
        ],
        model="gpt-4o",
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )

    async def mock_async_iter():
        yield content_chunk
        yield terminal

    adapter = LiteLLMAdapter(model="gpt-4o")
    with patch("litellm.acompletion", return_value=mock_async_iter()):
        chunks = await _collect(adapter, [{"role": "user", "content": "?"}])

    assert chunks[-1].is_final is True
    assert chunks[-1].response is not None
    assert isinstance(chunks[-1].response, LLMResponseDTO)
    assert chunks[-1].response.input_tokens == 5
    assert chunks[-1].response.output_tokens == 2
    # LiteLLM path measures real TTFT.
    assert chunks[-1].response.ttft_ms is not None
