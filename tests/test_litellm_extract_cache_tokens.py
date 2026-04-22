"""Phase 2 — AC-3 tests for `_extract_cache_tokens` in LiteLLMAdapter.

Covers the three provider schemas documented in the spec:
- Anthropic (`cache_read_input_tokens` + `cache_creation_input_tokens`)
- OpenAI (`prompt_tokens_details.cached_tokens`; no distinct write)
- Unknown providers (returns (0, 0))

Also covers the wiring into `_observe_chunk` so the accumulated
telemetry carries the extracted counts onto the returned DTO.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from rlmkit.infrastructure.llm.litellm_adapter import (
    LiteLLMAdapter,
    _extract_cache_tokens,
)


class TestExtractCacheTokens:
    def test_returns_zero_zero_for_none_usage(self):
        assert _extract_cache_tokens(None) == (0, 0)

    def test_anthropic_schema(self):
        usage = SimpleNamespace(
            cache_read_input_tokens=120,
            cache_creation_input_tokens=30,
        )
        assert _extract_cache_tokens(usage) == (120, 30)

    def test_anthropic_read_only(self):
        usage = SimpleNamespace(
            cache_read_input_tokens=45,
            cache_creation_input_tokens=0,
        )
        assert _extract_cache_tokens(usage) == (45, 0)

    def test_openai_schema_via_prompt_tokens_details(self):
        usage = SimpleNamespace(
            prompt_tokens_details=SimpleNamespace(cached_tokens=88),
        )
        read, write = _extract_cache_tokens(usage)
        assert read == 88
        assert write == 0  # OpenAI has no distinct write counter

    def test_unknown_provider_returns_zero_zero(self):
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=25,
            # no cache fields at all
        )
        assert _extract_cache_tokens(usage) == (0, 0)

    def test_does_not_raise_on_missing_fields(self):
        usage = SimpleNamespace()
        assert _extract_cache_tokens(usage) == (0, 0)


class TestCacheTokensFlowIntoDTO:
    """Verify _observe_chunk surfaces cache tokens on the returned DTO."""

    @patch("litellm.completion")
    def test_complete_populates_cache_tokens_from_anthropic_usage(self, mock_completion):
        usage = SimpleNamespace(
            prompt_tokens=200,
            completion_tokens=50,
            cache_read_input_tokens=150,
            cache_creation_input_tokens=10,
        )
        content_chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="ok", role=None),
                    finish_reason=None,
                    index=0,
                )
            ],
            model="claude-3-5-sonnet",
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
            model="claude-3-5-sonnet",
            usage=usage,
        )
        mock_completion.return_value = iter([content_chunk, terminal])

        adapter = LiteLLMAdapter(model="claude-3-5-sonnet")
        result = adapter.complete([{"role": "user", "content": "?"}])

        assert result.cached_tokens == 150
        assert result.cache_write_tokens == 10
        assert result.input_tokens == 200
        assert result.output_tokens == 50
