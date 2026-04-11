"""Unit tests for the LiteLLM adapter.

All litellm calls are mocked -- no actual API calls are made.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rlmkit.application.dto import LLMResponseDTO
from rlmkit.infrastructure.llm.litellm_adapter import LiteLLMAdapter

# ---------------------------------------------------------------------------
# Helpers to build mock litellm responses
# ---------------------------------------------------------------------------


def _mock_completion_response(
    content: str = "Hello!",
    model: str = "gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
):
    """Build a mock object mimicking litellm.completion() return value."""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = SimpleNamespace(content=content, role="assistant")
    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
        index=0,
    )
    return SimpleNamespace(
        choices=[choice],
        usage=usage,
        model=model,
        id="chatcmpl-test",
    )


def _mock_stream_chunks(texts):
    """Build a list of mock streaming chunks."""
    chunks = []
    for t in texts:
        delta = SimpleNamespace(content=t, role=None)
        choice = SimpleNamespace(delta=delta, index=0, finish_reason=None)
        chunks.append(SimpleNamespace(choices=[choice]))
    # Final chunk with finish_reason
    final_delta = SimpleNamespace(content=None, role=None)
    final_choice = SimpleNamespace(delta=final_delta, index=0, finish_reason="stop")
    chunks.append(SimpleNamespace(choices=[final_choice]))
    return chunks


# ---------------------------------------------------------------------------
# Tests: Construction
# ---------------------------------------------------------------------------


class TestLiteLLMAdapterConstruction:
    """Test adapter initialization and configuration."""

    def test_default_model(self):
        adapter = LiteLLMAdapter()
        assert adapter.active_model == "gpt-4o"
        assert adapter.root_model == "gpt-4o"
        assert adapter.recursive_model == "gpt-4o"
        assert adapter.is_two_model is False

    def test_custom_model(self):
        adapter = LiteLLMAdapter(model="claude-3-opus-20240229")
        assert adapter.active_model == "claude-3-opus-20240229"

    def test_two_model_config(self):
        adapter = LiteLLMAdapter(
            model="gpt-4o",
            root_model="gpt-4o",
            recursive_model="gpt-4o-mini",
        )
        assert adapter.root_model == "gpt-4o"
        assert adapter.recursive_model == "gpt-4o-mini"
        assert adapter.is_two_model is True
        # Default active model is root
        assert adapter.active_model == "gpt-4o"

    def test_model_switching(self):
        adapter = LiteLLMAdapter(
            root_model="gpt-4o",
            recursive_model="gpt-4o-mini",
        )
        assert adapter.active_model == "gpt-4o"

        adapter.use_recursive_model()
        assert adapter.active_model == "gpt-4o-mini"

        adapter.use_root_model()
        assert adapter.active_model == "gpt-4o"

    def test_repr_single_model(self):
        adapter = LiteLLMAdapter(model="gpt-4o")
        assert "gpt-4o" in repr(adapter)

    def test_repr_two_model(self):
        adapter = LiteLLMAdapter(root_model="gpt-4o", recursive_model="gpt-4o-mini")
        r = repr(adapter)
        assert "root=" in r
        assert "recursive=" in r

    def test_default_num_retries(self):
        adapter = LiteLLMAdapter()
        assert adapter._num_retries == 2

    def test_custom_num_retries(self):
        adapter = LiteLLMAdapter(num_retries=5)
        assert adapter._num_retries == 5

    def test_num_retries_in_build_params(self):
        adapter = LiteLLMAdapter(num_retries=3)
        params = adapter._build_params([{"role": "user", "content": "hi"}])
        assert params["num_retries"] == 3


# ---------------------------------------------------------------------------
# Tests: complete()
# ---------------------------------------------------------------------------


class TestLiteLLMAdapterComplete:
    """Test synchronous completion."""

    @patch("litellm.completion")
    def test_complete_basic(self, mock_completion):
        mock_completion.return_value = _mock_completion_response(
            content="The answer is 42.",
            model="gpt-4o",
            prompt_tokens=20,
            completion_tokens=10,
        )

        adapter = LiteLLMAdapter(model="gpt-4o")
        messages = [{"role": "user", "content": "What is 6*7?"}]
        result = adapter.complete(messages)

        assert isinstance(result, LLMResponseDTO)
        assert result.content == "The answer is 42."
        assert result.model == "gpt-4o"
        assert result.input_tokens == 20
        assert result.output_tokens == 10
        assert result.finish_reason == "stop"

        # Verify litellm.completion was called with correct params
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["messages"] == messages

    @patch("litellm.completion")
    def test_complete_with_api_key(self, mock_completion):
        mock_completion.return_value = _mock_completion_response()

        adapter = LiteLLMAdapter(model="gpt-4o", api_key="sk-test-key")
        adapter.complete([{"role": "user", "content": "Hi"}])

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["api_key"] == "sk-test-key"

    @patch("litellm.completion")
    def test_complete_with_api_base(self, mock_completion):
        mock_completion.return_value = _mock_completion_response()

        adapter = LiteLLMAdapter(model="gpt-4o", api_base="http://localhost:8080")
        adapter.complete([{"role": "user", "content": "Hi"}])

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["api_base"] == "http://localhost:8080"

    @patch("litellm.completion")
    def test_complete_with_max_tokens(self, mock_completion):
        mock_completion.return_value = _mock_completion_response()

        adapter = LiteLLMAdapter(model="gpt-4o", max_tokens=500)
        adapter.complete([{"role": "user", "content": "Hi"}])

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["max_tokens"] == 500

    @patch("litellm.completion")
    def test_complete_with_temperature(self, mock_completion):
        mock_completion.return_value = _mock_completion_response()

        adapter = LiteLLMAdapter(model="gpt-4o", temperature=0.2)
        adapter.complete([{"role": "user", "content": "Hi"}])

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.2

    @patch("litellm.completion")
    def test_complete_with_extra_params(self, mock_completion):
        mock_completion.return_value = _mock_completion_response()

        adapter = LiteLLMAdapter(
            model="gpt-4o",
            extra_params={"top_p": 0.9, "seed": 42},
        )
        adapter.complete([{"role": "user", "content": "Hi"}])

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["seed"] == 42

    @patch("litellm.completion")
    def test_complete_error_raises_runtime_error(self, mock_completion):
        mock_completion.side_effect = Exception("API key invalid")

        adapter = LiteLLMAdapter(model="gpt-4o")
        with pytest.raises(RuntimeError, match="LiteLLM completion failed"):
            adapter.complete([{"role": "user", "content": "Hi"}])

    @patch("litellm.completion")
    def test_complete_uses_active_model(self, mock_completion):
        """Verify that switching models changes which model is called."""
        mock_completion.return_value = _mock_completion_response(model="gpt-4o-mini")

        adapter = LiteLLMAdapter(
            root_model="gpt-4o",
            recursive_model="gpt-4o-mini",
        )
        adapter.use_recursive_model()
        adapter.complete([{"role": "user", "content": "Hi"}])

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"

    @patch("litellm.completion")
    def test_complete_handles_none_content(self, mock_completion):
        """Handle case where LLM returns None content."""
        mock_completion.return_value = _mock_completion_response(content=None)
        # Need to manually set content to None since helper sets it
        mock_completion.return_value.choices[0].message.content = None

        adapter = LiteLLMAdapter(model="gpt-4o")
        result = adapter.complete([{"role": "user", "content": "Hi"}])
        assert result.content == ""


# ---------------------------------------------------------------------------
# Tests: complete_stream()
# ---------------------------------------------------------------------------


class TestLiteLLMAdapterStream:
    """Test streaming completion."""

    @patch("litellm.completion")
    def test_stream_basic(self, mock_completion):
        chunks = _mock_stream_chunks(["Hello", ", ", "world", "!"])
        mock_completion.return_value = iter(chunks)

        adapter = LiteLLMAdapter(model="gpt-4o")
        messages = [{"role": "user", "content": "Hi"}]
        collected = list(adapter.complete_stream(messages))

        assert collected == ["Hello", ", ", "world", "!"]

        # Verify stream=True was passed
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["stream"] is True

    @patch("litellm.completion")
    def test_stream_error_raises_runtime_error(self, mock_completion):
        mock_completion.side_effect = Exception("Stream failed")

        adapter = LiteLLMAdapter(model="gpt-4o")
        with pytest.raises(RuntimeError, match="LiteLLM streaming failed"):
            list(adapter.complete_stream([{"role": "user", "content": "Hi"}]))


# ---------------------------------------------------------------------------
# Tests: count_tokens()
# ---------------------------------------------------------------------------


class TestLiteLLMAdapterTokenCount:
    """Test token counting."""

    @patch("litellm.token_counter")
    def test_count_tokens_uses_litellm(self, mock_counter):
        mock_counter.return_value = 42

        adapter = LiteLLMAdapter(model="gpt-4o")
        count = adapter.count_tokens("Hello, how are you?")

        assert count == 42
        mock_counter.assert_called_once_with(model="gpt-4o", text="Hello, how are you?")

    @patch("litellm.token_counter")
    def test_count_tokens_fallback_on_error(self, mock_counter):
        mock_counter.side_effect = Exception("Tokenizer unavailable")

        adapter = LiteLLMAdapter(model="gpt-4o")
        count = adapter.count_tokens("Hello world, this is a test")

        # Fallback: len // 4
        assert count == max(1, len("Hello world, this is a test") // 4)

    @patch("litellm.token_counter")
    def test_count_tokens_messages_overload(self, mock_counter):
        """count_tokens(messages=...) dispatches to litellm with messages kwarg."""
        mock_counter.return_value = 123

        adapter = LiteLLMAdapter(model="gpt-4o")
        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hi"},
        ]
        count = adapter.count_tokens(messages=messages)

        assert count == 123
        mock_counter.assert_called_once_with(model="gpt-4o", messages=messages)

    @patch("litellm.token_counter")
    def test_count_tokens_messages_fallback_on_error(self, mock_counter):
        """On tokenizer failure, messages overload falls back to char heuristic."""
        mock_counter.side_effect = Exception("Tokenizer unavailable")

        adapter = LiteLLMAdapter(model="gpt-4o")
        messages = [
            {"role": "user", "content": "abcdefghij"},  # 10 chars
            {"role": "assistant", "content": "klmnop"},  # 6 chars
        ]
        count = adapter.count_tokens(messages=messages)
        # Fallback: total_chars // 3 + 10 (chat template overhead)
        assert count == max(1, 16 // 3 + 10)

    def test_count_tokens_requires_exactly_one_of_text_or_messages(self):
        """Neither or both args is a ValueError."""
        adapter = LiteLLMAdapter(model="gpt-4o")
        with pytest.raises(ValueError, match="requires either text or messages"):
            adapter.count_tokens()
        with pytest.raises(ValueError, match="text OR messages"):
            adapter.count_tokens("hi", messages=[{"role": "user", "content": "hi"}])


class TestContextWindowProperty:
    """Tests for the new context_window / min_output_tokens properties."""

    def test_context_window_returns_configured_value(self):
        adapter = LiteLLMAdapter(model="test-model", context_window=8192)
        assert adapter.context_window == 8192

    @patch("litellm.get_model_info")
    def test_context_window_falls_back_to_get_model_info(self, mock_info):
        """Without a configured override, context_window queries litellm."""
        mock_info.return_value = {"max_input_tokens": 128_000}
        adapter = LiteLLMAdapter(model="gpt-4o")  # no context_window= arg
        assert adapter.context_window == 128_000

    @patch("litellm.get_model_info")
    def test_context_window_unknown_model_returns_none(self, mock_info):
        mock_info.side_effect = Exception("Model not found")
        adapter = LiteLLMAdapter(model="totally-unknown-model-xyz")
        assert adapter.context_window is None

    def test_context_window_configured_overrides_detection(self):
        """A configured value takes precedence over litellm detection."""
        adapter = LiteLLMAdapter(model="gpt-4o", context_window=42)
        assert adapter.context_window == 42

    def test_min_output_tokens_matches_clamp_floor(self):
        """Exposed min_output_tokens matches the value used by _build_params."""
        adapter = LiteLLMAdapter(model="gpt-4o")
        assert adapter.min_output_tokens == 128


# ---------------------------------------------------------------------------
# Tests: get_pricing()
# ---------------------------------------------------------------------------


class TestLiteLLMAdapterPricing:
    """Test pricing information retrieval."""

    @patch("litellm.get_model_info")
    def test_get_pricing_from_litellm(self, mock_info):
        mock_info.return_value = {
            "input_cost_per_token": 0.000005,  # $5 per 1M
            "output_cost_per_token": 0.000015,  # $15 per 1M
        }

        adapter = LiteLLMAdapter(model="gpt-4o")
        pricing = adapter.get_pricing()

        assert pricing["input_cost_per_1m"] == pytest.approx(5.0)
        assert pricing["output_cost_per_1m"] == pytest.approx(15.0)

    @patch("litellm.get_model_info")
    def test_get_pricing_fallback_on_error(self, mock_info):
        mock_info.side_effect = Exception("Model not found")

        adapter = LiteLLMAdapter(model="unknown-model")
        pricing = adapter.get_pricing()

        assert pricing["input_cost_per_1m"] == 0.0
        assert pricing["output_cost_per_1m"] == 0.0


# ---------------------------------------------------------------------------
# Tests: get_completion_cost()
# ---------------------------------------------------------------------------


class TestLiteLLMAdapterCost:
    """Test cost estimation."""

    @patch("litellm.cost_per_token")
    def test_get_completion_cost(self, mock_cost):
        mock_cost.return_value = (0.0015, 0.0010)  # (prompt_cost, completion_cost)

        adapter = LiteLLMAdapter(model="gpt-4o")
        cost = adapter.get_completion_cost(input_tokens=1000, output_tokens=500)

        assert cost == pytest.approx(0.0025)
        mock_cost.assert_called_once_with(
            model="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=500,
        )

    @patch("litellm.cost_per_token")
    def test_get_completion_cost_fallback_on_error(self, mock_cost):
        mock_cost.side_effect = Exception("Cost lookup failed")

        adapter = LiteLLMAdapter(model="gpt-4o")
        cost = adapter.get_completion_cost(input_tokens=1000, output_tokens=500)

        assert cost == 0.0


# ---------------------------------------------------------------------------
# Tests: Health check
# ---------------------------------------------------------------------------


class TestLiteLLMAdapterHealth:
    """Test health check functionality."""

    @patch("litellm.completion")
    def test_health_check_success(self, mock_completion):
        mock_completion.return_value = _mock_completion_response(content="pong")

        adapter = LiteLLMAdapter(model="gpt-4o")
        assert adapter.check_health() is True

    @patch("litellm.completion")
    def test_health_check_failure(self, mock_completion):
        mock_completion.side_effect = Exception("Connection refused")

        adapter = LiteLLMAdapter(model="gpt-4o")
        assert adapter.check_health() is False


# ---------------------------------------------------------------------------
# Tests: Two-model integration
# ---------------------------------------------------------------------------


class TestLiteLLMAdapterTwoModel:
    """Test two-model (root + recursive) configuration."""

    @patch("litellm.completion")
    def test_root_then_recursive_calls(self, mock_completion):
        """Simulate RLM: root call, then recursive call with cheaper model."""
        root_resp = _mock_completion_response(
            content="Let me explore...",
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
        )
        sub_resp = _mock_completion_response(
            content="Found the answer.",
            model="gpt-4o-mini",
            prompt_tokens=80,
            completion_tokens=30,
        )
        mock_completion.side_effect = [root_resp, sub_resp]

        adapter = LiteLLMAdapter(
            root_model="gpt-4o",
            recursive_model="gpt-4o-mini",
        )

        # Root call
        adapter.use_root_model()
        r1 = adapter.complete([{"role": "user", "content": "Analyze this"}])
        assert r1.model == "gpt-4o"
        assert mock_completion.call_args_list[0][1]["model"] == "gpt-4o"

        # Recursive call
        adapter.use_recursive_model()
        r2 = adapter.complete([{"role": "user", "content": "Explore section 3"}])
        assert r2.model == "gpt-4o-mini"
        assert mock_completion.call_args_list[1][1]["model"] == "gpt-4o-mini"

    def test_single_model_is_not_two_model(self):
        adapter = LiteLLMAdapter(model="gpt-4o")
        assert adapter.is_two_model is False

    def test_same_root_and_recursive_is_not_two_model(self):
        adapter = LiteLLMAdapter(
            root_model="gpt-4o",
            recursive_model="gpt-4o",
        )
        assert adapter.is_two_model is False

    def test_different_root_and_recursive_is_two_model(self):
        adapter = LiteLLMAdapter(
            root_model="gpt-4o",
            recursive_model="gpt-4o-mini",
        )
        assert adapter.is_two_model is True


# ---------------------------------------------------------------------------
# Tests: LLMPort protocol compliance
# ---------------------------------------------------------------------------


class TestLiteLLMAdapterProtocol:
    """Verify the adapter satisfies the LLMPort protocol."""

    def test_has_complete_method(self):
        adapter = LiteLLMAdapter()
        assert callable(getattr(adapter, "complete", None))

    def test_has_complete_stream_method(self):
        adapter = LiteLLMAdapter()
        assert callable(getattr(adapter, "complete_stream", None))

    def test_has_count_tokens_method(self):
        adapter = LiteLLMAdapter()
        assert callable(getattr(adapter, "count_tokens", None))

    def test_has_get_pricing_method(self):
        adapter = LiteLLMAdapter()
        assert callable(getattr(adapter, "get_pricing", None))


# ---------------------------------------------------------------------------
# Tests: Public client integration with LiteLLM
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestPublicClientLiteLLM:
    """Test that the public client correctly wires up the LiteLLM adapter."""

    def test_client_creates_litellm_adapter(self):
        from rlmkit.public.client import RLMKitClient

        client = RLMKitClient(provider="litellm", model="gpt-4o")
        assert type(client._llm).__name__ == "LiteLLMAdapter"
        assert client._llm.active_model == "gpt-4o"

    def test_client_creates_two_model_litellm(self):
        from rlmkit.public.client import RLMKitClient

        client = RLMKitClient(
            provider="litellm",
            model="gpt-4o",
            root_model="gpt-4o",
            recursive_model="gpt-4o-mini",
        )
        assert client._llm.is_two_model is True
        assert client._llm.root_model == "gpt-4o"
        assert client._llm.recursive_model == "gpt-4o-mini"

    def test_client_mock_still_works(self):
        from rlmkit.public.client import RLMKitClient

        client = RLMKitClient(provider="mock")
        assert type(client._llm).__name__ == "MockLLMAdapter"


class TestContextLengthChars:
    """Tests for LiteLLMAdapter.context_length_chars property."""

    def test_known_model_returns_chars(self):
        """A known model (gpt-4o) returns a positive context length in chars."""
        adapter = LiteLLMAdapter(model="gpt-4o")
        result = adapter.context_length_chars
        # gpt-4o has 128K input tokens → 128000 * 4 = 512000 chars
        assert result is not None
        assert result > 0

    def test_unknown_model_returns_none(self):
        """An unknown/local model returns None (graceful degradation)."""
        adapter = LiteLLMAdapter(model="totally-unknown-model-xyz")
        result = adapter.context_length_chars
        assert result is None

    def test_prefixed_model_tries_base_name(self):
        """A model with provider prefix (ollama/llama3) tries base name fallback."""
        adapter = LiteLLMAdapter(model="ollama/totally-unknown-xyz")
        result = adapter.context_length_chars
        # Both prefixed and base name are unknown → None
        assert result is None


# ---------------------------------------------------------------------------
# Tests: context_window dynamic clamping
# ---------------------------------------------------------------------------


class TestContextWindowClamping:
    """Verify that _build_params dynamically clamps max_tokens when
    context_window is set."""

    def test_no_clamping_without_context_window(self):
        """Without context_window, max_tokens is passed through unchanged."""
        adapter = LiteLLMAdapter(model="test", max_tokens=4096)
        params = adapter._build_params([{"role": "user", "content": "hello"}])
        assert params["max_tokens"] == 4096

    def test_no_clamping_when_prompt_is_small(self):
        """When prompt fits comfortably, max_tokens stays at configured value."""
        adapter = LiteLLMAdapter(model="test", max_tokens=1024, context_window=8192)
        # Small prompt: ~10 chars → ~3 tokens; 8192 - 3 - 10 >> 1024
        params = adapter._build_params([{"role": "user", "content": "hello"}])
        assert params["max_tokens"] == 1024

    @patch("litellm.token_counter", return_value=6000)
    def test_clamps_when_prompt_approaches_limit(self, mock_tc):
        """max_tokens should shrink when prompt consumes most of the context."""
        adapter = LiteLLMAdapter(model="test", max_tokens=4096, context_window=8192)
        # context_window=8192, prompt=6000, reserve=max(8192*0.05, 64)=409
        # remaining = 8192 - 6000 - 409 = 1783
        params = adapter._build_params([{"role": "user", "content": "big prompt"}])
        expected_reserve = max(int(8192 * 0.05), 64)  # 409
        assert params["max_tokens"] == 8192 - 6000 - expected_reserve
        assert params["max_tokens"] < 4096

    @patch("litellm.token_counter", return_value=4000)
    def test_raises_proactive_overflow_when_prompt_exceeds_context(self, mock_tc):
        """When prompt leaves less than 128 tokens even without reserve, raise proactively."""
        adapter = LiteLLMAdapter(model="test", max_tokens=4096, context_window=2048)
        # prompt=4000 > context_window=2048 → hard_headroom < 0 → ValueError
        with pytest.raises(ValueError, match="Context window exhausted"):
            adapter._build_params([{"role": "user", "content": "huge prompt"}])
        # Clamp info should still be populated for diagnostics
        info = adapter.last_clamp_info
        assert info["proactive_overflow"] is True
        assert info["effective_max_tokens"] == 0

    @patch("litellm.token_counter", return_value=7900)
    def test_skips_reserve_when_tight_but_feasible(self, mock_tc):
        """When remaining < 128 but hard headroom >= 128, skip reserve and use hard headroom."""
        adapter = LiteLLMAdapter(model="test", max_tokens=4096, context_window=8192)
        # prompt=7900, reserve=409, remaining=8192-7900-409=-117 < 128
        # but hard_headroom=8192-7900=292 >= 128 → use 292
        params = adapter._build_params([{"role": "user", "content": "tight prompt"}])
        assert params["max_tokens"] == 8192 - 7900  # 292
        info = adapter.last_clamp_info
        assert info["clamped"] is True

    @patch("litellm.token_counter", return_value=6000)
    def test_clamping_preserves_other_params(self, mock_tc):
        """Clamping max_tokens should not affect other parameters."""
        adapter = LiteLLMAdapter(
            model="test",
            max_tokens=4096,
            context_window=8192,
            temperature=0.5,
            api_key="sk-test",
            api_base="http://localhost:8000",
        )
        params = adapter._build_params([{"role": "user", "content": "big prompt"}])
        assert params["temperature"] == 0.5
        assert params["api_key"] == "sk-test"
        assert params["api_base"] == "http://localhost:8000"
        assert params["model"] == "test"

    def test_context_window_none_behaves_like_absent(self):
        """context_window=None should not trigger any clamping."""
        adapter = LiteLLMAdapter(model="test", max_tokens=4096, context_window=None)
        big_prompt = "x" * 24000
        params = adapter._build_params([{"role": "user", "content": big_prompt}])
        assert params["max_tokens"] == 4096

    @patch("litellm.token_counter")
    def test_multi_message_conversation_growth(self, mock_tc):
        """Simulates growing conversation: max_tokens should decrease as messages accumulate."""
        adapter = LiteLLMAdapter(model="test", max_tokens=4096, context_window=8192)

        # Small conversation: 100 tokens
        small_msgs = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hello"},
        ]
        mock_tc.return_value = 100
        params_small = adapter._build_params(small_msgs)

        # Big conversation: 7000 tokens
        big_msgs = list(small_msgs)
        for i in range(10):
            big_msgs.append({"role": "assistant", "content": f"Response {i}: " + "x" * 2000})
            big_msgs.append({"role": "user", "content": f"Follow-up {i}: " + "y" * 500})
        mock_tc.return_value = 7000
        params_big = adapter._build_params(big_msgs)

        reserve = max(int(8192 * 0.05), 64)  # 409
        # Big conversation should have smaller max_tokens
        assert params_small["max_tokens"] == 4096  # 8192 - 100 - 409 = 7683 > 4096 → no clamp
        assert params_big["max_tokens"] == 8192 - 7000 - reserve  # 783

    @patch("litellm.token_counter", return_value=4204)
    def test_reserve_prevents_one_token_context_overflow(self, mock_tc):
        """Keep headroom so a tokenizer mismatch cannot overflow by a single token.

        With 5% reserve (409 tokens for 8192 context), the clamp fires well
        before the boundary, unlike the old 32-token reserve that was off by 1.
        """
        adapter = LiteLLMAdapter(model="test", max_tokens=4096, context_window=8192)
        params = adapter._build_params([{"role": "user", "content": "prompt"}])
        reserve = max(int(8192 * 0.05), 64)  # 409
        expected = 8192 - 4204 - reserve  # 3579
        assert params["max_tokens"] == expected
        assert params["max_tokens"] < 4096
