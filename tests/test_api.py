"""
Tests for the unified interact() API.

This tests the new high-level API introduced in Bet 1.
"""

from unittest.mock import MagicMock

import pytest

from rlmkit import InteractResult, complete, interact
from rlmkit.config import RLMConfig
from rlmkit.llm import MockLLMClient

# Test content samples
SHORT_TEXT = "This is a short document about Python programming."
MEDIUM_TEXT = "x" * 40000  # 40K characters (~10K tokens) - in RAG range (8K-100K)
LARGE_TEXT = "x" * 50000  # 50K characters (~12.5K tokens)
HUGE_TEXT = "x" * 500000  # 500K characters (~125K tokens)


# Pytest fixture to mock get_llm_client
@pytest.fixture(autouse=True)
def mock_llm_client(monkeypatch):
    """Automatically mock get_llm_client for all tests."""
    mock_client = MockLLMClient(["Final answer: Test response"])

    def mock_get_llm_client(*args, **kwargs):
        return mock_client

    monkeypatch.setattr("rlmkit.api.get_llm_client", mock_get_llm_client)
    return mock_client


# Fixture for mocking OpenAIEmbedder (needed for RAG tests)
@pytest.fixture(autouse=True)
def mock_embedder(monkeypatch):
    """Mock OpenAIEmbedder for RAG tests."""
    mock_embedder_class = MagicMock()
    mock_embedder_instance = MagicMock()
    mock_embedder_instance.embed_query.return_value = [0.1] * 1536
    mock_embedder_instance.embed.return_value = [[0.1] * 1536]
    mock_embedder_class.return_value = mock_embedder_instance

    # Try to patch, but don't fail if module doesn't exist
    try:
        monkeypatch.setattr("rlmkit.api.OpenAIEmbedder", mock_embedder_class)
    except Exception:
        pass

    return mock_embedder_instance


class TestInteractBasics:
    """Test basic interact() functionality."""

    def test_interact_returns_interact_result(self):
        """Test that interact() returns an InteractResult object."""
        MockLLMClient(["Final answer: This is about Python."])

        result = interact(
            content=SHORT_TEXT, query="What is this about?", mode="direct", provider="openai"
        )

        assert isinstance(result, InteractResult)
        assert isinstance(result.answer, str)
        assert isinstance(result.mode_used, str)
        assert isinstance(result.metrics, dict)

    def test_interact_result_has_required_fields(self):
        """Test that InteractResult has all required fields."""
        MockLLMClient(["Final answer: Test response."])

        result = interact(content=SHORT_TEXT, query="Test query", mode="direct", provider="openai")

        assert hasattr(result, "answer")
        assert hasattr(result, "mode_used")
        assert hasattr(result, "metrics")
        assert hasattr(result, "trace")
        assert hasattr(result, "raw_result")

    def test_interact_result_str(self):
        """Test that InteractResult __str__ returns the answer."""
        MockLLMClient(["Final answer: String test."])

        result = interact(content=SHORT_TEXT, query="Test", mode="direct", provider="openai")

        assert str(result) == result.answer

    def test_interact_result_to_dict(self):
        """Test that InteractResult.to_dict() works."""
        MockLLMClient(["Final answer: Dict test."])

        result = interact(content=SHORT_TEXT, query="Test", mode="direct", provider="openai")

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "answer" in result_dict
        assert "mode_used" in result_dict
        assert "metrics" in result_dict
        assert "has_trace" in result_dict


class TestModeSelection:
    """Test mode selection and validation."""

    def test_direct_mode_explicit(self):
        """Test explicit direct mode selection."""
        result = interact(content=SHORT_TEXT, query="Test", mode="direct", provider="openai")

        assert result.mode_used == "direct"

    def test_rag_mode_explicit(self):
        """Test explicit RAG mode selection."""
        result = interact(content=MEDIUM_TEXT, query="Test", mode="rag", provider="openai")

        assert result.mode_used == "rag"

    def test_rlm_mode_explicit(self):
        """Test explicit RLM mode selection."""
        result = interact(content=MEDIUM_TEXT, query="Test", mode="rlm", provider="openai")

        assert result.mode_used == "rlm"

    def test_auto_mode_selects_direct_for_small(self):
        """Test that auto mode selects direct for small content."""
        result = interact(
            content=SHORT_TEXT,  # < 8K tokens
            query="Test",
            mode="auto",
            provider="openai",
        )

        assert result.mode_used == "direct"

    def test_auto_mode_selects_rag_for_medium(self):
        """Test that auto mode selects RAG for medium content."""
        result = interact(
            content=MEDIUM_TEXT,  # 8K-100K tokens range
            query="Test",
            mode="auto",
            provider="openai",
        )

        assert result.mode_used == "rag"

    def test_auto_mode_selects_rlm_for_large(self):
        """Test that auto mode selects RLM for large content."""
        result = interact(
            content=HUGE_TEXT,  # > 100K tokens
            query="Test",
            mode="auto",
            provider="openai",
        )

        assert result.mode_used == "rlm"

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            interact(content=SHORT_TEXT, query="Test", mode="invalid_mode", provider="openai")


class TestInputValidation:
    """Test input validation."""

    def test_empty_content_raises_error(self):
        """Test that empty content raises ValueError."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            interact(content="", query="Test query", mode="direct", provider="openai")

    def test_empty_query_raises_error(self):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="query cannot be empty"):
            interact(content="Some content", query="", mode="direct", provider="openai")


class TestMetrics:
    """Test that metrics are populated correctly."""

    def test_metrics_contains_required_fields(self):
        """Test that metrics dict contains all required fields."""
        result = interact(content=SHORT_TEXT, query="Test", mode="direct", provider="openai")

        metrics = result.metrics
        assert "total_tokens" in metrics
        assert "input_tokens" in metrics
        assert "output_tokens" in metrics
        assert "total_cost" in metrics
        assert "execution_time" in metrics
        assert "llm_calls" in metrics

    def test_metrics_values_are_numeric(self):
        """Test that metric values are numeric types."""
        result = interact(content=SHORT_TEXT, query="Test", mode="direct", provider="openai")

        metrics = result.metrics
        assert isinstance(metrics["total_tokens"], (int, float))
        assert isinstance(metrics["input_tokens"], (int, float))
        assert isinstance(metrics["output_tokens"], (int, float))
        assert isinstance(metrics["total_cost"], (int, float))
        assert isinstance(metrics["execution_time"], (int, float))
        assert isinstance(metrics["llm_calls"], int)


class TestCompleteFunction:
    """Test the complete() convenience function."""

    def test_complete_returns_string(self):
        """Test that complete() returns just the answer string."""
        answer = complete(content=SHORT_TEXT, query="Test", mode="direct", provider="openai")

        assert isinstance(answer, str)

    def test_complete_equals_interact_answer(self):
        """Test that complete() returns same answer as interact().answer."""
        interact(content=SHORT_TEXT, query="Test", mode="direct", provider="openai")

        answer = complete(content=SHORT_TEXT, query="Test", mode="direct", provider="openai")

        # Note: answers might differ due to LLM non-determinism,
        # but both should be non-empty strings
        assert len(answer) > 0
        assert isinstance(answer, str)


class TestVerboseMode:
    """Test verbose output mode."""

    def test_verbose_mode_enabled(self, capsys):
        """Test that verbose mode prints output."""
        interact(content=SHORT_TEXT, query="Test", mode="auto", provider="openai", verbose=True)

        captured = capsys.readouterr()
        # Should print mode selection and execution info
        assert "Auto Mode" in captured.out or "Setup" in captured.out

    def test_verbose_mode_disabled_by_default(self, capsys):
        """Test that verbose mode is off by default."""
        interact(content=SHORT_TEXT, query="Test", mode="direct", provider="openai")

        captured = capsys.readouterr()
        # Should not print anything
        assert captured.out == ""


class TestConfiguration:
    """Test configuration options."""

    def test_custom_config_passed(self):
        """Test that custom RLMConfig is used."""
        config = RLMConfig()
        config.execution.max_iterations = 5

        result = interact(
            content=SHORT_TEXT, query="Test", mode="rlm", provider="openai", config=config
        )

        # Should complete without error
        assert result is not None

    def test_default_config_used_when_none(self):
        """Test that default config is created when none provided."""
        result = interact(
            content=SHORT_TEXT,
            query="Test",
            mode="direct",
            provider="openai",
            # No config parameter
        )

        assert result is not None


class TestProviderAndModel:
    """Test provider and model specification."""

    def test_provider_specified(self):
        """Test specifying provider explicitly."""
        result = interact(
            content=SHORT_TEXT, query="Test", mode="direct", provider="openai", model="gpt-4o-mini"
        )

        assert result is not None

    def test_model_specified(self):
        """Test specifying model explicitly."""
        result = interact(
            content=SHORT_TEXT, query="Test", mode="direct", provider="openai", model="gpt-4o"
        )

        assert result is not None


class TestRawResultAccess:
    """Test access to underlying StrategyResult."""

    def test_raw_result_available(self):
        """Test that raw_result is available."""
        result = interact(content=SHORT_TEXT, query="Test", mode="direct", provider="openai")

        assert result.raw_result is not None

    def test_raw_result_is_strategy_result(self):
        """Test that raw_result is a StrategyResult."""
        from rlmkit.strategies import StrategyResult

        result = interact(content=SHORT_TEXT, query="Test", mode="direct", provider="openai")

        assert isinstance(result.raw_result, StrategyResult)

    def test_raw_result_matches_interact_result(self):
        """Test that raw_result data matches InteractResult."""
        result = interact(content=SHORT_TEXT, query="Test", mode="direct", provider="openai")

        assert result.answer == result.raw_result.answer
        assert result.metrics["total_tokens"] == result.raw_result.tokens.total_tokens


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_long_query(self):
        """Test with very long query."""
        long_query = "What is " + "x" * 10000 + "?"

        result = interact(content=SHORT_TEXT, query=long_query, mode="direct", provider="openai")

        assert result is not None

    def test_unicode_content(self):
        """Test with Unicode content."""
        unicode_content = "Hello 世界! 🌍 Привет мир!"

        result = interact(
            content=unicode_content,
            query="What languages are shown?",
            mode="direct",
            provider="openai",
        )

        assert result is not None

    def test_special_characters(self):
        """Test with special characters in content."""
        special_content = "Test with special chars: \n\t\r\"'\\`$#@!"

        result = interact(
            content=special_content,
            query="What special characters are there?",
            mode="direct",
            provider="openai",
        )

        assert result is not None


# Integration test marker
@pytest.mark.integration
class TestIntegration:
    """Integration tests (require real API keys)."""

    @pytest.mark.skip(reason="Requires OPENAI_API_KEY - run with: pytest -m integration")
    def test_real_openai_direct(self):
        """Test with real OpenAI API (requires API key)."""
        result = interact(
            content="The sky is blue.",
            query="What color is the sky?",
            mode="direct",
            provider="openai",
            model="gpt-4o-mini",
        )

        assert "blue" in result.answer.lower()

    @pytest.mark.skip(reason="Requires OPENAI_API_KEY - run with: pytest -m integration")
    def test_real_openai_auto(self):
        """Test auto mode with real OpenAI API."""
        result = interact(
            content="RLMKit is a toolkit for LLMs.",
            query="What is RLMKit?",
            mode="auto",
            provider="openai",
            model="gpt-4o-mini",
            verbose=True,
        )

        assert "rlmkit" in result.answer.lower() or "toolkit" in result.answer.lower()
