"""Provider fallback and switching integration tests.

Tests that strategies can be used with different LLM providers (represented
by separate MockLLMClient instances) and that switching between them does
not leak state.
"""

import pytest

from rlmkit import RLM, MockLLMClient, RLMConfig
from rlmkit.strategies.direct import DirectStrategy
from rlmkit.strategies.rlm_strategy import RLMStrategy
from rlmkit.core.model_config import ModelConfig, ModelMetrics


pytestmark = pytest.mark.integration


class TestProviderSwitching:
    """Switching between providers produces independent results."""

    def test_different_clients_return_different_answers(self, sample_document):
        """Two strategies with distinct mock clients return their own answers."""
        client_a = MockLLMClient(["Answer from provider A"])
        client_b = MockLLMClient(["Answer from provider B"])

        result_a = DirectStrategy(client=client_a).run(sample_document, "question")
        result_b = DirectStrategy(client=client_b).run(sample_document, "question")

        assert result_a.answer == "Answer from provider A"
        assert result_b.answer == "Answer from provider B"

    def test_no_state_leakage_between_rlm_instances(self, sample_document):
        """Two RLM instances using different clients have isolated environments."""
        client_a = MockLLMClient([
            "```python\nshared_var = 'from_a'\n```",
            "FINAL: A done",
        ])
        client_b = MockLLMClient([
            "```python\nprint(shared_var)\n```",  # shared_var should NOT exist
            "FINAL: B done",
        ])

        rlm_a = RLM(client=client_a)
        rlm_b = RLM(client=client_b)

        result_a = rlm_a.run(prompt=sample_document, query="Run A")
        result_b = rlm_b.run(prompt=sample_document, query="Run B")

        assert result_a.success
        assert result_b.success

        # Instance B should NOT see shared_var from instance A
        exec_b = [t for t in result_b.trace if t["role"] == "execution"]
        if exec_b:
            assert "from_a" not in exec_b[0]["content"]

    def test_sequential_provider_switch(self, sample_document):
        """Running with provider A then provider B yields correct results."""
        client_a = MockLLMClient(["FINAL: Provider A says hello"])
        client_b = MockLLMClient(["FINAL: Provider B says goodbye"])

        rlm = RLM(client=client_a)
        result_a = rlm.run(prompt=sample_document, query="Greet me")
        assert result_a.answer == "Provider A says hello"

        rlm = RLM(client=client_b)
        result_b = rlm.run(prompt=sample_document, query="Farewell")
        assert result_b.answer == "Provider B says goodbye"

    def test_mock_client_call_history_isolation(self, sample_document):
        """Each mock client tracks only its own call history."""
        client_a = MockLLMClient(["FINAL: A"])
        client_b = MockLLMClient(["FINAL: B"])

        RLM(client=client_a).run(prompt=sample_document, query="Q1")
        RLM(client=client_b).run(prompt=sample_document, query="Q2")

        assert client_a.call_count == 1
        assert client_b.call_count == 1

        # Call history should reference only the query sent to each client
        assert "Q1" in client_a.call_history[0][-1]["content"]
        assert "Q2" in client_b.call_history[0][-1]["content"]


class TestProviderErrorFallback:
    """Verify graceful handling when a provider fails."""

    def test_failing_provider_returns_error_result(self, sample_document):
        """A provider that raises returns a StrategyResult with success=False."""

        class _BrokenClient:
            def complete(self, messages):
                raise ConnectionError("API unreachable")

        strategy = DirectStrategy(client=_BrokenClient())
        result = strategy.run(sample_document, "Any question")

        assert not result.success
        assert "API unreachable" in result.error

    def test_rlm_with_failing_provider(self, sample_document):
        """RLM wraps provider errors into RLMResult.error."""

        class _BrokenClient:
            def complete(self, messages):
                raise TimeoutError("Request timed out")

        rlm = RLM(client=_BrokenClient())
        result = rlm.run(prompt=sample_document, query="Will fail")

        assert not result.success
        assert "Request timed out" in result.error


class TestModelConfigIntegration:
    """Verify ModelConfig and ModelMetrics work correctly."""

    def test_cost_optimized_openai(self):
        """cost_optimized('openai') sets expected model names."""
        config = ModelConfig.cost_optimized("openai")
        assert config.root_model == "gpt-4o"
        assert config.sub_model == "gpt-4o-mini"
        assert config.root_provider == "openai"
        assert config.sub_provider == "openai"

    def test_cost_optimized_anthropic(self):
        """cost_optimized('anthropic') sets expected model names."""
        config = ModelConfig.cost_optimized("anthropic")
        assert "claude" in config.root_model.lower() or "sonnet" in config.root_model.lower()
        assert "claude" in config.sub_model.lower() or "haiku" in config.sub_model.lower()
        assert config.root_provider == "anthropic"
        assert config.sub_provider == "anthropic"

    def test_from_single_model(self):
        """from_single_model uses the same model for root and sub."""
        config = ModelConfig.from_single_model("gpt-4o", provider="openai")
        assert config.root_model == "gpt-4o"
        assert config.sub_model == "gpt-4o"
        assert config.root_provider == "openai"
        assert config.sub_provider == "openai"

    def test_model_metrics_tracking(self):
        """ModelMetrics tracks root and sub usage independently."""
        metrics = ModelMetrics()

        metrics.root_calls = 2
        metrics.root_tokens = 1000
        metrics.root_cost = 0.10

        metrics.sub_calls = 5
        metrics.sub_tokens = 3000
        metrics.sub_cost = 0.03

        assert metrics.total_calls == 7
        assert metrics.total_tokens == 4000
        assert metrics.total_cost == pytest.approx(0.13)

    def test_model_metrics_savings_calculation(self):
        """savings_vs_single_model returns positive value when subs are cheaper."""
        metrics = ModelMetrics()
        metrics.root_tokens = 1000
        metrics.root_cost = 0.10
        metrics.sub_tokens = 3000
        metrics.sub_cost = 0.03

        # If we used root pricing for everything: 4000 * 0.0001 = 0.40
        single_model_cost_per_token = 0.10 / 1000  # root rate
        savings = metrics.savings_vs_single_model(single_model_cost_per_token)

        # hypothetical = 4000 * 0.0001 = 0.40, actual = 0.13, savings = 0.27
        assert savings == pytest.approx(0.27)
        assert savings > 0
