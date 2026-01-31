"""Tests for strategy protocol conformance and base types."""

import pytest
from rlmkit import MockLLMClient
from rlmkit.core.budget import TokenUsage
from rlmkit.strategies.base import LLMStrategy, StrategyResult
from rlmkit.strategies.direct import DirectStrategy
from rlmkit.strategies.rlm_strategy import RLMStrategy


class TestStrategyResult:
    """Test StrategyResult dataclass."""

    def test_defaults(self):
        r = StrategyResult(strategy="test", answer="hello")
        assert r.success is True
        assert r.steps == 0
        assert r.cost == 0.0
        assert r.error is None
        assert r.trace == []
        assert r.metadata == {}

    def test_to_dict(self):
        r = StrategyResult(strategy="direct", answer="42", steps=1, cost=0.01)
        d = r.to_dict()
        assert d["strategy"] == "direct"
        assert d["answer"] == "42"
        assert d["steps"] == 1
        assert d["cost"] == 0.01
        assert "tokens" in d

    def test_to_execution_metrics(self):
        tokens = TokenUsage(input_tokens=100, output_tokens=50)
        r = StrategyResult(
            strategy="rlm",
            answer="result",
            steps=3,
            tokens=tokens,
            elapsed_time=1.5,
            cost=0.02,
            trace=[{"step": 1}],
        )
        em = r.to_execution_metrics()
        assert em.mode == "rlm"
        assert em.answer == "result"
        assert em.steps == 3
        assert em.tokens.total_tokens == 150
        assert em.elapsed_time == 1.5
        assert em.cost == 0.02
        assert em.trace == [{"step": 1}]


class TestProtocolConformance:
    """Verify all strategies satisfy the LLMStrategy protocol."""

    def test_direct_is_strategy(self):
        client = MockLLMClient(["answer"])
        s = DirectStrategy(client=client)
        assert isinstance(s, LLMStrategy)

    def test_rlm_is_strategy(self):
        client = MockLLMClient(["FINAL: answer"])
        s = RLMStrategy(client=client)
        assert isinstance(s, LLMStrategy)


class TestDirectStrategy:
    """Test DirectStrategy execution."""

    def test_basic_run(self):
        client = MockLLMClient(["The answer is 42."])
        s = DirectStrategy(client=client)
        assert s.name == "direct"

        result = s.run("some content", "what is the answer?")
        assert result.success
        assert result.answer == "The answer is 42."
        assert result.strategy == "direct"
        assert result.steps == 1
        assert result.tokens.total_tokens > 0
        assert result.elapsed_time >= 0

    def test_error_handling(self):
        class FailingClient:
            def complete(self, messages):
                raise RuntimeError("LLM unavailable")

        s = DirectStrategy(client=FailingClient())
        result = s.run("content", "query")
        assert not result.success
        assert "LLM unavailable" in result.error
        assert result.answer == ""


class TestRLMStrategy:
    """Test RLMStrategy execution."""

    def test_basic_run(self):
        client = MockLLMClient(["FINAL: done"])
        s = RLMStrategy(client=client)
        assert s.name == "rlm"

        result = s.run("some content", "summarize")
        assert result.success
        assert result.answer == "done"
        assert result.strategy == "rlm"
        assert result.steps >= 1
        assert result.elapsed_time >= 0
        assert "max_steps" in result.metadata

    def test_multi_step(self):
        client = MockLLMClient([
            '```python\nx = peek(0, 5)\nprint(x)\n```',
            "FINAL: first five chars",
        ])
        s = RLMStrategy(client=client)
        result = s.run("Hello World", "first 5 chars?")
        assert result.success
        assert result.steps == 2

    def test_error_handling(self):
        class FailingClient:
            def complete(self, messages):
                raise RuntimeError("boom")

        s = RLMStrategy(client=FailingClient())
        result = s.run("content", "query")
        assert not result.success
        assert "boom" in result.error
