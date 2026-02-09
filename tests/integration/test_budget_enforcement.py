"""Budget enforcement integration tests.

Tests that budget limits (steps, tokens, cost, time, recursion depth) are
correctly tracked and enforced during RLM execution.
"""

import time

import pytest

from rlmkit import (
    RLM,
    MockLLMClient,
    RLMConfig,
    BudgetExceeded,
    BudgetTracker,
    BudgetLimits,
    CostTracker,
    TokenUsage,
    estimate_tokens,
)


pytestmark = pytest.mark.integration


class TestMaxStepsEnforcement:
    """Verify that the RLM loop halts at the step budget."""

    def test_max_steps_enforced(self, mock_llm_never_final, sample_document):
        """BudgetExceeded raised when max_steps reached without FINAL."""
        config = RLMConfig()
        config.execution.max_steps = 4

        rlm = RLM(client=mock_llm_never_final, config=config)

        with pytest.raises(BudgetExceeded) as exc_info:
            rlm.run(prompt=sample_document, query="Keep going")

        assert "4" in str(exc_info.value)

    def test_stops_early_if_final_before_limit(self, sample_document):
        """RLM stops as soon as FINAL is returned, even with budget remaining."""
        client = MockLLMClient([
            "```python\nx = 1\n```",
            "FINAL: Early answer",
        ])
        config = RLMConfig()
        config.execution.max_steps = 100

        rlm = RLM(client=client, config=config)
        result = rlm.run(prompt=sample_document, query="Quick check")

        assert result.success
        assert result.steps == 2
        assert result.answer == "Early answer"

    def test_step_count_matches_llm_calls(self, sample_document):
        """result.steps equals the number of LLM calls made."""
        client = MockLLMClient([
            "```python\na = 1\n```",
            "```python\nb = 2\n```",
            "```python\nc = 3\n```",
            "FINAL: Three code steps done",
        ])
        config = RLMConfig()
        config.execution.max_steps = 10

        rlm = RLM(client=client, config=config)
        result = rlm.run(prompt=sample_document, query="Count steps")

        assert result.success
        assert result.steps == 4
        assert client.call_count == 4


class TestMaxTokensEnforcement:
    """Verify that token limits trigger BudgetExceeded via BudgetTracker."""

    def test_max_tokens_enforced(self):
        """BudgetTracker raises when accumulated tokens exceed the limit."""
        limits = BudgetLimits(max_tokens=200)
        tracker = BudgetTracker(limits)

        # Each call adds ~50 input + ~25 output tokens (estimated from char count)
        for _ in range(3):
            tracker.add_llm_call("a" * 200, "b" * 100)

        # 3 calls * (50 + 25) = 225 tokens > 200 limit
        with pytest.raises(BudgetExceeded, match="Maximum tokens"):
            tracker.check_limits()

    def test_tokens_below_limit_pass(self):
        """check_limits does not raise when tokens are under the limit."""
        limits = BudgetLimits(max_tokens=1000)
        tracker = BudgetTracker(limits)

        tracker.add_llm_call("short input", "short output")
        tracker.check_limits()  # should not raise

    def test_token_accumulation_across_calls(self):
        """Tokens accumulate correctly across multiple add_llm_call invocations."""
        tracker = BudgetTracker()

        tracker.add_llm_call("a" * 40, "b" * 20)  # 10 + 5 = 15
        tracker.add_llm_call("c" * 80, "d" * 40)  # 20 + 10 = 30

        assert tracker.tokens.total_tokens == 45


class TestCostEnforcement:
    """Verify cost-based budget limits."""

    def test_max_cost_enforced(self):
        """BudgetExceeded raised when accumulated cost exceeds the limit."""
        limits = BudgetLimits(max_cost=0.10)
        cost_tracker = CostTracker(input_cost_per_1k=0.03, output_cost_per_1k=0.06)
        tracker = BudgetTracker(limits, cost_tracker)

        # Each call: (250 / 1000 * 0.03) + (125 / 1000 * 0.06) = 0.0075 + 0.0075 = 0.015
        for _ in range(7):
            tracker.add_llm_call("a" * 1000, "b" * 500)

        # 7 * 0.015 = 0.105 > 0.10
        with pytest.raises(BudgetExceeded, match="Maximum cost"):
            tracker.check_limits()

    def test_cost_stays_within_limit(self):
        """No exception when cost is under the limit."""
        limits = BudgetLimits(max_cost=1.00)
        cost_tracker = CostTracker(input_cost_per_1k=0.01, output_cost_per_1k=0.02)
        tracker = BudgetTracker(limits, cost_tracker)

        tracker.add_llm_call("input text", "output text")
        tracker.check_limits()  # should not raise


class TestTimeEnforcement:
    """Verify wall-clock time limits."""

    def test_max_time_enforced(self):
        """BudgetExceeded raised when execution time exceeds the limit."""
        limits = BudgetLimits(max_time_seconds=0.1)
        tracker = BudgetTracker(limits)

        tracker.start()
        time.sleep(0.15)

        with pytest.raises(BudgetExceeded, match="Maximum time"):
            tracker.check_limits()

    def test_time_within_limit_passes(self):
        """No exception when elapsed time is under the limit."""
        limits = BudgetLimits(max_time_seconds=5.0)
        tracker = BudgetTracker(limits)

        tracker.start()
        tracker.check_limits()  # should not raise (nearly instant)


class TestRecursionDepthEnforcement:
    """Verify recursion depth limits."""

    def test_max_recursion_depth_enforced(self):
        """BudgetExceeded raised when recursion depth exceeds the limit."""
        limits = BudgetLimits(max_recursion_depth=2)
        tracker = BudgetTracker(limits)

        tracker.enter_recursion()
        tracker.check_limits()  # depth=1, ok

        tracker.enter_recursion()
        with pytest.raises(BudgetExceeded, match="Maximum recursion depth"):
            tracker.check_limits()  # depth=2, exceeds limit

    def test_recursion_exit_resets_depth(self):
        """Exiting recursion decrements depth and allows further calls."""
        limits = BudgetLimits(max_recursion_depth=2)
        tracker = BudgetTracker(limits)

        tracker.enter_recursion()
        tracker.enter_recursion()

        # At depth=2, would fail
        with pytest.raises(BudgetExceeded):
            tracker.check_limits()

        # Exit one level
        tracker.exit_recursion()
        tracker.check_limits()  # depth=1, should pass


class TestBudgetTrackingAcrossSteps:
    """Verify that budget tracking integrates with multi-step RLM execution."""

    def test_budget_tracking_across_steps(self, sample_document):
        """BudgetTracker accurately reflects multi-step RLM execution."""
        limits = BudgetLimits(max_steps=10, max_tokens=50000)
        cost_tracker = CostTracker(input_cost_per_1k=0.01, output_cost_per_1k=0.02)
        tracker = BudgetTracker(limits, cost_tracker)

        tracker.start()

        # Simulate 3 RLM steps
        for step in range(3):
            tracker.add_step()
            tracker.check_limits()
            tracker.add_llm_call(
                sample_document + f" step {step}",
                f"```python\nresult = peek(0, 100)\nprint(result)\n```",
            )
            tracker.check_limits()

        tracker.stop()

        stats = tracker.get_stats()
        assert stats["steps"] == 3
        assert stats["tokens"]["total_tokens"] > 0
        assert stats["cost"]["total_cost"] > 0
        assert stats["elapsed_time"] >= 0

    def test_utilization_percentages(self):
        """get_utilization returns correct fractions."""
        limits = BudgetLimits(max_steps=10, max_tokens=1000, max_cost=1.0)
        cost_tracker = CostTracker(input_cost_per_1k=0.10, output_cost_per_1k=0.20)
        tracker = BudgetTracker(limits, cost_tracker)

        # 2 steps out of 10
        tracker.add_step()
        tracker.add_step()

        # ~50 + ~25 = 75 tokens out of 1000
        tracker.add_llm_call("a" * 200, "b" * 100)

        util = tracker.get_utilization()
        assert util["steps"] == pytest.approx(0.2)
        assert util["tokens"] is not None
        assert 0 < util["tokens"] < 1.0
        assert util["cost"] is not None
        assert 0 < util["cost"] < 1.0

    def test_step_history_records_per_call_data(self):
        """Each add_llm_call appends an entry to step_history."""
        tracker = BudgetTracker()
        tracker.start()

        tracker.add_step()
        tracker.add_llm_call("first input", "first output")
        tracker.add_step()
        tracker.add_llm_call("second input", "second output")

        assert len(tracker.step_history) == 2
        assert tracker.step_history[0]["step"] == 1
        assert tracker.step_history[1]["step"] == 2
        for entry in tracker.step_history:
            assert "input_tokens" in entry
            assert "output_tokens" in entry
            assert "cost" in entry
