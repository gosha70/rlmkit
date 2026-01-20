"""Tests for budget tracking and enforcement."""

import pytest
import time
from rlmkit.core.budget import (
    BudgetTracker,
    BudgetLimits,
    TokenUsage,
    CostTracker,
    estimate_tokens,
)
from rlmkit import BudgetExceeded


class TestTokenEstimation:
    """Test token estimation utilities."""
    
    def test_estimate_tokens_basic(self):
        """Test basic token estimation."""
        text = "Hello, world!"
        tokens = estimate_tokens(text)
        # ~4 chars per token, so 13 chars ≈ 3 tokens
        assert tokens > 0
        assert tokens == len(text) // 4
    
    def test_estimate_tokens_empty(self):
        """Test empty string returns at least 1 token."""
        assert estimate_tokens("") == 1
    
    def test_estimate_tokens_long(self):
        """Test long text estimation."""
        text = "a" * 1000
        tokens = estimate_tokens(text)
        assert tokens == 250  # 1000 / 4


class TestTokenUsage:
    """Test TokenUsage tracking."""
    
    def test_initial_state(self):
        """Test initial token usage is zero."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
    
    def test_add_input(self):
        """Test adding input tokens."""
        usage = TokenUsage()
        usage.add_input(100)
        assert usage.input_tokens == 100
        assert usage.total_tokens == 100
    
    def test_add_output(self):
        """Test adding output tokens."""
        usage = TokenUsage()
        usage.add_output(50)
        assert usage.output_tokens == 50
        assert usage.total_tokens == 50
    
    def test_add_both(self):
        """Test adding both input and output tokens."""
        usage = TokenUsage()
        usage.add_input(100)
        usage.add_output(50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        usage = TokenUsage()
        usage.add_input(100)
        usage.add_output(50)
        
        data = usage.to_dict()
        assert data['input_tokens'] == 100
        assert data['output_tokens'] == 50
        assert data['total_tokens'] == 150


class TestCostTracker:
    """Test cost tracking."""
    
    def test_initial_cost_zero(self):
        """Test initial cost is zero."""
        tracker = CostTracker()
        assert tracker.total_cost == 0.0
    
    def test_add_usage_no_pricing(self):
        """Test adding usage with zero pricing."""
        tracker = CostTracker(input_cost_per_1k=0.0, output_cost_per_1k=0.0)
        cost = tracker.add_usage(1000, 500)
        assert cost == 0.0
        assert tracker.total_cost == 0.0
    
    def test_add_usage_with_pricing(self):
        """Test adding usage with pricing."""
        # GPT-4 pricing: $0.03 input, $0.06 output per 1k tokens
        tracker = CostTracker(input_cost_per_1k=0.03, output_cost_per_1k=0.06)
        
        cost = tracker.add_usage(1000, 500)
        expected = (1000/1000 * 0.03) + (500/1000 * 0.06)
        assert cost == pytest.approx(expected)
        assert tracker.total_cost == pytest.approx(expected)
    
    def test_accumulate_cost(self):
        """Test cost accumulation over multiple calls."""
        tracker = CostTracker(input_cost_per_1k=0.01, output_cost_per_1k=0.02)
        
        tracker.add_usage(1000, 500)
        tracker.add_usage(2000, 1000)
        
        # First: (1000/1000 * 0.01) + (500/1000 * 0.02) = 0.02
        # Second: (2000/1000 * 0.01) + (1000/1000 * 0.02) = 0.04
        # Total: 0.06
        assert tracker.total_cost == pytest.approx(0.06)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        tracker = CostTracker(input_cost_per_1k=0.03, output_cost_per_1k=0.06)
        tracker.add_usage(1000, 500)
        
        data = tracker.to_dict()
        assert data['input_cost_per_1k'] == 0.03
        assert data['output_cost_per_1k'] == 0.06
        assert 'total_cost' in data


class TestBudgetLimits:
    """Test budget limits configuration."""
    
    def test_default_limits_unlimited(self):
        """Test default limits are None (unlimited)."""
        limits = BudgetLimits()
        assert limits.max_steps is None
        assert limits.max_tokens is None
        assert limits.max_cost is None
        assert limits.max_time_seconds is None
        assert limits.max_recursion_depth is None
    
    def test_custom_limits(self):
        """Test setting custom limits."""
        limits = BudgetLimits(
            max_steps=10,
            max_tokens=5000,
            max_cost=1.0,
            max_time_seconds=60.0,
            max_recursion_depth=3,
        )
        assert limits.max_steps == 10
        assert limits.max_tokens == 5000
        assert limits.max_cost == 1.0
        assert limits.max_time_seconds == 60.0
        assert limits.max_recursion_depth == 3
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        limits = BudgetLimits(max_steps=10, max_tokens=5000)
        data = limits.to_dict()
        
        assert data['max_steps'] == 10
        assert data['max_tokens'] == 5000
        assert data['max_cost'] is None


class TestBudgetTracker:
    """Test budget tracker."""
    
    def test_initial_state(self):
        """Test initial tracker state."""
        tracker = BudgetTracker()
        assert tracker.steps == 0
        assert tracker.tokens.total_tokens == 0
        assert tracker.recursion_depth == 0
        assert tracker.start_time is None
        assert tracker.end_time is None
    
    def test_start_stop_timing(self):
        """Test timing functionality."""
        tracker = BudgetTracker()
        tracker.start()
        assert tracker.start_time is not None
        
        time.sleep(0.1)
        tracker.stop()
        
        assert tracker.end_time is not None
        assert tracker.elapsed_time >= 0.1
    
    def test_add_step(self):
        """Test adding steps."""
        tracker = BudgetTracker()
        tracker.add_step()
        assert tracker.steps == 1
        
        tracker.add_step()
        assert tracker.steps == 2
    
    def test_add_llm_call(self):
        """Test recording LLM calls."""
        tracker = BudgetTracker()
        tracker.start()
        
        input_text = "a" * 100
        output_text = "b" * 50
        
        tracker.add_llm_call(input_text, output_text)
        
        assert tracker.tokens.input_tokens > 0
        assert tracker.tokens.output_tokens > 0
        assert len(tracker.step_history) == 1
    
    def test_add_llm_call_with_actual_tokens(self):
        """Test recording LLM call with actual token counts."""
        tracker = BudgetTracker()
        
        tracker.add_llm_call(
            "input text",
            "output text",
            actual_input_tokens=150,
            actual_output_tokens=75,
        )
        
        assert tracker.tokens.input_tokens == 150
        assert tracker.tokens.output_tokens == 75
    
    def test_recursion_tracking(self):
        """Test recursion depth tracking."""
        tracker = BudgetTracker()
        
        assert tracker.recursion_depth == 0
        
        tracker.enter_recursion()
        assert tracker.recursion_depth == 1
        
        tracker.enter_recursion()
        assert tracker.recursion_depth == 2
        
        tracker.exit_recursion()
        assert tracker.recursion_depth == 1
        
        tracker.exit_recursion()
        assert tracker.recursion_depth == 0
        
        # Should not go below 0
        tracker.exit_recursion()
        assert tracker.recursion_depth == 0
    
    def test_check_limits_no_limits(self):
        """Test check_limits with no limits set."""
        tracker = BudgetTracker()
        tracker.add_step()
        tracker.add_step()
        
        # Should not raise
        tracker.check_limits()
    
    def test_check_limits_step_exceeded(self):
        """Test step limit exceeded."""
        limits = BudgetLimits(max_steps=3)
        tracker = BudgetTracker(limits)
        
        tracker.add_step()
        tracker.add_step()
        tracker.check_limits()  # Should pass
        
        tracker.add_step()
        with pytest.raises(BudgetExceeded) as exc:
            tracker.check_limits()
        
        assert "Maximum steps (3) exceeded" in str(exc.value)
    
    def test_check_limits_token_exceeded(self):
        """Test token limit exceeded."""
        limits = BudgetLimits(max_tokens=100)
        tracker = BudgetTracker(limits)
        
        tracker.add_llm_call("a" * 200, "b" * 200)  # ~100 tokens
        
        with pytest.raises(BudgetExceeded) as exc:
            tracker.check_limits()
        
        assert "Maximum tokens (100) exceeded" in str(exc.value)
    
    def test_check_limits_cost_exceeded(self):
        """Test cost limit exceeded."""
        limits = BudgetLimits(max_cost=0.05)
        cost_tracker = CostTracker(input_cost_per_1k=0.03, output_cost_per_1k=0.06)
        tracker = BudgetTracker(limits, cost_tracker)
        
        # Add enough tokens to exceed cost
        # 1000 chars = 250 tokens (at 4 chars/token)
        # First call: (250/1000 * 0.03) + (250/1000 * 0.06) = 0.0075 + 0.015 = 0.0225
        tracker.add_llm_call("a" * 1000, "b" * 1000)
        tracker.check_limits()  # Should pass (0.0225 < 0.05)
        
        # Second call: another 0.0225, total = 0.045
        tracker.add_llm_call("a" * 1000, "b" * 1000)
        tracker.check_limits()  # Should still pass (0.045 < 0.05)
        
        # Third call: another 0.0225, total = 0.0675 > 0.05
        tracker.add_llm_call("a" * 1000, "b" * 1000)
        
        with pytest.raises(BudgetExceeded) as exc:
            tracker.check_limits()
        
        assert "Maximum cost" in str(exc.value)
    
    def test_check_limits_time_exceeded(self):
        """Test time limit exceeded."""
        limits = BudgetLimits(max_time_seconds=0.1)
        tracker = BudgetTracker(limits)
        
        tracker.start()
        tracker.check_limits()  # Should pass
        
        time.sleep(0.15)
        
        with pytest.raises(BudgetExceeded) as exc:
            tracker.check_limits()
        
        assert "Maximum time" in str(exc.value)
    
    def test_check_limits_recursion_exceeded(self):
        """Test recursion limit exceeded."""
        limits = BudgetLimits(max_recursion_depth=2)
        tracker = BudgetTracker(limits)
        
        tracker.enter_recursion()
        tracker.check_limits()  # depth=1, should pass
        
        tracker.enter_recursion()
        
        with pytest.raises(BudgetExceeded) as exc:
            tracker.check_limits()  # depth=2, should fail
        
        assert "Maximum recursion depth (2) exceeded" in str(exc.value)
    
    def test_get_stats(self):
        """Test getting usage statistics."""
        limits = BudgetLimits(max_steps=10)
        tracker = BudgetTracker(limits)
        tracker.start()
        
        tracker.add_step()
        tracker.add_llm_call("input", "output")
        
        stats = tracker.get_stats()
        
        assert stats['steps'] == 1
        assert 'tokens' in stats
        assert 'cost' in stats
        assert 'elapsed_time' in stats
        assert 'recursion_depth' in stats
        assert 'limits' in stats
    
    def test_get_utilization(self):
        """Test getting budget utilization percentages."""
        limits = BudgetLimits(max_steps=10, max_tokens=1000)
        tracker = BudgetTracker(limits)
        
        tracker.add_step()
        tracker.add_step()
        
        utilization = tracker.get_utilization()
        
        assert utilization['steps'] == 0.2  # 2/10
        assert utilization['tokens'] == 0.0  # 0/1000
        assert utilization['cost'] is None  # No cost limit
    
    def test_step_history(self):
        """Test step history recording."""
        tracker = BudgetTracker()
        tracker.start()
        
        tracker.add_step()
        tracker.add_llm_call("input1", "output1")
        
        tracker.add_step()
        tracker.add_llm_call("input2", "output2")
        
        assert len(tracker.step_history) == 2
        assert tracker.step_history[0]['step'] == 1
        assert tracker.step_history[1]['step'] == 2
        assert 'input_tokens' in tracker.step_history[0]
        assert 'output_tokens' in tracker.step_history[0]
        assert 'cost' in tracker.step_history[0]


class TestBudgetTrackerIntegration:
    """Integration tests for budget tracker."""
    
    def test_realistic_usage_scenario(self):
        """Test realistic RLM usage scenario."""
        # Simulate a realistic RLM run
        limits = BudgetLimits(
            max_steps=5,
            max_tokens=10000,
            max_cost=0.50,
            max_time_seconds=30.0,
        )
        cost_tracker = CostTracker(input_cost_per_1k=0.03, output_cost_per_1k=0.06)
        tracker = BudgetTracker(limits, cost_tracker)
        
        tracker.start()
        
        # Simulate 3 steps of RLM execution
        for step in range(3):
            tracker.add_step()
            tracker.check_limits()
            
            # Simulate LLM call
            input_prompt = "system prompt + query" * 100
            output_code = "result = peek(0, 100)" * 10
            
            tracker.add_llm_call(input_prompt, output_code)
            tracker.check_limits()
        
        tracker.stop()
        
        # Verify stats
        stats = tracker.get_stats()
        assert stats['steps'] == 3
        assert stats['tokens']['total_tokens'] > 0
        assert stats['cost']['total_cost'] > 0
        assert stats['elapsed_time'] >= 0
        
        # Verify utilization
        util = tracker.get_utilization()
        assert 0 < util['steps'] < 1.0
        assert 0 < util['tokens'] < 1.0
    
    def test_budget_enforcement_prevents_overuse(self):
        """Test that budget limits prevent excessive usage."""
        limits = BudgetLimits(max_steps=3)
        tracker = BudgetTracker(limits)
        
        # Should work for 3 steps
        for i in range(3):
            tracker.add_step()
            if i < 2:
                tracker.check_limits()  # Only first 2 should pass
        
        # 3rd step should fail
        with pytest.raises(BudgetExceeded):
            tracker.check_limits()
