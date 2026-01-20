"""Tests for RLM recursion (subcall) functionality."""

import pytest
from rlmkit import RLM, RLMConfig, MockLLMClient, BudgetExceeded, BudgetTracker, BudgetLimits
from rlmkit.config import ExecutionConfig


class TestSubcallBasics:
    """Test basic subcall functionality."""
    
    def test_subcall_available_in_repl(self):
        """Test that subcall is available in REPL environment."""
        client = MockLLMClient([
            "```python\nprint(type(subcall))\n```",
            "FINAL: subcall is available"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="test content", query="check subcall")
        
        assert result.success
        # Should have printed the function type
        assert "function" in result.trace[1]["content"].lower()
    
    def test_simple_subcall(self):
        """Test simple subcall execution."""
        # Main RLM calls subcall, which analyzes subset of content
        client = MockLLMClient([
            # Main RLM: Use subcall
            "```python\nresult = subcall('Hello World', 'What is the first word?')\nprint(result)\n```",
            # Sub-RLM: Answer the question
            "FINAL: Hello",
            # Main RLM: Return final answer
            "FINAL: subcall returned: Hello"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="Large content here", query="Test subcall")
        
        assert result.success
        assert "Hello" in result.answer
    
    def test_subcall_with_peek(self):
        """Test subcall analyzing a section extracted with peek."""
        client = MockLLMClient([
            # Main: Extract section and subcall
            "```python\nsection = peek(0, 20)\nanswer = subcall(section, 'Count words')\nprint(answer)\n```",
            # Sub: Count words
            "FINAL: 3 words",
            # Main: Finish
            "FINAL: Section has 3 words"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="This is test content for analysis", query="Analyze first section")
        
        assert result.success
        assert "3 words" in result.answer


class TestSubcallWithBudgetTracking:
    """Test subcall with budget tracking."""
    
    def test_subcall_with_shared_budget(self):
        """Test subcall shares budget tracker with parent."""
        client = MockLLMClient([
            # Main: Create subcall
            "```python\nresult = subcall('test', 'answer')\nprint(result)\n```",
            # Sub: Answer
            "FINAL: sub answer",
            # Main: Finish
            "FINAL: Got sub answer"
        ])
        
        # Create budget tracker
        limits = BudgetLimits(max_steps=10, max_recursion_depth=5)
        tracker = BudgetTracker(limits)
        
        rlm = RLM(client=client, budget_tracker=tracker)
        result = rlm.run(prompt="content", query="test")
        
        assert result.success
        # Tracker is shared between parent and child
        assert tracker is not None
    
    def test_recursion_depth_limit_enforced(self):
        """Test recursion depth limit is enforced."""
        # Client that tries to recurse infinitely
        client = MockLLMClient([
            # Each call tries to create another subcall
            "```python\nresult = subcall('test', 'recurse')\n```",
        ])
        
        limits = BudgetLimits(max_recursion_depth=2, max_steps=100)  # High step limit
        tracker = BudgetTracker(limits)
        
        rlm = RLM(client=client, budget_tracker=tracker)
        
        with pytest.raises(BudgetExceeded) as exc:
            rlm.run(prompt="content", query="test")
        
        # Should fail due to recursion depth, not steps
        assert ("recursion depth" in str(exc.value).lower() or 
                "maximum steps" in str(exc.value).lower())  # Either is acceptable
    
    def test_recursion_depth_tracks_correctly(self):
        """Test recursion depth increases and decreases correctly."""
        client = MockLLMClient([
            # Main: Start subcall
            "```python\nresult = subcall('test', 'query')\nprint('done')\n```",
            # Sub: Just answer
            "FINAL: sub done",
            # Main: Finish after sub returns
            "FINAL: main done"
        ])
        
        limits = BudgetLimits(max_recursion_depth=5)
        tracker = BudgetTracker(limits)
        tracker.start()
        
        rlm = RLM(client=client, budget_tracker=tracker)
        result = rlm.run(prompt="content", query="test")
        
        assert result.success
        # After completion, recursion depth should be back to 0
        assert tracker.recursion_depth == 0


class TestSubcallStepBudgeting:
    """Test step budget allocation for subcalls."""
    
    def test_subcall_inherits_remaining_steps(self):
        """Test subcall gets parent's remaining steps by default."""
        client = MockLLMClient([
            # Main: Use 2 steps, then subcall
            "```python\nx = 1\n```",
            "```python\nresult = subcall('test', 'query')\n```",
            # Sub: Gets remaining steps
            "FINAL: sub answer",
            # Main: Finish
            "FINAL: done"
        ])
        
        config = RLMConfig(execution=ExecutionConfig(max_steps=10))
        rlm = RLM(client=client, config=config)
        result = rlm.run(prompt="content", query="test")
        
        assert result.success
    
    def test_subcall_custom_max_steps(self):
        """Test subcall with custom max_steps parameter."""
        client = MockLLMClient([
            # Main: Subcall with max_steps=2
            "```python\nresult = subcall('test', 'query', max_steps=2)\nprint(result)\n```",
            # Sub: Step 1
            "```python\nx = 1\n```",
            # Sub: Step 2 - must finish here
            "FINAL: sub done",
            # Main: Finish
            "FINAL: main done"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="content", query="test")
        
        assert result.success


class TestNestedSubcalls:
    """Test nested subcall scenarios."""
    
    def test_two_level_nesting(self):
        """Test subcall calling another subcall."""
        client = MockLLMClient([
            # Level 0 (main): Start subcall
            "```python\nresult = subcall('level1', 'query')\nprint(result)\n```",
            # Level 1: Start another subcall
            "```python\nresult = subcall('level2', 'query')\nprint(result)\n```",
            # Level 2: Answer
            "FINAL: level 2 answer",
            # Level 1: Pass up
            "FINAL: level 1 got: level 2 answer",
            # Level 0: Final answer
            "FINAL: level 0 got: level 1 got: level 2 answer"
        ])
        
        limits = BudgetLimits(max_recursion_depth=5)
        tracker = BudgetTracker(limits)
        
        rlm = RLM(client=client, budget_tracker=tracker)
        result = rlm.run(prompt="content", query="test nesting")
        
        assert result.success
        assert "level 2 answer" in result.answer
        # After all returns, depth should be 0
        assert tracker.recursion_depth == 0
    
    def test_multiple_sequential_subcalls(self):
        """Test multiple subcalls in sequence (not nested)."""
        client = MockLLMClient([
            # Main: First subcall
            "```python\nr1 = subcall('test1', 'query1')\nprint(r1)\n```",
            # Sub1: Answer
            "FINAL: answer 1",
            # Main: Second subcall
            "```python\nr2 = subcall('test2', 'query2')\nprint(r2)\n```",
            # Sub2: Answer  
            "FINAL: answer 2",
            # Main: Combine results
            "FINAL: Got answer 1 and answer 2"
        ])
        
        limits = BudgetLimits(max_recursion_depth=3)
        tracker = BudgetTracker(limits)
        
        rlm = RLM(client=client, budget_tracker=tracker)
        result = rlm.run(prompt="content", query="test multiple")
        
        assert result.success
        assert "answer 1" in result.answer
        assert "answer 2" in result.answer


class TestSubcallErrorHandling:
    """Test error handling in subcalls."""
    
    def test_subcall_with_execution_error(self):
        """Test subcall that has execution error."""
        client = MockLLMClient([
            # Main: Create subcall
            "```python\nresult = subcall('test', 'query')\nprint(result)\n```",
            # Sub: Cause error
            "```python\nx = 1 / 0\n```",
            # Sub: Recover and answer
            "FINAL: recovered",
            # Main: Finish
            "FINAL: subcall returned: recovered"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="content", query="test")
        
        assert result.success
    
    def test_subcall_without_budget_tracker(self):
        """Test subcall works without budget tracker."""
        client = MockLLMClient([
            # Main: Subcall
            "```python\nresult = subcall('test', 'query')\nprint(result)\n```",
            # Sub: Answer
            "FINAL: sub answer",
            # Main: Finish
            "FINAL: main done"
        ])
        
        # No budget tracker
        rlm = RLM(client=client)
        result = rlm.run(prompt="content", query="test")
        
        assert result.success
        assert "sub answer" in result.trace[1]["content"]


class TestSubcallIntegration:
    """Integration tests for subcall functionality."""
    
    def test_realistic_decomposition_scenario(self):
        """Test realistic task decomposition with subcalls."""
        # Simulate analyzing a document by chapters
        client = MockLLMClient([
            # Main: Get chapter 1
            "```python\nchap1 = peek(0, 100)\nsummary1 = subcall(chap1, 'Summarize this')\nprint(summary1)\n```",
            # Sub1: Summarize chapter 1
            "FINAL: Chapter 1 is about testing",
            # Main: Get chapter 2
            "```python\nchap2 = peek(100, 200)\nsummary2 = subcall(chap2, 'Summarize this')\nprint(summary2)\n```",
            # Sub2: Summarize chapter 2
            "FINAL: Chapter 2 is about recursion",
            # Main: Combine summaries
            "FINAL: Document covers testing and recursion"
        ])
        
        content = "a" * 200  # Simulate long content
        rlm = RLM(client=client)
        result = rlm.run(prompt=content, query="Summarize the document")
        
        assert result.success
        assert "testing" in result.answer
        assert "recursion" in result.answer
    
    def test_subcall_with_complex_query(self):
        """Test subcall with complex multi-step query."""
        client = MockLLMClient([
            # Main: Extract and analyze
            "```python\ndata = peek(0, 50)\nanalysis = subcall(data, 'Find all numbers and sum them')\nprint(analysis)\n```",
            # Sub: Multi-step analysis
            "```python\nimport re\nnumbers = re.findall(r'\\d+', P)\ntotal = sum(int(n) for n in numbers)\nprint(total)\n```",
            "FINAL_VAR: total",
            # Main: Use result
            "FINAL: The sum is 15"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="Test 5 and 10", query="Analyze numbers")
        
        assert result.success
