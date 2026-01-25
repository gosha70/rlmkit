"""Tests for RLM controller."""

import pytest
from rlmkit import RLM, RLMConfig, MockLLMClient, BudgetExceeded


class TestBasicRLMExecution:
    """Test basic RLM execution scenarios."""
    
    def test_simple_code_execution(self):
        """Test RLM executes code and gets result."""
        client = MockLLMClient([
            "```python\nx = peek(0, 10)\nprint(x)\n```",
            "FINAL: The first 10 characters"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(
            prompt="Hello, World! This is a test.",
            query="What are the first 10 characters?"
        )
        
        assert result.success
        assert result.answer == "The first 10 characters"
        assert result.steps == 2
        assert len(result.trace) == 3  # 2 assistant + 1 execution
    
    def test_direct_final_answer(self):
        """Test RLM with immediate FINAL answer (no code)."""
        client = MockLLMClient([
            "FINAL: This is the answer"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(
            prompt="Some content",
            query="What's the answer?"
        )
        
        assert result.success
        assert result.answer == "This is the answer"
        assert result.steps == 1
    
    def test_final_var_extraction(self):
        """Test FINAL_VAR extracts variable from environment."""
        client = MockLLMClient([
            "```python\nmy_result = peek(0, 5)\n```",
            "FINAL_VAR: my_result"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(
            prompt="Hello World",
            query="Get first 5 chars"
        )
        
        assert result.success
        assert result.answer == "Hello"
        assert result.steps == 2
    
    def test_multiple_execution_steps(self):
        """Test RLM with multiple code execution steps."""
        client = MockLLMClient([
            "```python\nmatches = grep(r'test')\nprint(len(matches))\n```",
            "```python\nfirst_match = matches[0] if matches else None\nprint(first_match)\n```",
            "FINAL: Found test pattern"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(
            prompt="This is a test. Another test here.",
            query="Find 'test' pattern"
        )
        
        assert result.success
        assert result.answer == "Found test pattern"
        assert result.steps == 3


class TestBudgetEnforcement:
    """Test budget and limit enforcement."""
    
    def test_max_steps_exceeded(self):
        """Test BudgetExceeded raised when max_steps reached."""
        # Client that always returns code (never FINAL)
        client = MockLLMClient([
            "```python\nx = 1\n```"
        ])
        
        config = RLMConfig()
        config.execution.max_steps = 3
        
        rlm = RLM(client=client, config=config)
        
        with pytest.raises(BudgetExceeded) as exc_info:
            rlm.run(prompt="test", query="test")
        
        assert "Maximum steps (3) exceeded" in str(exc_info.value)
    
    def test_stops_at_final_before_budget(self):
        """Test RLM stops when FINAL found, even with budget remaining."""
        client = MockLLMClient([
            "```python\nx = peek(0, 5)\n```",
            "FINAL: Done"
        ])
        
        config = RLMConfig()
        config.execution.max_steps = 100  # Large budget
        
        rlm = RLM(client=client, config=config)
        result = rlm.run(prompt="test", query="test")
        
        assert result.success
        assert result.steps == 2  # Stopped early
        assert result.answer == "Done"


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_code_execution_error(self):
        """Test RLM rejects FINAL after execution errors."""
        client = MockLLMClient([
            "```python\nx = 1 / 0\n```",  # Will cause error
            "FINAL: Handled error",  # This should be REJECTED
            "Let me fix: ```python\nx = 1 + 1\nprint(x)\n```",  # Fixed code
            "FINAL: The result is 2"  # This should be ACCEPTED
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="test", query="test")
        
        # Should eventually succeed after fixing the error
        assert result.success
        assert "result is 2" in result.answer or "2" in result.answer
        
        # Should have rejected first FINAL attempt
        rejections = [t for t in result.trace if t.get("role") == "system" and "Rejected" in t.get("content", "")]
        assert len(rejections) >= 1, "Should reject FINAL after execution error"
    
    def test_missing_final_var(self):
        """Test error when FINAL_VAR references non-existent variable."""
        client = MockLLMClient([
            "FINAL_VAR: nonexistent_variable"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="test", query="test")
        
        assert result.success
        assert "not found" in result.answer.lower()
    
    def test_unclear_response_handling(self):
        """Test RLM prompts for clarity when no code or FINAL."""
        client = MockLLMClient([
            "I'm thinking about it...",  # No code, no FINAL
            "FINAL: Here's my answer"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="test", query="test")
        
        assert result.success
        assert result.answer == "Here's my answer"


class TestMessageHistory:
    """Test message history management."""
    
    def test_message_history_structure(self):
        """Test that messages are properly structured."""
        client = MockLLMClient([
            "```python\nx = peek(0, 5)\n```",
            "FINAL: Done"
        ])
        
        rlm = RLM(client=client)
        rlm.run(prompt="test content", query="test query")
        
        # Check call history
        assert len(client.call_history) == 2
        
        # First call: system + user
        first_call = client.call_history[0]
        assert len(first_call) == 2
        assert first_call[0]["role"] == "system"
        assert first_call[1]["role"] == "user"
        assert first_call[1]["content"] == "test query"
        
        # Second call: includes assistant response and execution result
        second_call = client.call_history[1]
        assert len(second_call) == 4  # system + user + assistant + user(execution)
        assert second_call[2]["role"] == "assistant"
        assert second_call[3]["role"] == "user"
        assert "Execution result:" in second_call[3]["content"]
    
    def test_system_prompt_includes_prompt_length(self):
        """Test system prompt mentions prompt length."""
        client = MockLLMClient(["FINAL: Test"])
        
        rlm = RLM(client=client)
        rlm.run(prompt="x" * 12345, query="test")
        
        # Check system prompt
        system_msg = client.call_history[0][0]
        assert "12,345" in system_msg["content"]  # Formatted with commas


class TestREPLIntegration:
    """Test integration with PyReplEnv."""
    
    def test_content_set_as_p(self):
        """Test prompt is set as 'P' variable in REPL."""
        client = MockLLMClient([
            "```python\nprint(len(P))\n```",
            "FINAL: Length checked"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="Hello World", query="Check length")
        
        assert result.success
        # Execution should have access to P
        exec_trace = [t for t in result.trace if t.get("role") == "execution"]
        assert len(exec_trace) == 1
        assert "11" in exec_trace[0]["content"]  # len("Hello World")
    
    def test_tools_available_in_repl(self):
        """Test peek, grep, etc. are available."""
        client = MockLLMClient([
            "```python\nresult = peek(0, 5)\nprint(result)\n```",
            "FINAL: Tools work"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="Testing tools", query="Test peek")
        
        assert result.success
        exec_trace = [t for t in result.trace if t.get("role") == "execution"]
        assert "Testi" in exec_trace[0]["content"]
    
    def test_variables_persist_across_steps(self):
        """Test variables persist between execution steps."""
        client = MockLLMClient([
            "```python\nmy_var = peek(0, 5)\nprint(my_var)\n```",
            "```python\nprint(my_var.upper())\n```",  # Use my_var from previous step
            "FINAL: Persistence works"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="hello world", query="Test persistence")
        
        assert result.success
        exec_traces = [t for t in result.trace if t.get("role") == "execution"]
        assert len(exec_traces) == 2
        assert "hello" in exec_traces[0]["content"]
        assert "HELLO" in exec_traces[1]["content"]


class TestCustomSystemPrompt:
    """Test custom system prompt functionality."""
    
    def test_custom_system_prompt(self):
        """Test RLM accepts custom system prompt."""
        client = MockLLMClient(["FINAL: Custom prompt test"])
        
        custom_prompt = "You are a test assistant. Answer everything with 'test'."
        
        rlm = RLM(client=client)
        result = rlm.run(
            prompt="content",
            query="query",
            system_prompt=custom_prompt
        )
        
        assert result.success
        
        # Check custom prompt was used
        system_msg = client.call_history[0][0]
        assert system_msg["content"] == custom_prompt


class TestTraceLogging:
    """Test execution trace logging."""
    
    def test_trace_includes_all_steps(self):
        """Test trace includes all execution steps."""
        client = MockLLMClient([
            "```python\nx = 1\n```",
            "```python\ny = 2\n```",
            "FINAL: Done"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="test", query="test")
        
        # Trace should have: assistant, execution, assistant, execution, assistant
        assert len(result.trace) == 5
        assert result.trace[0]["role"] == "assistant"
        assert result.trace[1]["role"] == "execution"
        assert result.trace[2]["role"] == "assistant"
        assert result.trace[3]["role"] == "execution"
        assert result.trace[4]["role"] == "assistant"
    
    def test_trace_includes_step_numbers(self):
        """Test trace entries include step numbers."""
        client = MockLLMClient([
            "```python\nx = 1\n```",
            "FINAL: Done"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="test", query="test")
        
        assert result.trace[0]["step"] == 1
        assert result.trace[1]["step"] == 1
        assert result.trace[2]["step"] == 2
    
    def test_trace_includes_raw_results(self):
        """Test execution trace includes raw results."""
        client = MockLLMClient([
            "```python\nprint('test')\n```",
            "FINAL: Done"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="test", query="test")
        
        exec_trace = [t for t in result.trace if t.get("role") == "execution"]
        assert len(exec_trace) == 1
        assert "raw_result" in exec_trace[0]
        assert exec_trace[0]["raw_result"]["stdout"] == "test\n"


class TestRLMResult:
    """Test RLMResult dataclass."""
    
    def test_result_success_true(self):
        """Test successful execution sets success=True."""
        client = MockLLMClient(["FINAL: Success"])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="test", query="test")
        
        assert result.success is True
        assert result.error is None
    
    def test_result_includes_steps(self):
        """Test result includes step count."""
        client = MockLLMClient([
            "```python\nx = 1\n```",
            "```python\ny = 2\n```",
            "FINAL: Done"
        ])
        
        rlm = RLM(client=client)
        result = rlm.run(prompt="test", query="test")
        
        assert result.steps == 3
