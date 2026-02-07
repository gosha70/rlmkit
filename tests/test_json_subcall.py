"""
Test JSON-based subcall action (v2.0 protocol).

This tests the newly wired subcall handler that executes subcalls
via the JSON action protocol.
"""

import pytest
from rlmkit import RLM, RLMConfig
from rlmkit.llm import MockLLMClient


class TestJSONSubcallAction:
    """Test JSON v2.0 subcall action."""
    
    def test_json_subcall_action(self):
        """Test subcall via JSON action protocol."""
        # Mock LLM that returns JSON subcall action, then final answer
        responses = [
            # Step 1: Return a JSON subcall action
            '''{
                "action": "subcall",
                "query": "What is in section 1?",
                "prompt": "Section 1: Introduction\\nThis is the introduction.",
                "note": "Analyzing section 1"
            }''',
            # Step 2: After subcall returns, provide final answer
            "FINAL: The document has an introduction section."
        ]
        
        client = MockLLMClient(responses)
        config = RLMConfig()
        config.execution.max_steps = 10
        rlm = RLM(client=client, config=config)
        
        # Run with content that could benefit from subcall
        result = rlm.run(
            prompt="Section 1: Introduction\nThis is the introduction.\n\nSection 2: Main content here.",
            query="What sections does this document have?"
        )
        
        assert result.success
        assert "introduction" in result.answer.lower()
        assert result.steps == 2
        
        # Check trace contains subcall execution
        assert any("subcall" in str(item.get("content", "")).lower() 
                   for item in result.trace)
    
    def test_json_subcall_with_error_handling(self):
        """Test that subcall errors are handled gracefully."""
        # Mock LLM that returns invalid subcall action
        responses = [
            # Step 1: Return subcall with missing fields (should error)
            '''{
                "action": "subcall",
                "query": "Test query"
            }''',  # Missing 'prompt' field
            # Step 2: Recover with direct answer
            "FINAL: Could not complete subcall, but here's my answer: The document is about testing."
        ]
        
        client = MockLLMClient(responses)
        config = RLMConfig()
        config.execution.max_steps = 10
        rlm = RLM(client=client, config=config)
        
        result = rlm.run(
            prompt="Test content about error handling.",
            query="What is this about?"
        )
        
        # Should still succeed with final answer
        assert result.success
        assert "testing" in result.answer.lower()
    
    def test_json_subcall_result_in_trace(self):
        """Test that subcall results appear in execution trace."""
        # Mock LLM responses
        responses = [
            # Subcall action
            '''{
                "action": "subcall",
                "query": "Summarize this",
                "prompt": "Content to summarize: The quick brown fox."
            }''',
            # Final answer using subcall result
            "FINAL: Based on the subcall, the content mentions a fox."
        ]
        
        client = MockLLMClient(responses)
        config = RLMConfig()
        config.execution.max_steps = 10
        rlm = RLM(client=client, config=config)
        
        result = rlm.run(
            prompt="Full document content here.",
            query="What does the document say?"
        )
        
        assert result.success
        assert len(result.trace) >= 2
        
        # Check that subcall appears in trace (either in content or as separate item)
        trace_str = str(result.trace)
        assert "subcall" in trace_str.lower() or "fox" in trace_str.lower()
    
    def test_json_subcall_vs_direct_python_subcall(self):
        """Compare JSON subcall action vs direct Python subcall."""
        # Both should work identically
        
        # Test 1: JSON action
        json_responses = [
            '''{
                "action": "subcall",
                "query": "Count words",
                "prompt": "One two three"
            }''',
            "FINAL: Three words found."
        ]
        
        json_client = MockLLMClient(json_responses)
        json_config = RLMConfig()
        json_config.execution.max_steps = 10
        json_rlm = RLM(client=json_client, config=json_config)
        
        json_result = json_rlm.run(
            prompt="Test document",
            query="How many words?"
        )
        
        # Test 2: Direct Python code (old way)
        python_responses = [
            '''```python
result = subcall(query="Count words", prompt="One two three")
print(result)
```''',
            "FINAL: Three words found."
        ]
        
        python_client = MockLLMClient(python_responses)
        python_config = RLMConfig()
        python_config.execution.max_steps = 10
        python_rlm = RLM(client=python_client, config=python_config)
        
        python_result = python_rlm.run(
            prompt="Test document",
            query="How many words?"
        )
        
        # Both should succeed
        assert json_result.success
        assert python_result.success
        
        # Both should have similar structure
        assert json_result.steps == python_result.steps


class TestJSONSubcallIntegration:
    """Integration tests for JSON subcalls."""
    
    def test_realistic_json_subcall_scenario(self):
        """Test a realistic scenario using JSON subcalls."""
        # Simulate LLM that uses inspect, then subcall, then final
        responses = [
            # Step 1: Inspect with JSON
            '''{
                "action": "inspect",
                "tool": "peek",
                "args": {"start": 0, "end": 100}
            }''',
            # Step 2: Use subcall to analyze a section
            '''{
                "action": "subcall",
                "query": "Extract the date",
                "prompt": "Date: 2026-02-06\\nAuthor: Test",
                "note": "Getting metadata"
            }''',
            # Step 3: Final answer
            "FINAL: Document dated 2026-02-06 by Test author."
        ]
        
        client = MockLLMClient(responses)
        config = RLMConfig()
        config.execution.max_steps = 10
        rlm = RLM(client=client, config=config)
        
        result = rlm.run(
            prompt="Date: 2026-02-06\nAuthor: Test\n\nBody: Lorem ipsum...",
            query="When was this written and by whom?"
        )
        
        assert result.success
        assert "2026" in result.answer
        assert result.steps == 3
        
        # Verify trace contains both inspect and subcall
        trace_str = str(result.trace)
        assert "peek" in trace_str.lower()
        assert "subcall" in trace_str.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
