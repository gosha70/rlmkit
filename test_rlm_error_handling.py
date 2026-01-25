"""
Test to demonstrate the RLM error handling bug.

The issue: When code execution fails, the LLM can still provide
a FINAL answer based on what it *thinks* the code should produce,
rather than acknowledging the execution failure.
"""

from rlmkit.core.rlm import RLM
from rlmkit.llm import MockLLMClient
from rlmkit.config import RLMConfig


def test_error_then_final_answer():
    """
    Reproduces the bug where:
    1. LLM provides code that will fail
    2. Code execution fails with exception
    3. LLM ignores error and provides FINAL answer anyway
    
    Expected behavior: RLM should detect that the LLM is answering
    without successfully executing its code, and prompt for acknowledgment.
    """
    
    # Mock responses simulating the bug
    responses = [
        # Step 1: LLM writes code to calculate
        "```python\nresult = 7 * 9 + 1\nresult\n```",
        
        # Step 2: After seeing execution error, LLM provides final answer anyway
        # This is wrong because 'result' variable was never set due to error
        "FINAL: Based on my calculation, the answer is 64."
    ]
    
    client = MockLLMClient(responses)
    config = RLMConfig(max_steps=10)
    rlm = RLM(client=client, config=config)
    
    # Simulate a simple math question
    result = rlm.run(
        prompt="Calculate 7 * 9 + 1",
        query="What is 7 * 9 + 1?"
    )
    
    print(f"Result success: {result.success}")
    print(f"Result answer: {result.answer}")
    print(f"Result steps: {result.steps}")
    print("\nTrace:")
    for i, step in enumerate(result.trace):
        print(f"\nStep {i+1} ({step.get('role', 'unknown')}):")
        content = step.get('content', '')
        if len(content) > 200:
            content = content[:200] + "..."
        print(content)
    
    # The bug is that result.success is True even though:
    # 1. Code execution failed
    # 2. LLM never successfully computed the result
    # 3. Answer is based on imagination, not execution
    
    return result


if __name__ == "__main__":
    result = test_error_then_final_answer()
