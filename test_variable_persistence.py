# Test variable persistence in different execution modes

from rlmkit.core.rlm import RLM
from rlmkit.llm import MockLLMClient
from rlmkit.config import RLMConfig, ExecutionConfig
import threading

def test_variable_persistence():
    """Test that variables persist between executions."""
    
    print("=" * 70)
    print("Testing variable persistence")
    print(f"Main thread: {threading.current_thread() == threading.main_thread()}")
    print("=" * 70)
    
    responses = [
        # Step 1: Set variable
        "```python\nresult = 9 + 8 * 55\nprint(f'result = {result}')\n```",
        
        # Step 2: Use FINAL_VAR to reference it
        "FINAL_VAR: result"
    ]
    
    client = MockLLMClient(responses)
    config = RLMConfig(execution=ExecutionConfig(max_steps=10))
    rlm = RLM(client=client, config=config)
    
    result = rlm.run(
        prompt="Calculate",
        query="What is 9+8*55?"
    )
    
    print(f"\n[RESULT] Success: {result.success}")
    print(f"[RESULT] Answer: {result.answer}")
    print(f"[RESULT] Error: {result.error}")
    print(f"[RESULT] Steps: {result.steps}")
    
    print("\nTrace:")
    for step in result.trace:
        role = step.get('role')
        content = step.get('content', '')[:100]
        print(f"  {role}: {content}")
    
    # Check if error about variable not found
    if "not found" in result.answer.lower():
        print("\n[FAIL] Variable persistence broken!")
        return False
    else:
        print("\n[PASS] Variable persistence working!")
        return True

if __name__ == "__main__":
    success = test_variable_persistence()
    exit(0 if success else 1)
