# Test simple math that shouldn't need code execution

from rlmkit.core.rlm import RLM
from rlmkit.llm import MockLLMClient
from rlmkit.config import RLMConfig, ExecutionConfig

# Test 1: Simple math, no code needed
print("Test 1: Simple math - direct answer (no code)")
client = MockLLMClient([
    "FINAL: 8 + 9 * 54 / 2 = 8 + (9 * 54 / 2) = 8 + 243 = 251"
])

config = RLMConfig(execution=ExecutionConfig(max_steps=20))
rlm = RLM(client=client, config=config)

try:
    result = rlm.run(
        prompt="No document content needed",
        query="What is 8+9*54/2?"
    )
    print(f"Success: {result.success}")
    print(f"Answer: {result.answer}")
    print(f"Steps: {result.steps}")
except Exception as e:
    print(f"FAILED: {e}")

print("\n" + "="*70 + "\n")

# Test 2: LLM tries code first (fails), then realizes no code needed
print("Test 2: LLM tries code, fails, then calculates without code")
client = MockLLMClient([
    "```python\nresult = undefined_var\n```",  # Tries code, fails
    "FINAL: The answer is 251",  # Should this be rejected?
])

rlm = RLM(client=client, config=config)

try:
    result = rlm.run(
        prompt="No document",
        query="What is 8+9*54/2?"
    )
    print(f"Success: {result.success}")
    print(f"Answer: {result.answer}")
    print(f"Steps: {result.steps}")
    
    print("\nTrace:")
    for step in result.trace:
        print(f"  {step.get('role')}: {step.get('content', '')[:80]}...")
        
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
