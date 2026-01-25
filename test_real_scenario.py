# -*- coding: utf-8 -*-
"""
Test matching the ACTUAL user scenario that was failing.

Step 1: LLM response: ```python result = 8 + 9/2 result
Step 2: Code execution: Exception: ValueError...
Step 3: LLM response: FINAL: Based on my analysis, the answer is 12.5
Step 4: (Warning sent)
Step 5: LLM response: I apologize for the confusion...
Step 6: LLM response: FINAL: Based on the order of operations...

Before fix: Step 6 FINAL would be accepted (WRONG!)
After fix: Step 6 FINAL should be rejected (CORRECT!)
"""

from rlmkit.core.rlm import RLM
from rlmkit.llm import MockLLMClient
from rlmkit.config import RLMConfig, ExecutionConfig


def test_user_scenario():
    """
    Test the exact scenario the user reported.
    
    The LLM writes code that fails, then tries multiple FINAL answers
    with text responses in between. All FINAL answers should be rejected
    until code executes successfully.
    """
    
    print("=" * 70)
    print("TESTING ACTUAL USER SCENARIO")
    print("=" * 70)
    
    responses = [
        # Step 1: Code that will fail (undefined variable)
        "```python\nresult = 8 + 9/2\nprint(undefined_var)\n```",
        
        # Step 3: First FINAL attempt after error (SHOULD BE REJECTED)
        "FINAL: Based on my analysis, the answer is 12.5. The operation 9/2 is performed first...",
        
        # Step 5: Text response (no code, no FINAL) - flag should STAY True
        "I apologize for the confusion. It seems there was a technical issue with running the calculation...",
        
        # Step 6: Second FINAL attempt (SHOULD ALSO BE REJECTED)
        "FINAL: Based on the order of operations (BODMAS/PEDMAS), the division operation is performed first...",
        
        # Step 7: Finally provides working code
        "Let me execute this correctly:\n```python\nresult = 8 + 9/2\nprint(result)\n```",
        
        # Step 8: After successful execution, provides FINAL
        "FINAL: The answer is 12.5"
    ]
    
    client = MockLLMClient(responses)
    config = RLMConfig(execution=ExecutionConfig(max_steps=30))
    rlm = RLM(client=client, config=config)
    
    result = rlm.run(
        prompt="Calculate 8 + 9/2",
        query="What is 8 + 9/2?"
    )
    
    print(f"\n[RESULT] Success: {result.success}")
    print(f"[RESULT] Answer: {result.answer}")
    print(f"[RESULT] Steps: {result.steps}")
    
    print("\nFull trace:")
    for i, step in enumerate(result.trace):
        role = step.get('role', 'unknown')
        content = step.get('content', '')
        
        if len(content) > 100:
            content = content[:100] + "..."
        
        print(f"  Step {i+1} [{role}]: {content}")
    
    # Count rejections
    rejection_count = sum(
        1 for step in result.trace 
        if step.get('role') == 'system' and 'Rejected FINAL' in step.get('content', '')
    )
    
    print(f"\n[CHECK] Number of FINAL rejections: {rejection_count}")
    print(f"[CHECK] Expected: At least 2 (one at step 3, one at step 6)")
    
    # Verify the fix worked
    assert result.success, "Execution should eventually succeed"
    assert rejection_count >= 2, f"Should reject FINAL at least 2 times, got {rejection_count}"
    assert "12.5" in result.answer, f"Final answer should be 12.5"
    
    print("\n" + "=" * 70)
    print("[PASS] FIX VERIFIED!")
    print("=" * 70)
    print(f"[PASS] Rejected {rejection_count} premature FINAL answers")
    print("[PASS] Only accepted FINAL after successful code execution")
    print("[PASS] Prevents hallucinated answers based on failed executions")
    print("=" * 70 + "\n")
    
    return result


if __name__ == "__main__":
    try:
        test_user_scenario()
        print("\n✓ TEST PASSED - Bug is fixed!\n")
        exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        exit(1)
