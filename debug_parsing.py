# Test to understand parsing behavior

from rlmkit.core.parsing import parse_response

# Simulate various LLM responses

# Test 1: Code with FINAL in same response
response1 = """Let me calculate:
```python
result = 8 + 9/2
result
```

FINAL: Based on my analysis, the answer is 12.5"""

parsed1 = parse_response(response1)
print("Test 1: Code + FINAL in same response")
print(f"  has_code: {parsed1.has_code}")
print(f"  has_final: {parsed1.has_final}")
print(f"  is_complete: {parsed1.is_complete}")
print(f"  code: {parsed1.code}")
print(f"  final_answer: {parsed1.final_answer}")
print()

# Test 2: Just FINAL
response2 = "FINAL: Based on my analysis, the answer is 12.5"
parsed2 = parse_response(response2)
print("Test 2: Just FINAL")
print(f"  has_code: {parsed2.has_code}")
print(f"  has_final: {parsed2.has_final}")
print(f"  is_complete: {parsed2.is_complete}")
print(f"  final_answer: {parsed2.final_answer}")
print()

# Test 3: Text response
response3 = "I apologize for the confusion. Let me recalculate..."
parsed3 = parse_response(response3)
print("Test 3: Text response")
print(f"  has_code: {parsed3.has_code}")
print(f"  has_final: {parsed3.has_final}")
print(f"  is_complete: {parsed3.is_complete}")
print()
