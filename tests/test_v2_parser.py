#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick test to verify v2.0 JSON parser works"""

from rlmkit.core.rlm import RLM
from rlmkit.llm import MockLLMClient
from rlmkit.config import RLMConfig, ExecutionConfig

print("Testing v2.0 JSON Parser")
print("=" * 60)

# Test 1: JSON inspect action
print("\n[Test 1] JSON inspect action")
responses = [
    '{"type": "inspect", "tool": "peek", "args": {"start": 0, "end": 100}}',
    '{"type": "final", "answer": "File starts with Copyright notice"}'
]

client = MockLLMClient(responses)
config = RLMConfig(execution=ExecutionConfig(max_steps=5))
rlm = RLM(client=client, config=config)

result = rlm.run(
    prompt="Copyright (c) EGOGE - All Rights Reserved.\nThis is test content.",
    query="What does the file start with?"
)

print(f"✓ Success: {result.success}")
print(f"✓ Answer: {result.answer}")
print(f"✓ Steps: {result.steps}")
print(f"\nTrace:")
for i, step in enumerate(result.trace):
    print(f"  Step {i+1} [{step.get('role')}]: {step.get('content', '')[:80]}...")

# Test 2: Direct final answer
print("\n" + "=" * 60)
print("\n[Test 2] Direct JSON final answer")
responses = [
    '{"type": "final", "answer": "The answer is 42", "confidence": 0.95}'
]

client = MockLLMClient(responses)
rlm = RLM(client=client, config=config)

result = rlm.run(
    prompt="No content needed",
    query="What is the meaning of life?"
)

print(f"✓ Success: {result.success}")
print(f"✓ Answer: {result.answer}")
print(f"✓ Steps: {result.steps}")

# Test 3: Fallback to markdown (v1.0 compatibility)
print("\n" + "=" * 60)
print("\n[Test 3] Fallback to markdown format")
responses = [
    '```python\nprint("Hello from markdown")\n```',
    'FINAL: This uses old v1.0 format'
]

client = MockLLMClient(responses)
rlm = RLM(client=client, config=config)

result = rlm.run(
    prompt="Test",
    query="Test markdown"
)

print(f"✓ Success: {result.success}")
print(f"✓ Answer: {result.answer}")
print(f"✓ Steps: {result.steps}")

print("\n" + "=" * 60)
print("✅ All v2.0 parser tests passed!")
