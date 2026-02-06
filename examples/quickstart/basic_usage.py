"""
Basic usage examples for RLMKit's unified interact() API.

This demonstrates the three main interaction modes:
1. Direct - Full context in one call
2. RAG - Retrieval-augmented generation
3. RLM - Recursive exploration with code generation
4. Auto - Automatically chooses the best mode
"""

from rlmkit import interact

# Sample content for demonstrations
SHORT_CONTENT = """
RLMKit is a Python toolkit for building LLM-based systems that can handle
arbitrarily large contexts through code generation and recursive exploration.
It provides three interaction modes: Direct, RAG, and RLM.
"""

MEDIUM_CONTENT = """
# RLMKit Documentation

## Introduction
RLMKit is a Recursive Language Model toolkit that enables LLMs to handle
arbitrarily large contexts by treating the prompt as an external environment.

## Features
- **Direct Mode**: Traditional LLM interaction with full context
- **RAG Mode**: Retrieval-Augmented Generation with automatic chunking
- **RLM Mode**: Recursive exploration using code generation
- **Auto Mode**: Automatically selects the best strategy

## Installation
pip install rlmkit

## Quick Start
from rlmkit import interact
result = interact("your content", "your query")
print(result.answer)

## Configuration
The toolkit supports multiple LLM providers:
- OpenAI (GPT-4, GPT-4o, etc.)
- Anthropic (Claude 3 Opus, Sonnet, etc.)
- Ollama (local models)
- LM Studio
- vLLM

## Use Cases
1. Long document Q&A
2. Code repository analysis
3. Research paper summarization
4. Multi-document synthesis
""" * 50  # Repeat to make it larger


def example_1_direct_mode():
    """Example 1: Direct mode for small content."""
    print("=" * 70)
    print("Example 1: Direct Mode (Small Content)")
    print("=" * 70)
    
    result = interact(
        content=SHORT_CONTENT,
        query="What is RLMKit?",
        mode="direct",
        provider="openai",
        model="gpt-4o-mini"  # Using mini for cost savings
    )
    
    print(f"\nMode Used: {result.mode_used}")
    print(f"Answer: {result.answer}")
    print(f"\nMetrics:")
    print(f"  Tokens: {result.metrics['total_tokens']:,}")
    print(f"  Cost: ${result.metrics['total_cost']:.4f}")
    print(f"  Time: {result.metrics['execution_time']:.2f}s")


def example_2_rag_mode():
    """Example 2: RAG mode for medium-sized content."""
    print("\n" + "=" * 70)
    print("Example 2: RAG Mode (Medium Content)")
    print("=" * 70)
    
    result = interact(
        content=MEDIUM_CONTENT,
        query="What are the main features of RLMKit?",
        mode="rag",
        provider="openai",
        model="gpt-4o-mini",
        top_k=3  # Retrieve top 3 chunks
    )
    
    print(f"\nMode Used: {result.mode_used}")
    print(f"Answer: {result.answer}")
    print(f"\nMetrics:")
    print(f"  Tokens: {result.metrics['total_tokens']:,}")
    print(f"  Cost: ${result.metrics['total_cost']:.4f}")
    print(f"  Time: {result.metrics['execution_time']:.2f}s")


def example_3_rlm_mode():
    """Example 3: RLM mode for exploration."""
    print("\n" + "=" * 70)
    print("Example 3: RLM Mode (Recursive Exploration)")
    print("=" * 70)
    
    result = interact(
        content=MEDIUM_CONTENT,
        query="How do I install and use RLMKit?",
        mode="rlm",
        provider="openai",
        model="gpt-4o-mini",
        verbose=True  # Show execution steps
    )
    
    print(f"\nMode Used: {result.mode_used}")
    print(f"Answer: {result.answer}")
    print(f"\nMetrics:")
    print(f"  Tokens: {result.metrics['total_tokens']:,}")
    print(f"  Cost: ${result.metrics['total_cost']:.4f}")
    print(f"  LLM Calls: {result.metrics['llm_calls']}")
    print(f"  Time: {result.metrics['execution_time']:.2f}s")


def example_4_auto_mode():
    """Example 4: Auto mode - let RLMKit choose."""
    print("\n" + "=" * 70)
    print("Example 4: Auto Mode (Automatic Selection)")
    print("=" * 70)
    
    # Will automatically choose based on content size
    result = interact(
        content=SHORT_CONTENT,
        query="What modes does RLMKit support?",
        mode="auto",  # Let RLMKit decide
        provider="openai",
        model="gpt-4o-mini",
        verbose=True
    )
    
    print(f"\nAuto-selected Mode: {result.mode_used}")
    print(f"Answer: {result.answer}")
    print(f"\nMetrics:")
    print(f"  Tokens: {result.metrics['total_tokens']:,}")
    print(f"  Cost: ${result.metrics['total_cost']:.4f}")


def example_5_simple_completion():
    """Example 5: Simple complete() wrapper."""
    print("\n" + "=" * 70)
    print("Example 5: Simple complete() Function")
    print("=" * 70)
    
    from rlmkit import complete
    
    # Just get the answer string
    answer = complete(
        content=SHORT_CONTENT,
        query="List the interaction modes",
        mode="direct"
    )
    
    print(f"Answer: {answer}")


def example_6_accessing_raw_result():
    """Example 6: Accessing the underlying strategy result."""
    print("\n" + "=" * 70)
    print("Example 6: Accessing Raw Strategy Result")
    print("=" * 70)
    
    result = interact(
        content=SHORT_CONTENT,
        query="What is RLMKit?",
        mode="direct"
    )
    
    # Access the underlying StrategyResult
    raw = result.raw_result
    print(f"Strategy Name: {raw.strategy_name}")
    print(f"Success: {raw.success}")
    print(f"Answer Length: {len(raw.answer)} characters")
    
    # Convert result to dict
    result_dict = result.to_dict()
    print(f"\nResult as Dict: {result_dict}")


def example_7_error_handling():
    """Example 7: Error handling."""
    print("\n" + "=" * 70)
    print("Example 7: Error Handling")
    print("=" * 70)
    
    try:
        # Empty content should raise ValueError
        result = interact(
            content="",
            query="What is this?",
            mode="direct"
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    try:
        # Invalid mode should raise ValueError
        result = interact(
            content="Some content",
            query="What is this?",
            mode="invalid_mode"
        )
    except ValueError as e:
        print(f"Caught expected error: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RLMKit Unified API Examples")
    print("=" * 70)
    print("\nNote: These examples require OPENAI_API_KEY environment variable")
    print("Set it with: export OPENAI_API_KEY=sk-...")
    print("\n")
    
    # Run all examples
    # Comment out examples you don't want to run
    
    try:
        example_1_direct_mode()
        example_2_rag_mode()
        example_3_rlm_mode()
        example_4_auto_mode()
        example_5_simple_completion()
        example_6_accessing_raw_result()
        example_7_error_handling()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("Make sure to install RLMKit and dependencies:")
        print("  pip install -e '.[dev]'")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure OPENAI_API_KEY is set and valid.")
