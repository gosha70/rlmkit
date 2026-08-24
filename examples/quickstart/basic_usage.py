"""
Basic usage examples for RLM Studio's unified interact() API.

This demonstrates the three main interaction modes:
1. Direct - Full context in one call
2. Direct (larger document) - Same mode, bigger content
3. RLM - Recursive exploration with code generation
4. Auto - Automatically chooses the best mode
"""

from rlmstudio import interact

# Sample content for demonstrations
SHORT_CONTENT = """
RLM Studio is a Python toolkit for building LLM-based systems that can handle
arbitrarily large contexts through code generation and recursive exploration.
It provides three interaction modes: Direct, RAG, and RLM.
"""

MEDIUM_CONTENT = (
    """
# RLM Studio Documentation

## Introduction
RLM Studio is a Recursive Language Model toolkit that enables LLMs to handle
arbitrarily large contexts by treating the prompt as an external environment.

## Features
- **Direct Mode**: Traditional LLM interaction with full context
- **RAG Mode**: Retrieval-Augmented Generation with automatic chunking
- **RLM Mode**: Recursive exploration using code generation
- **Auto Mode**: Automatically selects the best strategy

## Installation
pip install rlmstudio

## Quick Start
from rlmstudio import interact
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
"""
    * 50
)  # Repeat to make it larger


def example_1_direct_mode() -> None:
    """Example 1: Direct mode for small content."""
    print("=" * 70)
    print("Example 1: Direct Mode (Small Content)")
    print("=" * 70)

    result = interact(
        content=SHORT_CONTENT,
        query="What is RLM Studio?",
        mode="direct",
        provider="openai",
        model="gpt-4o-mini",  # Using mini for cost savings
    )

    print(f"\nMode Used: {result.mode_used}")
    print(f"Answer: {result.answer}")
    print("\nMetrics:")
    print(f"  Tokens: {result.total_tokens:,}")
    print(f"  Cost: ${result.total_cost:.4f}")
    print(f"  Time: {result.elapsed_time:.2f}s")


def example_2_direct_mode_large() -> None:
    """Example 2: Direct mode with a larger document."""
    print("\n" + "=" * 70)
    print("Example 2: Direct mode with a larger document")
    print("=" * 70)

    result = interact(
        content=MEDIUM_CONTENT,
        query="What are the main features of RLM Studio?",
        mode="direct",
        provider="openai",
        model="gpt-4o-mini",
    )

    print(f"\nMode Used: {result.mode_used}")
    print(f"Answer: {result.answer}")
    print("\nMetrics:")
    print(f"  Tokens: {result.total_tokens:,}")
    print(f"  Cost: ${result.total_cost:.4f}")
    print(f"  Time: {result.elapsed_time:.2f}s")


def example_3_rlm_mode() -> None:
    """Example 3: RLM mode for exploration."""
    print("\n" + "=" * 70)
    print("Example 3: RLM Mode (Recursive Exploration)")
    print("=" * 70)

    result = interact(
        content=MEDIUM_CONTENT,
        query="How do I install and use RLM Studio?",
        mode="rlm",
        provider="openai",
        model="gpt-4o-mini",
        verbose=True,  # Show execution steps
    )

    print(f"\nMode Used: {result.mode_used}")
    print(f"Answer: {result.answer}")
    print("\nMetrics:")
    print(f"  Tokens: {result.total_tokens:,}")
    print(f"  Cost: ${result.total_cost:.4f}")
    print(f"  Steps: {result.steps}")
    print(f"  Time: {result.elapsed_time:.2f}s")


def example_4_auto_mode() -> None:
    """Example 4: Auto mode - let RLM Studio choose."""
    print("\n" + "=" * 70)
    print("Example 4: Auto Mode (Automatic Selection)")
    print("=" * 70)

    # Will automatically choose based on content size
    result = interact(
        content=SHORT_CONTENT,
        query="What modes does RLM Studio support?",
        mode="auto",  # Let RLM Studio decide
        provider="openai",
        model="gpt-4o-mini",
        verbose=True,
    )

    print(f"\nAuto-selected Mode: {result.mode_used}")
    print(f"Answer: {result.answer}")
    print("\nMetrics:")
    print(f"  Tokens: {result.total_tokens:,}")
    print(f"  Cost: ${result.total_cost:.4f}")


def example_5_simple_completion() -> None:
    """Example 5: Simple complete() wrapper."""
    print("\n" + "=" * 70)
    print("Example 5: Simple complete() Function")
    print("=" * 70)

    from rlmstudio import complete

    # Just get the answer string
    answer = complete(content=SHORT_CONTENT, query="List the interaction modes", mode="direct")

    print(f"Answer: {answer}")


def example_6_error_handling() -> None:
    """Example 6: Error handling."""
    print("\n" + "=" * 70)
    print("Example 6: Error Handling")
    print("=" * 70)

    try:
        # Empty content should raise ValueError
        interact(content="", query="What is this?", mode="direct")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        # Invalid mode should raise ValueError
        interact(content="Some content", query="What is this?", mode="invalid_mode")
    except ValueError as e:
        print(f"Caught expected error: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RLM Studio Unified API Examples")
    print("=" * 70)
    print("\nNote: These examples require OPENAI_API_KEY environment variable")
    print("Set it with: export OPENAI_API_KEY=sk-...")
    print("\n")

    # Run all examples
    # Comment out examples you don't want to run

    try:
        example_1_direct_mode()
        example_2_direct_mode_large()
        example_3_rlm_mode()
        example_4_auto_mode()
        example_5_simple_completion()
        example_6_error_handling()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except ImportError as e:
        print(f"\n Import Error: {e}")
        print("Make sure to install RLM Studio and dependencies:")
        print("  pip install -e '.[dev]'")

    except Exception as e:
        print(f"\n Error running examples: {e}")
        print("Make sure OPENAI_API_KEY is set and valid.")
