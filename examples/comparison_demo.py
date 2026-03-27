"""
Comparison demo: run RLM and Direct modes on the same content and compare metrics.

Usage:
    OPENAI_API_KEY=sk-... python examples/comparison_demo.py
"""

from rlmkit import interact

CONTENT = (
    """
# The Recursive Language Model Paper — Summary

## Abstract
The Recursive Language Model (RLM) addresses the fundamental challenge of
processing arbitrarily large inputs that exceed an LLM's context window.
Instead of truncating input or relying on retrieval heuristics, RLM gives
the LLM a Python sandbox where it can write code to explore the content
iteratively — inspecting sections, searching for patterns, and recursing
into sub-problems.

## Key Contributions
1. A recursive loop architecture where the LLM acts as both planner and executor
2. Code-based navigation tools: peek(), grep(), chunk(), select()
3. Budget controls: step limits, token caps, cost caps
4. Two-model cost optimization: expensive root model + cheap recursive model
5. Empirical evaluation showing 30-50% token reduction on long-document tasks

## Method
At each step the model receives:
- The user query
- Current context window budget
- Available navigation tools
- Results from previous steps

It then either emits a FINAL answer or generates Python code to gather
more information. This continues until the budget is exhausted or a final
answer is produced.

## Results
Evaluated on LongBench, QuALITY, and SCROLLS benchmarks.
RLM outperforms RAG on tasks requiring precise multi-hop reasoning while
using 30-50% fewer tokens than full-context approaches on documents > 100K tokens.
"""
    * 3
)  # Repeat to make it a realistic size


def run_comparison(query: str) -> None:
    print(f"Query: {query}")
    print("-" * 60)

    # Direct mode
    direct = interact(
        content=CONTENT,
        query=query,
        mode="direct",
        provider="mock",  # Change to "openai" / "anthropic" with a real key
    )
    print(
        f"Direct  | tokens={direct.total_tokens:>6,} | cost=${direct.total_cost:.4f} | {direct.elapsed_time:.2f}s"
    )

    # RLM mode
    rlm = interact(
        content=CONTENT,
        query=query,
        mode="rlm",
        provider="mock",  # Change to "openai" / "anthropic" with a real key
        verbose=True,
    )
    print(
        f"RLM     | tokens={rlm.total_tokens:>6,} | cost=${rlm.total_cost:.4f} | {rlm.elapsed_time:.2f}s | steps={rlm.steps}"
    )

    if rlm.total_tokens and direct.total_tokens:
        savings = 1 - rlm.total_tokens / direct.total_tokens
        print(f"Savings | {savings:+.1%} tokens vs Direct")

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("RLMKit Comparison Demo — RLM vs Direct")
    print("=" * 60)
    print()
    run_comparison("What are the key contributions of the RLM paper?")
    run_comparison("How does RLM compare to RAG on long-document tasks?")
    print("Done. Change provider='mock' to 'openai' or 'anthropic' for real results.")
