# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""
Demonstration of RLM vs Direct mode comparison.

This example shows how to:
1. Run RLM mode and Direct mode on the same content
2. Compare performance metrics (tokens, time, cost)
3. Visualize the differences

Usage:
    python examples/comparison_demo.py
"""

from rlmkit import RLM, RLMConfig
from rlmkit.llm import MockLLMClient
from rlmkit.config import ExecutionConfig
import json


def main():
    """Run comparison demo."""
    
    print("=" * 70)
    print("RLMKit Comparison Demo")
    print("=" * 70)
    print()
    
    # Sample content - simulating a large document
    content = """
    This is a sample document about artificial intelligence and machine learning.
    
    Machine learning is a subset of artificial intelligence that focuses on the 
    development of algorithms that can learn from and make predictions on data.
    
    Key concepts in machine learning include:
    - Supervised learning: Learning from labeled data
    - Unsupervised learning: Finding patterns in unlabeled data
    - Reinforcement learning: Learning through trial and error
    - Deep learning: Using neural networks with multiple layers
    
    Applications of machine learning include:
    - Image recognition
    - Natural language processing
    - Recommendation systems
    - Autonomous vehicles
    - Medical diagnosis
    
    The field has seen tremendous growth in recent years due to:
    - Increased computing power
    - Availability of large datasets
    - Advances in algorithms
    - Cloud computing infrastructure
    """ * 10  # Repeat to make it larger
    
    query = "What are the main types of machine learning mentioned in this document?"
    
    print(f"Content size: {len(content)} characters")
    print(f"Query: {query}")
    print()
    
    # Create mock LLM client for demo
    # In real usage, you would use OpenAI, Anthropic, etc.
    mock_responses = [
        "```python\n# Explore the content\nprint(len(P))\n```",
        "```python\n# Search for machine learning types\nimport re\nmatches = re.findall(r'(\\w+ learning)', P)\nprint(set(matches))\n```",
        "FINAL: The document mentions four main types of machine learning: Supervised learning, Unsupervised learning, Reinforcement learning, and Deep learning."
    ]
    
    client = MockLLMClient(mock_responses)
    
    # Create RLM configuration
    config = RLMConfig(
        execution=ExecutionConfig(
            max_steps=16,
            default_timeout=5.0,
            enable_rlm=True,
        )
    )
    
    # Create RLM instance
    rlm = RLM(client=client, config=config)
    
    # Run comparison
    print("Running comparison (RLM vs Direct mode)...")
    print("-" * 70)
    
    result = rlm.run_comparison(
        prompt=content,
        query=query
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # RLM Mode Answer
    if result.rlm_metrics:
        print("\n🤖 RLM Mode")
        print("-" * 70)
        print(f"Answer: {result.rlm_metrics.answer}")
        print(f"Steps: {result.rlm_metrics.steps}")
        print(f"Time: {result.rlm_metrics.elapsed_time:.2f}s")
        print(f"Tokens: {result.rlm_metrics.tokens.total_tokens}")
        print(f"  - Input: {result.rlm_metrics.tokens.input_tokens}")
        print(f"  - Output: {result.rlm_metrics.tokens.output_tokens}")
    
    # Direct Mode Answer
    if result.direct_metrics:
        print("\n📝 Direct Mode")
        print("-" * 70)
        print(f"Answer: {result.direct_metrics.answer}")
        print(f"Time: {result.direct_metrics.elapsed_time:.2f}s")
        print(f"Tokens: {result.direct_metrics.tokens.total_tokens}")
        print(f"  - Input: {result.direct_metrics.tokens.input_tokens}")
        print(f"  - Output: {result.direct_metrics.tokens.output_tokens}")
    
    # Comparison Summary
    summary = result.get_summary()
    
    if summary.get('can_compare'):
        print("\n📊 Comparison Summary")
        print("-" * 70)
        
        token_savings = summary.get('token_savings')
        if token_savings:
            print(f"Token Savings: {token_savings['savings_tokens']:,} tokens ({token_savings['savings_percent']:.1f}%)")
            if token_savings['rlm_is_better']:
                print("  ✓ RLM uses fewer tokens")
            else:
                print("  ✗ Direct mode uses fewer tokens")
        
        time_comp = summary.get('time_comparison')
        if time_comp:
            print(f"Time Difference: {abs(time_comp['difference']):.2f}s")
            if time_comp['direct_is_faster']:
                print("  Direct mode is faster")
            else:
                print("  RLM mode is faster")
        
        # Recommendation
        if summary.get('recommendation'):
            rec = summary['recommendation']
            reason = summary.get('recommendation_reason', '')
            print(f"\nRecommendation: {rec.upper()}")
            print(f"Reason: {reason}")
    
    # Export results
    print("\n" + "=" * 70)
    print("EXPORT")
    print("=" * 70)
    
    # Save to JSON
    output_file = "comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("\nTo run the interactive UI:")
    print("  pip install -e \".[ui]\"")
    print("  streamlit run src/rlmkit/ui/app.py")
    print()


if __name__ == "__main__":
    main()
