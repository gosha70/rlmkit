"""
Multi-Model Cost Optimization Example (Bet 4)

This demonstrates using different models for root reasoning vs. sub-agent exploration,
achieving 50-80% cost reduction while maintaining quality.

Key Concept:
- Root agent (expensive model): Main reasoning, final answer generation
- Sub-agents (cheap model): Information gathering, exploration subcalls
"""

from rlmkit.core import ModelConfig, ModelMetrics


def example_1_cost_optimized_openai():
    """Example 1: Cost-optimized configuration for OpenAI."""
    print("=" * 60)
    print("Example 1: Cost-Optimized OpenAI Configuration")
    print("=" * 60)
    
    # Use convenience method for cost optimization
    config = ModelConfig.cost_optimized("openai")
    
    print(f"\nConfiguration: {config}")
    print(f"  Root model: {config.root_model} (quality for final reasoning)")
    print(f"  Sub model: {config.sub_model} (cheap for exploration)")
    
    # Simulate realistic usage metrics
    metrics = ModelMetrics()
    
    # Root: 2 calls for main reasoning
    metrics.root_calls = 2
    metrics.root_tokens = 5000
    metrics.root_cost = 0.15  # GPT-4o pricing (~$0.03/1K tokens)
    
    # Subs: 8 exploration subcalls
    metrics.sub_calls = 8
    metrics.sub_tokens = 20000
    metrics.sub_cost = 0.02  # GPT-4o-mini pricing (~$0.001/1K tokens)
    
    print(f"\nUsage Metrics:")
    print(f"  Root: {metrics.root_calls} calls, {metrics.root_tokens:,} tokens, ${metrics.root_cost:.4f}")
    print(f"  Subs: {metrics.sub_calls} calls, {metrics.sub_tokens:,} tokens, ${metrics.sub_cost:.4f}")
    print(f"  Total: {metrics.total_calls} calls, {metrics.total_tokens:,} tokens, ${metrics.total_cost:.4f}")
    
    # Calculate savings vs single model
    gpt4_cost_per_token = 0.00003
    savings = metrics.savings_vs_single_model(gpt4_cost_per_token)
    savings_pct = (savings / (metrics.total_tokens * gpt4_cost_per_token)) * 100
    
    print(f"\nCost Analysis:")
    print(f"  If using GPT-4o only: ${metrics.total_tokens * gpt4_cost_per_token:.4f}")
    print(f"  With multi-model: ${metrics.total_cost:.4f}")
    print(f"  💰 Savings: ${savings:.4f} ({savings_pct:.1f}% reduction!)")
    print()


def example_2_cross_provider_mixed():
    """Example 2: Maximum savings with cross-provider configuration."""
    print("=" * 60)
    print("Example 2: Cross-Provider (Mixed) Configuration")
    print("=" * 60)
    
    # Use Claude for quality, local Llama for cost
    config = ModelConfig.cost_optimized("mixed")
    
    print(f"\nConfiguration: {config}")
    print(f"  Root: {config.root_provider}:{config.root_model} (cloud, quality)")
    print(f"  Sub: {config.sub_provider}:{config.sub_model} (local, FREE!)")
    
    # Simulate usage with local model for subs
    metrics = ModelMetrics()
    
    # Root: Claude Opus (cloud, expensive)
    metrics.root_calls = 2
    metrics.root_tokens = 5000
    metrics.root_cost = 0.15  # Claude pricing
    
    # Subs: Local Llama (FREE!)
    metrics.sub_calls = 10
    metrics.sub_tokens = 30000
    metrics.sub_cost = 0.00  # Local inference is free!
    
    print(f"\nUsage Metrics:")
    print(f"  Root: {metrics.root_calls} calls, {metrics.root_tokens:,} tokens, ${metrics.root_cost:.4f}")
    print(f"  Subs: {metrics.sub_calls} calls, {metrics.sub_tokens:,} tokens, ${metrics.sub_cost:.4f} (FREE!)")
    print(f"  Total: {metrics.total_calls} calls, {metrics.total_tokens:,} tokens, ${metrics.total_cost:.4f}")
    
    # Calculate max savings
    claude_cost_per_token = 0.00003
    savings = metrics.savings_vs_single_model(claude_cost_per_token)
    savings_pct = (savings / (metrics.total_tokens * claude_cost_per_token)) * 100
    
    print(f"\nCost Analysis:")
    print(f"  If using Claude only: ${metrics.total_tokens * claude_cost_per_token:.4f}")
    print(f"  With mixed providers: ${metrics.total_cost:.4f}")
    print(f"  🎉 Savings: ${savings:.4f} ({savings_pct:.1f}% reduction!)")
    print(f"  💡 Tip: Local models are perfect for exploration tasks!")
    print()


def example_3_custom_configuration():
    """Example 3: Custom multi-model configuration."""
    print("=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)
    
    # Create custom config
    config = ModelConfig(
        root_model="gpt-4o",
        root_provider="openai",
        sub_model="claude-3-5-haiku-20241022",
        sub_provider="anthropic"
    )
    
    print(f"\nCustom Configuration: {config}")
    print(f"  Mixing providers: OpenAI for root, Anthropic for subs")
    print(f"  Use case: Best of both ecosystems")
    print()


def example_4_single_model_fallback():
    """Example 4: Single model configuration (no optimization)."""
    print("=" * 60)
    print("Example 4: Single Model (Backward Compatible)")
    print("=" * 60)
    
    # For backward compatibility or when optimization isn't needed
    config = ModelConfig.from_single_model(
        model="gpt-4o",
        provider="openai"
    )
    
    print(f"\nSingle Model Config: {config}")
    print(f"  Same model for root and subs: {config.root_model}")
    print(f"  Use case: Simplicity over cost optimization")
    print()


def usage_in_practice():
    """Show how to use ModelConfig with interact()."""
    print("=" * 60)
    print("Usage with interact() API")
    print("=" * 60)
    
    print("""
# In practice, you would use it like this:

from rlmkit import interact
from rlmkit.core import ModelConfig

# Cost-optimized configuration
models = ModelConfig.cost_optimized("openai")

# Use with interact() for RLM mode
result = interact(
    content="Large document...",
    query="Analyze and summarize",
    mode="rlm",
    models=models,  # <- Pass multi-model config
    verbose=True
)

print(f"Answer: {result.answer}")
print(f"Cost: ${result.metrics['total_cost']:.4f}")

# The RLM strategy will automatically:
# - Use GPT-4o for root reasoning
# - Use GPT-4o-mini for all subcalls
# - Save 50-80% on costs!
""")
    print()


if __name__ == "__main__":
    print("\n🚀 RLMKit Multi-Model Cost Optimization Examples\n")
    
    example_1_cost_optimized_openai()
    example_2_cross_provider_mixed()
    example_3_custom_configuration()
    example_4_single_model_fallback()
    usage_in_practice()
    
    print("=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("✓ Multi-model reduces costs by 50-80%")
    print("✓ Quality maintained (expensive model for final reasoning)")
    print("✓ Flexible: same provider, cross-provider, or local models")
    print("✓ Simple API: ModelConfig.cost_optimized(\"openai\")")
    print("=" * 60)
