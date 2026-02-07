"""
Unified high-level API for RLMKit.

This module provides a simple `interact()` function that serves as the main
entry point for all interaction modes (Direct, RAG, RLM).
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal

from .core.rlm import RLM, RLMConfig
from .core.budget import BudgetTracker, estimate_tokens
from .llm.base import BaseLLMProvider
from .llm import get_llm_client, auto_detect_provider, ConfigurationError
from .strategies import DirectStrategy, RAGStrategy, RLMStrategy, StrategyResult


# Type alias for interaction modes
InteractionMode = Literal["direct", "rag", "rlm", "auto"]


@dataclass
class InteractResult:
    """
    Result from an interact() call.
    
    Attributes:
        answer: The generated response text
        mode_used: Which strategy was actually used
        metrics: Token usage, cost, and timing information
        trace: Optional execution trace (for RLM mode)
        raw_result: The underlying strategy result for advanced access
    """
    answer: str
    mode_used: str
    metrics: Dict[str, Any]
    trace: Optional[Any] = None
    raw_result: Optional[StrategyResult] = None
    
    def __str__(self) -> str:
        return self.answer
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "answer": self.answer,
            "mode_used": self.mode_used,
            "metrics": self.metrics,
            "has_trace": self.trace is not None,
        }


def _determine_auto_mode(content: str) -> str:
    """
    Automatically determine the best mode based on content size.
    
    Rules:
    - < 8K tokens: Direct (fits in most context windows)
    - 8K-100K tokens: RAG (retrieval helps)
    - > 100K tokens: RLM (recursive exploration needed)
    
    Args:
        content: The content to analyze
        
    Returns:
        The recommended mode: "direct", "rag", or "rlm"
    """
    token_count = estimate_tokens(content)
    
    if token_count < 8000:
        return "direct"
    elif token_count < 100000:
        return "rag"
    else:
        return "rlm"


def interact(
    content: str,
    query: str,
    mode: InteractionMode = "auto",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    config: Optional[RLMConfig] = None,
    verbose: bool = False,
    **kwargs
) -> InteractResult:
    """
    Interact with content using an LLM through Direct, RAG, or RLM mode.
    
    This is the main entry point for RLMKit. It automatically handles:
    - LLM client configuration
    - Strategy selection (if mode="auto")
    - Result formatting
    
    Args:
        content: The document/text content to analyze
        query: The question or instruction for the LLM
        mode: Interaction mode - "direct", "rag", "rlm", or "auto"
            - "direct": Send full content + query to LLM in one call
            - "rag": Use retrieval-augmented generation (chunk + rank)
            - "rlm": Use recursive exploration with code generation
            - "auto": Automatically choose based on content size
        provider: LLM provider name ("openai", "anthropic", "ollama", etc.)
                  If None, will auto-detect from environment variables
        model: Model name (e.g., "gpt-4o", "claude-3-opus")
               If None, uses provider default
        api_key: API key for the provider (if not in environment)
        config: RLMConfig for advanced configuration
        verbose: If True, print execution progress (useful for RLM mode)
        **kwargs: Additional arguments passed to the strategy
        
    Returns:
        InteractResult with answer, mode used, and metrics
        
    Raises:
        ValueError: If mode is invalid or configuration is incomplete
        
    Examples:
        >>> # Simple usage with auto mode
        >>> result = interact("Long document...", "Summarize this")
        >>> print(result.answer)
        
        >>> # Explicit RLM mode with specific model
        >>> result = interact(
        ...     content="Large document...",
        ...     query="Find the key points",
        ...     mode="rlm",
        ...     provider="openai",
        ...     model="gpt-4o"
        ... )
        >>> print(f"Used {result.mode_used} mode")
        >>> print(f"Cost: ${result.metrics['total_cost']:.4f}")
        
        >>> # Direct mode for small content
        >>> result = interact(
        ...     content="Short text",
        ...     query="What is this?",
        ...     mode="direct"
        ... )
    """
    # Validate inputs
    if not content:
        raise ValueError("content cannot be empty")
    if not query:
        raise ValueError("query cannot be empty")
    
    # Determine actual mode if auto
    actual_mode = mode
    if mode == "auto":
        actual_mode = _determine_auto_mode(content)
        if verbose:
            print(f"[Auto Mode] Selected '{actual_mode}' based on content size "
                  f"({estimate_tokens(content):,} tokens)")
    
    # Validate mode
    if actual_mode not in ("direct", "rag", "rlm"):
        raise ValueError(f"Invalid mode: {actual_mode}. Must be 'direct', 'rag', 'rlm', or 'auto'")
    
    # Get LLM client - auto-detect if not specified
    if provider is None:
        provider = auto_detect_provider()
        if provider is None:
            raise ConfigurationError(
                "No LLM provider configured. Please set one of:\n"
                "  • OPENAI_API_KEY=sk-...\n"
                "  • ANTHROPIC_API_KEY=sk-ant-...\n"
                "  • OLLAMA_BASE_URL=http://localhost:11434\n\n"
                "Or explicitly pass provider='openai' and api_key='...' to interact()"
            )
        if verbose:
            print(f"[Auto-Detect] Using '{provider}' provider from environment")
    
    if verbose:
        print(f"[Setup] Configuring {provider} provider...")
    
    client: BaseLLMProvider = get_llm_client(
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs
    )
    
    # Use provided config or create default
    if config is None:
        config = RLMConfig()
    
    # Execute based on mode
    if verbose:
        print(f"[Execution] Running in '{actual_mode}' mode...")
    
    strategy_result: StrategyResult
    
    if actual_mode == "direct":
        strategy = DirectStrategy(client=client)
        strategy_result = strategy.run(content=content, query=query)
        
    elif actual_mode == "rag":
        # RAG requires embeddings - use OpenAI embeddings by default
        try:
            from .strategies import OpenAIEmbedder
            embedder = OpenAIEmbedder()
        except ImportError:
            raise ValueError(
                "RAG mode requires OpenAI embeddings. "
                "Install with: pip install openai"
            )
        
        strategy = RAGStrategy(
            client=client,
            embedder=embedder,
            top_k=kwargs.get("top_k", 5)
        )
        strategy_result = strategy.run(content=content, query=query)
        
    elif actual_mode == "rlm":
        strategy = RLMStrategy(client=client, config=config)
        strategy_result = strategy.run(content=content, query=query)
        
    else:
        raise ValueError(f"Unsupported mode: {actual_mode}")
    
    # Extract metrics from strategy result
    metrics = {
        "total_tokens": strategy_result.tokens.total_tokens,
        "input_tokens": strategy_result.tokens.input_tokens,
        "output_tokens": strategy_result.tokens.output_tokens,
        "total_cost": strategy_result.cost,
        "execution_time": strategy_result.elapsed_time,
        "llm_calls": strategy_result.steps,
    }
    
    if verbose:
        print(f"[Complete] Generated {len(strategy_result.answer)} character response")
        print(f"  Tokens: {metrics['total_tokens']:,} | Cost: ${metrics['total_cost']:.4f}")
    
    # Build result
    return InteractResult(
        answer=strategy_result.answer,
        mode_used=actual_mode,
        metrics=metrics,
        trace=None,  # TODO: Add tracing in Bet 3
        raw_result=strategy_result
    )


# Convenience function for backward compatibility
def complete(content: str, query: str, **kwargs) -> str:
    """
    Simple completion function that returns just the answer string.
    
    This is a convenience wrapper around interact() for quick usage.
    
    Args:
        content: The document/text content
        query: The question or instruction
        **kwargs: Arguments passed to interact()
        
    Returns:
        The answer string
        
    Example:
        >>> answer = complete("Document text", "Summarize this")
        >>> print(answer)
    """
    result = interact(content=content, query=query, **kwargs)
    return result.answer
