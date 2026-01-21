"""MetricsCollector - Collect and aggregate execution metrics."""

from typing import Optional, Dict, Any, List
from .models import ExecutionMetrics, ComparisonMetrics


class MetricsCollector:
    """
    Collect and manage execution metrics.
    
    Responsibilities:
    - Count tokens from execution traces
    - Calculate costs based on provider pricing
    - Monitor memory usage
    - Compare metrics between execution paths
    - Aggregate session-wide metrics
    """
    
    def __init__(self):
        """Initialize MetricsCollector."""
        self.provider_pricing: Dict[str, Dict[str, float]] = {
            "openai": {
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4-turbo": {"input": 0.01, "output": 0.03},
                "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            },
            "anthropic": {
                "claude-3-opus": {"input": 0.015, "output": 0.075},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            },
        }
    
    async def collect_rlm_metrics(
        self,
        rlm_result: Dict[str, Any],
        rlm_trace: List[Dict[str, Any]],
        memory_monitor: "MemoryMonitor",
        provider: str = "openai",
        model: str = "gpt-4",
    ) -> ExecutionMetrics:
        """
        Collect metrics from RLM execution.
        
        Args:
            rlm_result: Result dict from RLM execution
            rlm_trace: Step-by-step execution trace
            memory_monitor: MemoryMonitor instance
            provider: LLM provider name
            model: Model name
        
        Returns:
            ExecutionMetrics for RLM execution
            
        Implementation notes:
        - Should sum tokens across all steps in trace
        - Should calculate cost using provider pricing
        - Should get memory peak from MemoryMonitor
        - Should handle missing data gracefully
        """
        raise NotImplementedError("To be implemented")
    
    async def collect_direct_metrics(
        self,
        direct_result: Dict[str, Any],
        memory_monitor: "MemoryMonitor",
        provider: str = "openai",
        model: str = "gpt-4",
    ) -> ExecutionMetrics:
        """
        Collect metrics from direct LLM execution.
        
        Args:
            direct_result: Result dict from direct LLM call
            memory_monitor: MemoryMonitor instance
            provider: LLM provider name
            model: Model name
        
        Returns:
            ExecutionMetrics for direct execution
            
        Implementation notes:
        - Should extract token counts from LLM response
        - Should calculate cost using provider pricing
        - Should get memory usage from MemoryMonitor
        """
        raise NotImplementedError("To be implemented")
    
    def compare_metrics(
        self,
        rlm_metrics: ExecutionMetrics,
        direct_metrics: ExecutionMetrics,
    ) -> ComparisonMetrics:
        """
        Compare metrics between RLM and Direct execution.
        
        Args:
            rlm_metrics: Metrics from RLM
            direct_metrics: Metrics from Direct
        
        Returns:
            ComparisonMetrics with savings and recommendation
            
        Implementation notes:
        - Should calculate token savings (absolute and %)
        - Should calculate cost savings (absolute and %)
        - Should note time difference
        - Should generate recommendation based on trade-offs
        """
        raise NotImplementedError("To be implemented")
    
    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        model: str,
    ) -> float:
        """
        Calculate cost for token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider: Provider name (openai, anthropic, etc)
            model: Model name
        
        Returns:
            Cost in USD
            
        Implementation notes:
        - Should look up pricing from self.provider_pricing
        - Should handle unknown providers/models gracefully
        - Should return 0.0 for unsupported combinations
        """
        raise NotImplementedError("To be implemented")
    
    def _cost_breakdown(
        self,
        input_tokens: int,
        output_tokens: int,
        provider: str,
        model: str,
    ) -> Dict[str, float]:
        """
        Get cost breakdown (input vs output).
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider: Provider name
            model: Model name
        
        Returns:
            Dict with {'input': $x, 'output': $y}
            
        Implementation notes:
        - Should calculate separately for input/output
        - Should sum to total cost
        """
        raise NotImplementedError("To be implemented")
    
    def add_provider_pricing(
        self,
        provider: str,
        model: str,
        input_cost_per_1k: float,
        output_cost_per_1k: float,
    ) -> None:
        """
        Add or update pricing for a provider/model combination.
        
        Args:
            provider: Provider name
            model: Model name
            input_cost_per_1k: Input cost per 1K tokens
            output_cost_per_1k: Output cost per 1K tokens
        """
        if provider not in self.provider_pricing:
            self.provider_pricing[provider] = {}
        
        self.provider_pricing[provider][model] = {
            "input": input_cost_per_1k,
            "output": output_cost_per_1k,
        }
