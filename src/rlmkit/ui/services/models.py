# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.
"""
Data models for RLMKit UI - Chat messages, metrics, responses.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from uuid import uuid4


@dataclass
class Response:
    """LLM Response from a single model."""
    
    content: str
    """The actual response text."""
    
    stop_reason: str
    """Why the model stopped: 'stop', 'length', 'error'."""
    
    raw_response: Optional[str] = None
    """Raw response object for debugging."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ExecutionMetrics:
    """Metrics for single execution (RLM or Direct LLM)."""
    
    # Tokens
    input_tokens: int
    """Number of input tokens consumed."""
    
    output_tokens: int
    """Number of output tokens generated."""
    
    total_tokens: int
    """Total tokens (input + output)."""
    
    # Cost
    cost_usd: float
    """Total cost in USD for this execution."""
    
    cost_breakdown: Dict[str, float]
    """Cost breakdown: {'input': $x, 'output': $y}."""
    
    # Performance
    execution_time_seconds: float
    """Total execution time in seconds."""
    
    steps_taken: int
    """Number of steps (0 for direct, >0 for RLM)."""
    
    # Memory (RLM specific)
    memory_used_mb: float
    """Current memory usage in MB."""
    
    memory_peak_mb: float
    """Peak memory usage in MB."""
    
    # Status
    success: bool
    """Whether execution was successful."""
    
    error: Optional[str] = None
    """Error message if execution failed."""
    
    execution_type: Literal["rlm", "direct"] = "rlm"
    """Type of execution."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ComparisonMetrics:
    """Comparison metrics between RLM and Direct execution."""
    
    token_savings_percent: float
    """Percentage of tokens saved by RLM."""
    
    token_savings_absolute: int
    """Absolute number of tokens saved."""
    
    cost_savings_percent: float
    """Percentage of cost saved by RLM."""
    
    cost_savings_absolute: float
    """Absolute cost savings in USD."""
    
    time_difference_seconds: float
    """Time difference (positive = Direct faster)."""
    
    recommendation: str
    """Recommendation: 'Use RLM', 'Use Direct', 'Trade-off'."""
    
    reasoning: Optional[str] = None
    """Explanation of the recommendation."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ChatMessage:
    """Single chat message with optional dual responses."""
    
    # Identifiers
    id: str = field(default_factory=lambda: str(uuid4()))
    """Unique message ID."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    """When message was created."""
    
    # User input
    user_query: str = ""
    """The user's question/query."""
    
    file_context: Optional[str] = None
    """Optional document/file content for context."""
    
    file_info: Optional[Dict[str, Any]] = None
    """File metadata: {'name': str, 'size_mb': float, 'tokens': int}."""
    
    # Execution mode
    mode: Literal["rlm_only", "direct_only", "compare"] = "compare"
    """Which execution path to use."""
    
    # RLM Response (if applicable)
    rlm_response: Optional[Response] = None
    """Response from RLM execution."""
    
    rlm_metrics: Optional[ExecutionMetrics] = None
    """Metrics from RLM execution."""
    
    rlm_trace: Optional[List[Dict[str, Any]]] = None
    """Step-by-step trace of RLM execution."""
    
    # Direct Response (if applicable)
    direct_response: Optional[Response] = None
    """Response from direct LLM execution."""
    
    direct_metrics: Optional[ExecutionMetrics] = None
    """Metrics from direct LLM execution."""
    
    # Comparison
    comparison_metrics: Optional[ComparisonMetrics] = None
    """Metrics comparing RLM and Direct (if both were run)."""
    
    # Status
    in_progress: bool = False
    """Whether this message is still being processed."""
    
    error: Optional[str] = None
    """Error message if something went wrong."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def has_rlm_response(self) -> bool:
        """Check if RLM response exists."""
        return self.rlm_response is not None and self.rlm_response.content
    
    def has_direct_response(self) -> bool:
        """Check if Direct response exists."""
        return self.direct_response is not None and self.direct_response.content
    
    def is_complete(self) -> bool:
        """Check if all expected responses are complete."""
        if self.mode == "rlm_only":
            return self.has_rlm_response() and not self.in_progress
        elif self.mode == "direct_only":
            return self.has_direct_response() and not self.in_progress
        elif self.mode == "compare":
            return self.has_rlm_response() and self.has_direct_response() and not self.in_progress
        return False


@dataclass
class LLMProviderConfig:
    """Configuration for a single LLM provider."""
    
    # Provider info
    provider: str
    """Provider name: 'openai', 'anthropic', 'ollama', 'lmstudio'."""
    
    enabled: bool = True
    """Whether this provider is available."""
    
    # Connection details
    model: str = ""
    """Model name/ID."""
    
    api_endpoint: Optional[str] = None
    """Custom API endpoint (for local models)."""
    
    timeout_seconds: int = 30
    """Request timeout in seconds."""
    
    # Parameters
    temperature: float = 0.7
    """Sampling temperature (0.0-2.0)."""
    
    max_tokens: int = 2000
    """Maximum tokens in response."""
    
    top_p: float = 1.0
    """Nucleus sampling parameter."""
    
    # Authentication (never logged or displayed)
    api_key: Optional[str] = None
    """API key (temporary in memory, never saved to disk)."""
    
    api_key_env_var: Optional[str] = None
    """Environment variable name for API key."""
    
    # Pricing (for cost calculation)
    input_cost_per_1k_tokens: float = 0.0
    """Cost per 1K input tokens in USD."""
    
    output_cost_per_1k_tokens: float = 0.0
    """Cost per 1K output tokens in USD."""
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    """When this config was created."""
    
    last_used_at: Optional[datetime] = None
    """When this provider was last used."""
    
    test_successful: bool = False
    """Whether last connection test succeeded."""
    
    def to_dict(self, include_api_key: bool = False) -> Dict[str, Any]:
        """Convert to dictionary (without API key by default)."""
        data = asdict(self)
        if not include_api_key:
            data["api_key"] = None
        return data
    
    @property
    def is_configured(self) -> bool:
        """Check if provider has minimal configuration."""
        return self.model and (self.api_key or self.api_key_env_var)
    
    @property
    def is_ready(self) -> bool:
        """Check if provider is ready to use."""
        return self.is_configured and self.test_successful


@dataclass
class SessionMetrics:
    """Aggregated metrics for entire session."""
    
    total_messages: int = 0
    """Total number of messages in session."""
    
    total_tokens: int = 0
    """Total tokens used in session."""
    
    total_cost: float = 0.0
    """Total cost in USD for session."""
    
    total_duration_seconds: float = 0.0
    """Total time spent in session."""
    
    peak_memory_mb: float = 0.0
    """Peak memory usage in MB."""
    
    rlm_savings_tokens: int = 0
    """Total tokens saved by using RLM."""
    
    rlm_savings_cost: float = 0.0
    """Total cost saved by using RLM."""
    
    requests_by_mode: Dict[str, int] = field(default_factory=lambda: {
        "rlm_only": 0,
        "direct_only": 0,
        "compare": 0,
    })
    """Breakdown of requests by mode."""
    
    avg_response_time_seconds: float = 0.0
    """Average response time per request."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
