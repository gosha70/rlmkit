"""Pydantic v2 request/response models for the RLMKit API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    uptime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    content: Optional[str] = None
    file_id: Optional[str] = None
    query: str
    mode: Literal["auto", "rlm", "direct", "rag", "compare"] = "auto"
    provider: Optional[str] = None
    model: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    execution_id: str
    session_id: str
    status: str = "running"


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


class FileUploadResponse(BaseModel):
    id: str
    name: str
    size_bytes: int
    type: str
    token_count: int = 0
    created_at: datetime


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class MessageMetrics(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    steps: int = 0


class SessionMessage(BaseModel):
    id: str
    role: str
    content: str
    file_id: Optional[str] = None
    mode: Optional[str] = None
    mode_used: Optional[str] = None
    execution_id: Optional[str] = None
    metrics: Optional[MessageMetrics] = None
    timestamp: datetime


class SessionSummary(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class SessionDetail(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    messages: List[SessionMessage] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class ModeSummary(BaseModel):
    queries: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_seconds: float = 0.0


class ProviderSummary(BaseModel):
    queries: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0


class MetricsSummary(BaseModel):
    total_queries: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_seconds: float = 0.0
    avg_token_savings_percent: float = 0.0


class TimelineEntry(BaseModel):
    timestamp: datetime
    tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    mode: str = ""


class MetricsResponse(BaseModel):
    session_id: str
    summary: MetricsSummary = Field(default_factory=MetricsSummary)
    by_mode: Dict[str, ModeSummary] = Field(default_factory=dict)
    by_provider: Dict[str, ProviderSummary] = Field(default_factory=dict)
    timeline: List[TimelineEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


class TraceStep(BaseModel):
    index: int
    action_type: str = "inspect"
    code: Optional[str] = None
    output: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    recursion_depth: int = 0
    model: Optional[str] = None
    timestamp: Optional[datetime] = None


class TraceBudget(BaseModel):
    steps_used: int = 0
    steps_limit: int = 16
    tokens_used: int = 0
    tokens_limit: int = 50000
    cost_used: float = 0.0
    cost_limit: float = 2.0
    max_depth_reached: int = 0
    depth_limit: int = 5


class TraceResult(BaseModel):
    answer: str = ""
    success: bool = True


class TraceResponse(BaseModel):
    execution_id: str
    session_id: str
    query: str
    mode: str
    status: str = "complete"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: TraceResult = Field(default_factory=TraceResult)
    budget: TraceBudget = Field(default_factory=TraceBudget)
    steps: List[TraceStep] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Runtime Settings
# ---------------------------------------------------------------------------


class RuntimeSettings(BaseModel):
    temperature: float = 0.7
    top_p: float = 1.0
    max_output_tokens: int = 4096
    timeout_seconds: int = 30


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class ModelPricing(BaseModel):
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0


class ModelInfo(BaseModel):
    name: str
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0


class ProviderInfo(BaseModel):
    name: str
    display_name: str
    status: str = "not_configured"
    models: List[ModelInfo] = Field(default_factory=list)
    default_model: Optional[str] = None
    configured: bool = False
    requires_api_key: bool = True
    default_endpoint: Optional[str] = None
    model_input_hint: str = ""


class ProviderConfig(BaseModel):
    """Per-provider configuration: which provider/model to use with what settings."""
    provider: str
    model: str
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    enabled: bool = False


class ProviderTestRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    model: Optional[str] = None


class ProviderSaveRequest(BaseModel):
    api_key: Optional[str] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    runtime_settings: Optional[RuntimeSettings] = None
    enabled: Optional[bool] = None


class ProviderSaveResponse(BaseModel):
    saved: bool
    provider: str
    env_var: Optional[str] = None
    message: str = ""


class ProviderTestResponse(BaseModel):
    connected: bool
    latency_ms: Optional[int] = None
    model: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class BudgetConfig(BaseModel):
    max_steps: int = 16
    max_tokens: int = 50000
    max_cost_usd: float = 2.0
    max_time_seconds: int = 30
    max_recursion_depth: int = 5


class SandboxConfig(BaseModel):
    type: str = "restricted"
    docker_image: Optional[str] = None


class AppearanceConfig(BaseModel):
    theme: str = "system"
    sidebar_collapsed: bool = False


class RAGConfig(BaseModel):
    """RAG-specific execution settings."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    embedding_model: str = "text-embedding-3-small"


class ModeConfig(BaseModel):
    """Mode selection and per-mode settings."""
    enabled_modes: List[str] = Field(default_factory=lambda: ["direct", "rlm"])
    default_mode: str = "auto"
    rag_config: RAGConfig = Field(default_factory=RAGConfig)
    rlm_max_steps: int = 16
    rlm_timeout_seconds: int = 60


class ConfigResponse(BaseModel):
    active_provider: str = "openai"
    active_model: str = "gpt-4o"
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    appearance: AppearanceConfig = Field(default_factory=AppearanceConfig)
    provider_configs: List[ProviderConfig] = Field(default_factory=list)
    default_runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    mode_config: ModeConfig = Field(default_factory=ModeConfig)


class ConfigUpdateRequest(BaseModel):
    active_provider: Optional[str] = None
    active_model: Optional[str] = None
    budget: Optional[BudgetConfig] = None
    sandbox: Optional[SandboxConfig] = None
    appearance: Optional[AppearanceConfig] = None
    provider_configs: Optional[List[ProviderConfig]] = None
    default_runtime_settings: Optional[RuntimeSettings] = None
    mode_config: Optional[ModeConfig] = None


# ---------------------------------------------------------------------------
# Run Profiles
# ---------------------------------------------------------------------------


class RunProfile(BaseModel):
    """A named experiment configuration preset."""
    id: str
    name: str
    description: str = ""
    strategy: str = "direct"
    default_provider: Optional[str] = None
    providers_enabled: List[str] = Field(default_factory=list)
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    system_prompts: Dict[str, str] = Field(default_factory=dict)
    is_builtin: bool = False


class RunProfileCreate(BaseModel):
    """Request to create a new profile."""
    name: str
    description: str = ""
    strategy: str = "direct"
    default_provider: Optional[str] = None
    providers_enabled: List[str] = Field(default_factory=list)
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    system_prompts: Dict[str, str] = Field(default_factory=dict)


class RunProfileUpdate(BaseModel):
    """Request to update an existing profile (all fields optional)."""
    name: Optional[str] = None
    description: Optional[str] = None
    strategy: Optional[str] = None
    default_provider: Optional[str] = None
    providers_enabled: Optional[List[str]] = None
    runtime_settings: Optional[RuntimeSettings] = None
    budget: Optional[BudgetConfig] = None
    system_prompts: Optional[Dict[str, str]] = None


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------


class SystemPrompts(BaseModel):
    """Per-mode system prompts."""
    direct: str = ""
    rlm: str = ""
    rag: str = ""


class SystemPromptTemplate(BaseModel):
    """A named system prompt template."""
    name: str
    description: str = ""
    prompts: Dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# WebSocket events
# ---------------------------------------------------------------------------


class WSTokenEvent(BaseModel):
    type: str = "token"
    id: str
    data: str
    mode: str = ""


class WSStepEvent(BaseModel):
    type: str = "step"
    id: str
    data: TraceStep


class WSMetricsEvent(BaseModel):
    type: str = "metrics"
    id: str
    data: Dict[str, Any]


class WSCompleteEvent(BaseModel):
    type: str = "complete"
    id: str
    data: Dict[str, Any]


class WSErrorEvent(BaseModel):
    type: str = "error"
    id: str
    data: Dict[str, Any]
