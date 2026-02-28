"""Pydantic v2 request/response models for the RLMKit API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


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
    content: str | None = None
    file_id: str | None = None
    query: str
    mode: Literal["auto", "rlm", "direct", "rag", "compare"] = "auto"
    provider: str | None = None
    model: str | None = None
    session_id: str | None = None
    chat_provider_id: str | None = None  # NEW: reference a Chat Provider


class ChatResponse(BaseModel):
    execution_id: str
    session_id: str
    status: str = "running"
    chat_provider_id: str | None = None  # NEW: echo back the Chat Provider used


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
    file_id: str | None = None
    mode: str | None = None
    mode_used: str | None = None
    execution_id: str | None = None
    metrics: MessageMetrics | None = None
    chat_provider_id: str | None = None  # NEW: which Chat Provider generated this
    chat_provider_name: str | None = None  # NEW: display name for convenience
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
    messages: list[SessionMessage] = Field(default_factory=list)
    conversations: dict[str, list[SessionMessage]] = Field(default_factory=dict)  # NEW: keyed by chat_provider_id


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
    avg_latency_seconds: float = 0.0


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
    provider: str = ""


class MetricsResponse(BaseModel):
    session_id: str
    summary: MetricsSummary = Field(default_factory=MetricsSummary)
    by_mode: dict[str, ModeSummary] = Field(default_factory=dict)
    by_provider: dict[str, ProviderSummary] = Field(default_factory=dict)
    timeline: list[TimelineEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


class TraceStep(BaseModel):
    index: int
    action_type: str = "inspect"
    code: str | None = None
    output: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    recursion_depth: int = 0
    model: str | None = None
    timestamp: datetime | None = None


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
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: TraceResult = Field(default_factory=TraceResult)
    budget: TraceBudget = Field(default_factory=TraceBudget)
    steps: list[TraceStep] = Field(default_factory=list)


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
    models: list[ModelInfo] = Field(default_factory=list)
    default_model: str | None = None
    configured: bool = False
    requires_api_key: bool = True
    default_endpoint: str | None = None
    model_input_hint: str = ""
    masked_api_key: str | None = None


class ProviderConfig(BaseModel):
    """Per-provider configuration: which provider/model to use with what settings."""

    provider: str
    model: str
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    enabled: bool = False
    last_tested_status: str | None = None


class ProviderTestRequest(BaseModel):
    provider: str
    api_key: str | None = None
    endpoint: str | None = None
    model: str | None = None


class ProviderSaveRequest(BaseModel):
    api_key: str | None = None
    model: str | None = None
    endpoint: str | None = None
    runtime_settings: RuntimeSettings | None = None
    enabled: bool | None = None


class ProviderSaveResponse(BaseModel):
    saved: bool
    provider: str
    env_var: str | None = None
    message: str = ""


class ProviderTestResponse(BaseModel):
    connected: bool
    latency_ms: int | None = None
    model: str | None = None
    error: str | None = None


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
    docker_image: str | None = None


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

    enabled_modes: list[str] = Field(default_factory=lambda: ["direct", "rlm"])
    default_mode: str = "auto"
    rag_config: RAGConfig = Field(default_factory=RAGConfig)
    rlm_max_steps: int = 16
    rlm_timeout_seconds: int = 60


# ---------------------------------------------------------------------------
# Chat Providers
# ---------------------------------------------------------------------------


class ChatProviderConfig(BaseModel):
    """A named combination of LLM Provider + Execution Mode + Settings."""

    id: str
    name: str  # e.g. "DIRECT-CLAUDE", "RLM-OPENAI"
    llm_provider: str  # e.g. "anthropic", "openai"
    llm_model: str  # e.g. "claude-sonnet-4-5"
    execution_mode: Literal["direct", "rlm", "rag"]
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    rag_config: RAGConfig | None = None  # only for RAG mode
    rlm_max_steps: int = 16  # only for RLM mode
    rlm_timeout_seconds: int = 60  # only for RLM mode
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ChatProviderCreateRequest(BaseModel):
    name: str
    llm_provider: str
    llm_model: str
    execution_mode: Literal["direct", "rlm", "rag"]
    runtime_settings: RuntimeSettings | None = None
    rag_config: RAGConfig | None = None
    rlm_max_steps: int | None = None
    rlm_timeout_seconds: int | None = None


class ChatProviderUpdateRequest(BaseModel):
    name: str | None = None
    llm_model: str | None = None
    execution_mode: Literal["direct", "rlm", "rag"] | None = None
    runtime_settings: RuntimeSettings | None = None
    rag_config: RAGConfig | None = None
    rlm_max_steps: int | None = None
    rlm_timeout_seconds: int | None = None


class ConfigResponse(BaseModel):
    active_provider: str = "openai"
    active_model: str = "gpt-4o"
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    appearance: AppearanceConfig = Field(default_factory=AppearanceConfig)
    provider_configs: list[ProviderConfig] = Field(default_factory=list)
    default_runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    mode_config: ModeConfig = Field(default_factory=ModeConfig)
    chat_providers: list[ChatProviderConfig] = Field(default_factory=list)
    trajectory_dir: str | None = None


class ConfigUpdateRequest(BaseModel):
    active_provider: str | None = None
    active_model: str | None = None
    budget: BudgetConfig | None = None
    sandbox: SandboxConfig | None = None
    appearance: AppearanceConfig | None = None
    provider_configs: list[ProviderConfig] | None = None
    default_runtime_settings: RuntimeSettings | None = None
    mode_config: ModeConfig | None = None
    chat_providers: list[ChatProviderConfig] | None = None


# ---------------------------------------------------------------------------
# Run Profiles
# ---------------------------------------------------------------------------


class RunProfile(BaseModel):
    """A named experiment configuration preset."""

    id: str
    name: str
    description: str = ""
    strategy: str = "direct"
    default_provider: str | None = None
    providers_enabled: list[str] = Field(default_factory=list)
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    system_prompts: dict[str, str] = Field(default_factory=dict)
    is_builtin: bool = False


class RunProfileCreate(BaseModel):
    """Request to create a new profile."""

    name: str
    description: str = ""
    strategy: str = "direct"
    default_provider: str | None = None
    providers_enabled: list[str] = Field(default_factory=list)
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    system_prompts: dict[str, str] = Field(default_factory=dict)


class RunProfileUpdate(BaseModel):
    """Request to update an existing profile (all fields optional)."""

    name: str | None = None
    description: str | None = None
    strategy: str | None = None
    default_provider: str | None = None
    providers_enabled: list[str] | None = None
    runtime_settings: RuntimeSettings | None = None
    budget: BudgetConfig | None = None
    system_prompts: dict[str, str] | None = None


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
    prompts: dict[str, str] = Field(default_factory=dict)


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
    data: dict[str, Any]


class WSCompleteEvent(BaseModel):
    type: str = "complete"
    id: str
    data: dict[str, Any]


class WSErrorEvent(BaseModel):
    type: str = "error"
    id: str
    data: dict[str, Any]
