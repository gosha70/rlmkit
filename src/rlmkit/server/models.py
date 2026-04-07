"""Pydantic v2 request/response models for the RLMKit API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

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
    version: str = "1.0.0"
    uptime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    content: str | None = None
    file_id: str | None = None  # deprecated: use file_ids
    file_ids: list[str] | None = None
    query: str
    mode: Literal["auto", "rlm", "direct", "rag", "compare"] = "auto"
    provider: str | None = None
    model: str | None = None
    session_id: str | None = None
    chat_provider_id: str | None = None  # NEW: reference a Chat Provider
    num_retries: int | None = Field(default=None, ge=0)  # override retry count (0 = no retries)

    @model_validator(mode="before")
    @classmethod
    def _coerce_file_ids(cls, data: Any) -> Any:
        """Promote legacy file_id to file_ids list for backward compat."""
        if isinstance(data, dict):
            if not data.get("file_ids") and data.get("file_id"):
                data = dict(data, file_ids=[data["file_id"]])
        return data


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
    size_bytes: int  # raw upload size (bytes)
    text_size_bytes: int  # extracted-text size; matches what the per-message 50 MB guard measures
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
    file_id: str | None = None  # deprecated: use file_ids
    file_ids: list[str] | None = None
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


class SessionRenameRequest(BaseModel):
    name: str


class SessionDetail(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    messages: list[SessionMessage] = Field(default_factory=list)
    conversations: dict[str, list[SessionMessage]] = Field(
        default_factory=dict
    )  # NEW: keyed by chat_provider_id


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
    avg_token_savings_percent: float | None = None


class TimelineEntry(BaseModel):
    timestamp: datetime
    tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    mode: str = ""
    provider: str = ""
    chat_provider_name: str | None = None
    execution_id: str | None = None


class MetricsResponse(BaseModel):
    session_id: str
    summary: MetricsSummary = Field(default_factory=MetricsSummary)
    by_mode: dict[str, ModeSummary] = Field(default_factory=dict)
    by_provider: dict[str, ProviderSummary] = Field(default_factory=dict)
    by_chat_provider: dict[str, ProviderSummary] = Field(default_factory=dict)
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
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0


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
    chat_provider_id: str | None = None
    chat_provider_name: str | None = None


# ---------------------------------------------------------------------------
# Runtime Settings
# ---------------------------------------------------------------------------


class RuntimeSettings(BaseModel):
    temperature: float = 0.7
    top_p: float = 1.0
    max_output_tokens: int = 4096
    timeout_seconds: int = 120


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
    endpoint: str | None = None
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
# Named LLM Provider instances
# ---------------------------------------------------------------------------


class LLMProviderConfig(BaseModel):
    """A named, user-created instance of an LLM backend with full connection config."""

    id: str
    name: str  # User-defined, e.g. "Ollama-llama3.2", "GPT4-mini"
    backend: str  # Catalog key: "openai", "anthropic", "ollama", "lmstudio"
    model: str  # e.g. "gpt-4o-mini", "llama3.2"
    endpoint: str | None = None  # Custom endpoint (overrides catalog default for local providers)
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    status: str = "not_configured"  # "connected" | "configured" | "offline" | "not_configured"
    created_at: datetime | None = None
    updated_at: datetime | None = None


class LLMProviderCreateRequest(BaseModel):
    name: str
    backend: str  # catalog key: "openai", "anthropic", "ollama", "lmstudio"
    model: str
    api_key: str | None = None  # persisted to SecretStore, not stored in config
    endpoint: str | None = None
    runtime_settings: RuntimeSettings | None = None


class LLMProviderUpdateRequest(BaseModel):
    name: str | None = None
    model: str | None = None
    api_key: str | None = None
    endpoint: str | None = None
    runtime_settings: RuntimeSettings | None = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class BudgetConfig(BaseModel):
    max_steps: int = 16
    max_tokens: int = 50000
    max_cost_usd: float = 2.0
    max_time_seconds: int = 30
    max_recursion_depth: int = 5
    repeat_limit: int = 2  # RLM: duplicate inspections before forced finalization
    nudge_at_fraction: float = 0.6  # RLM: fraction of max_steps for soft nudge


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
    """A named combination of LLM Provider + Execution Mode + Profile."""

    id: str
    name: str  # e.g. "DIRECT-CLAUDE", "RLM-OPENAI"
    llm_provider_id: str = ""  # UUID reference to LLMProviderConfig; empty = not yet migrated
    llm_provider_name: str | None = None  # resolved display name (read-only, populated on GET)
    # Deprecated legacy fields kept for backward-compat migration; cleared after migration
    llm_provider: str = ""  # deprecated: backend key like "anthropic"
    llm_model: str = ""  # deprecated: model name like "claude-sonnet-4-6"
    profile_id: str | None = None  # reference to a RunProfile
    profile_name: str | None = None  # resolved profile name (read-only)
    execution_mode: Literal["direct", "rlm", "rag"] = "direct"
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    rag_config: RAGConfig | None = None  # only for RAG mode
    rlm_max_steps: int = 16  # only for RLM mode
    rlm_timeout_seconds: int = 60  # only for RLM mode
    rlm_repeat_limit: int = 2  # only for RLM mode — duplicate inspections before forced finalization
    rlm_nudge_at_fraction: float = 0.6  # only for RLM mode — fraction of max_steps for soft nudge
    num_retries: int | None = Field(default=None, ge=0)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ChatProviderCreateRequest(BaseModel):
    name: str
    llm_provider_id: str  # UUID of the LLMProviderConfig to use
    profile_id: str | None = None
    execution_mode: Literal["direct", "rlm", "rag"] = "direct"
    rag_config: RAGConfig | None = None
    rlm_max_steps: int | None = None
    rlm_timeout_seconds: int | None = None
    rlm_repeat_limit: int | None = None
    rlm_nudge_at_fraction: float | None = None
    num_retries: int | None = Field(default=None, ge=0)


class ChatProviderUpdateRequest(BaseModel):
    name: str | None = None
    llm_provider_id: str | None = None  # switch to a different LLM Provider
    profile_id: str | None = None
    execution_mode: Literal["direct", "rlm", "rag"] | None = None
    rag_config: RAGConfig | None = None
    rlm_max_steps: int | None = None
    rlm_timeout_seconds: int | None = None
    rlm_repeat_limit: int | None = None
    rlm_nudge_at_fraction: float | None = None
    num_retries: int | None = Field(default=None, ge=0)


class ConfigResponse(BaseModel):
    active_provider: str = "openai"
    active_model: str = "gpt-4o"
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    appearance: AppearanceConfig = Field(default_factory=AppearanceConfig)
    provider_configs: list[ProviderConfig] = Field(default_factory=list)
    llm_providers: list[LLMProviderConfig] = Field(default_factory=list)
    default_runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    mode_config: ModeConfig = Field(default_factory=ModeConfig)
    chat_providers: list[ChatProviderConfig] = Field(default_factory=list)
    active_profile_id: str | None = None
    trajectory_dir: str | None = None
    judge_chat_provider_id: str | None = None


class ConfigUpdateRequest(BaseModel):
    # Note: active_provider/active_model are set exclusively via
    # PUT /api/providers/{name} to prevent dual-save-path corruption.
    budget: BudgetConfig | None = None
    sandbox: SandboxConfig | None = None
    appearance: AppearanceConfig | None = None
    provider_configs: list[ProviderConfig] | None = None
    llm_providers: list[LLMProviderConfig] | None = None
    default_runtime_settings: RuntimeSettings | None = None
    mode_config: ModeConfig | None = None
    chat_providers: list[ChatProviderConfig] | None = None
    judge_chat_provider_id: str | None = None


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


# ---------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------


class ThumbRatingRequest(BaseModel):
    execution_id: str
    session_id: str
    chat_provider_id: str
    rating: Literal["up", "down"]


class BestPickRequest(BaseModel):
    session_id: str
    winner_execution_id: str


class JudgeRequest(BaseModel):
    session_id: str
    execution_ids: list[str]
    mode: Literal["pointwise", "pairwise", "both"] = "both"


class JudgeScore(BaseModel):
    id: str
    execution_id: str
    session_id: str
    chat_provider_id: str
    judge_provider_id: str
    dimensions: dict[str, float]
    overall_score: float
    reasoning: str
    created_at: datetime


class JudgePairwise(BaseModel):
    id: str
    session_id: str
    execution_id_a: str
    execution_id_b: str
    winner: Literal["a", "b", "tie"]
    judge_provider_id: str
    reasoning: str
    created_at: datetime


class ProviderQualityScore(BaseModel):
    chat_provider_id: str
    chat_provider_name: str
    thumb_up: int = 0
    thumb_down: int = 0
    thumb_score: float = 0.0
    best_picks: int = 0
    pick_rate: float = 0.0
    judge_avg_score: float | None = None
    combined_score: float = 0.0


class EvaluationSummaryResponse(BaseModel):
    session_id: str
    by_chat_provider: dict[str, ProviderQualityScore]
    recommendation: str | None = None
    recommendation_reason: str = ""


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
