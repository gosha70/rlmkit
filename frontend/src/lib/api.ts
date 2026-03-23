/**
 * Typed API client for the RLMKit backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

// ---------------------------------------------------------------------------
// Types matching the backend Pydantic models
// ---------------------------------------------------------------------------

export type ChatMode = "auto" | "rlm" | "direct" | "rag" | "compare";

export interface ChatRequest {
  content?: string | null;
  file_id?: string | null;
  query: string;
  mode?: ChatMode;
  provider?: string | null;
  model?: string | null;
  session_id?: string | null;
  chat_provider_id?: string | null;
}

export interface ChatResponse {
  execution_id: string;
  session_id: string;
  status: string;
  chat_provider_id?: string | null;
}

export interface FileUploadResponse {
  id: string;
  name: string;
  size_bytes: number;
  type: string;
  token_count: number;
  created_at: string;
}

export interface MessageMetrics {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number;
  elapsed_seconds: number;
  steps: number;
}

export interface SessionMessage {
  id: string;
  role: string;
  content: string;
  file_id?: string | null;
  mode?: ChatMode | null;
  mode_used?: ChatMode | null;
  execution_id?: string | null;
  metrics?: MessageMetrics | null;
  chat_provider_id?: string | null;
  chat_provider_name?: string | null;
  timestamp: string;
}

export interface SessionSummary {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface SessionDetail {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  messages: SessionMessage[];
  conversations?: Record<string, SessionMessage[]>;
}

export interface MetricsSummary {
  total_queries: number;
  total_tokens: number;
  total_cost_usd: number;
  avg_latency_seconds: number;
  avg_token_savings_percent: number;
}

export interface ModeSummary {
  queries: number;
  total_tokens: number;
  total_cost_usd: number;
  avg_latency_seconds: number;
}

export interface TimelineEntry {
  timestamp: string;
  tokens: number;
  cost_usd: number;
  latency_seconds: number;
  mode: ChatMode;
  provider: string;
  chat_provider_name?: string | null;
  execution_id?: string | null;
}

export interface MetricsResponse {
  session_id: string;
  summary: MetricsSummary;
  by_mode: Partial<Record<ChatMode, ModeSummary>>;
  by_provider: Record<string, { queries: number; total_tokens: number; total_cost_usd: number; avg_latency_seconds: number }>;
  by_chat_provider?: Record<string, { queries: number; total_tokens: number; total_cost_usd: number; avg_latency_seconds: number }>;
  timeline: TimelineEntry[];
}

export interface TraceStep {
  index: number;
  action_type: string;
  code?: string | null;
  output?: string | null;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  duration_seconds: number;
  recursion_depth: number;
  model?: string | null;
  timestamp?: string | null;
}

export interface TraceResponse {
  execution_id: string;
  session_id: string;
  query: string;
  mode: ChatMode;
  status: string;
  started_at?: string | null;
  completed_at?: string | null;
  result: { answer: string; success: boolean; input_tokens?: number; output_tokens?: number; total_cost?: number };
  budget: {
    steps_used: number;
    steps_limit: number;
    tokens_used: number;
    tokens_limit: number;
    cost_used: number;
    cost_limit: number;
    max_depth_reached: number;
    depth_limit: number;
  };
  steps: TraceStep[];
  chat_provider_id?: string | null;
  chat_provider_name?: string | null;
}

export interface ModelInfo {
  name: string;
  input_cost_per_1k: number;
  output_cost_per_1k: number;
}

export interface RuntimeSettings {
  temperature: number;
  top_p: number;
  max_output_tokens: number;
  timeout_seconds: number;
}

export interface ProviderConfigEntry {
  provider: string;
  model: string;
  endpoint: string | null;
  runtime_settings: RuntimeSettings;
  enabled: boolean;
}

export interface ProviderInfo {
  name: string;
  display_name: string;
  status: string;
  models: ModelInfo[];
  default_model: string | null;
  configured: boolean;
  requires_api_key: boolean;
  default_endpoint: string | null;
  model_input_hint: string;
  masked_api_key: string | null;
}

export interface ProviderTestRequest {
  provider: string;
  api_key?: string | null;
  endpoint?: string | null;
  model?: string | null;
}

export interface ProviderTestResponse {
  connected: boolean;
  latency_ms?: number | null;
  model?: string | null;
  error?: string | null;
}

export interface BudgetConfig {
  max_steps: number;
  max_tokens: number;
  max_cost_usd: number;
  max_time_seconds: number;
  max_recursion_depth: number;
}

export interface RAGConfig {
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  embedding_model: string;
}

export interface ModeConfig {
  enabled_modes: string[];
  default_mode: string;
  rag_config: RAGConfig;
  rlm_max_steps: number;
  rlm_timeout_seconds: number;
}

// Chat Providers
export interface ChatProviderConfig {
  id: string;
  name: string;
  llm_provider: string;
  llm_model: string;
  profile_id?: string | null;
  profile_name?: string | null;
  execution_mode: "direct" | "rlm" | "rag";
  runtime_settings: RuntimeSettings;
  rag_config?: RAGConfig | null;
  rlm_max_steps: number;
  rlm_timeout_seconds: number;
  created_at: string;
  updated_at: string;
}

export interface ChatProviderCreateRequest {
  name: string;
  llm_provider: string;
  llm_model: string;
  profile_id?: string | null;
  execution_mode?: "direct" | "rlm" | "rag";
  runtime_settings?: RuntimeSettings | null;
  rag_config?: RAGConfig | null;
  rlm_max_steps?: number | null;
  rlm_timeout_seconds?: number | null;
}

export interface ChatProviderUpdateRequest {
  name?: string | null;
  llm_model?: string | null;
  profile_id?: string | null;
  execution_mode?: "direct" | "rlm" | "rag" | null;
  runtime_settings?: RuntimeSettings | null;
  rag_config?: RAGConfig | null;
  rlm_max_steps?: number | null;
  rlm_timeout_seconds?: number | null;
}

export interface AppConfig {
  active_provider: string;
  active_model: string;
  budget: BudgetConfig;
  sandbox: { type: string; docker_image: string | null };
  appearance: { theme: string; sidebar_collapsed: boolean };
  provider_configs: ProviderConfigEntry[];
  default_runtime_settings: RuntimeSettings;
  mode_config: ModeConfig;
  chat_providers: ChatProviderConfig[];
  active_profile_id?: string | null;
  judge_chat_provider_id?: string | null;
}

// Evaluations
export interface ThumbRatingRequest {
  execution_id: string;
  session_id: string;
  chat_provider_id: string;
  rating: "up" | "down";
}

export interface BestPickRequest {
  session_id: string;
  winner_execution_id: string;
}

export interface JudgeRequest {
  session_id: string;
  execution_ids: string[];
  mode?: "pointwise" | "pairwise" | "both";
}

export interface JudgeScoreData {
  id: string;
  execution_id: string;
  session_id: string;
  chat_provider_id: string;
  judge_provider_id: string;
  dimensions: Record<string, number>;
  overall_score: number;
  reasoning: string;
  created_at: string;
}

export interface ProviderQualityScore {
  chat_provider_id: string;
  chat_provider_name: string;
  thumb_up: number;
  thumb_down: number;
  thumb_score: number;
  best_picks: number;
  pick_rate: number;
  judge_avg_score: number | null;
  combined_score: number;
}

export interface EvaluationSummaryResponse {
  session_id: string;
  by_chat_provider: Record<string, ProviderQualityScore>;
  recommendation: string | null;
  recommendation_reason: string;
}

export interface SessionEvaluations {
  thumb_ratings: Array<{
    id: string;
    execution_id: string;
    session_id: string;
    chat_provider_id: string;
    rating: "up" | "down";
    created_at: string;
  }>;
  best_picks: Array<{
    id: string;
    session_id: string;
    winner_execution_id: string;
    created_at: string;
  }>;
  judge_scores: JudgeScoreData[];
  judge_pairwise: Array<{
    id: string;
    session_id: string;
    execution_id_a: string;
    execution_id_b: string;
    winner: "a" | "b" | "tie";
    judge_provider_id: string;
    reasoning: string;
    created_at: string;
  }>;
}

export interface HealthResponse {
  status: string;
  version: string;
  uptime_seconds: number;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API error ${resp.status}: ${text}`);
  }
  if (resp.status === 204) return undefined as unknown as T;
  return resp.json();
}

// Health
export const getHealth = () => fetchJSON<HealthResponse>("/health");

// Chat
export const submitChat = (req: ChatRequest) =>
  fetchJSON<ChatResponse>("/api/chat", { method: "POST", body: JSON.stringify(req) });

// Files
export async function uploadFile(file: File): Promise<FileUploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${API_BASE}/api/files/upload`, { method: "POST", body: form });
  if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
  return resp.json();
}

// Sessions
export const getSessions = (limit = 20, offset = 0) =>
  fetchJSON<SessionSummary[]>(`/api/sessions?limit=${limit}&offset=${offset}`);

export const getSession = (id: string) => fetchJSON<SessionDetail>(`/api/sessions/${id}`);

export const deleteSession = (id: string) =>
  fetchJSON<void>(`/api/sessions/${id}`, { method: "DELETE" });

export const renameSession = (id: string, name: string) =>
  fetchJSON<SessionSummary>(`/api/sessions/${id}`, {
    method: "PUT",
    body: JSON.stringify({ name }),
  });

// Metrics
export const getMetrics = (sessionId: string) =>
  fetchJSON<MetricsResponse>(`/api/metrics/${sessionId}`);

// Executions & Traces
export interface ExecutionSummary {
  execution_id: string;
  session_id: string;
  query: string;
  mode: ChatMode;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  total_tokens: number;
  total_cost: number;
  chat_provider_id?: string | null;
  chat_provider_name?: string | null;
}

export const getExecutions = (limit = 20, chatProviderId?: string) => {
  const params = new URLSearchParams({ limit: String(limit) });
  if (chatProviderId) params.set("chat_provider_id", chatProviderId);
  return fetchJSON<ExecutionSummary[]>(`/api/executions?${params}`);
};

export const getTrace = (executionId: string) =>
  fetchJSON<TraceResponse>(`/api/traces/${executionId}`);

// Providers
export const getProviders = () => fetchJSON<ProviderInfo[]>("/api/providers");

export const testProvider = (req: ProviderTestRequest) =>
  fetchJSON<ProviderTestResponse>("/api/providers/test", {
    method: "POST",
    body: JSON.stringify(req),
  });

export interface ProviderSaveRequest {
  api_key?: string | null;
  model?: string | null;
  endpoint?: string | null;
  runtime_settings?: RuntimeSettings | null;
  enabled?: boolean | null;
}

export interface ProviderSaveResponse {
  saved: boolean;
  provider: string;
  env_var?: string | null;
  message: string;
}

export const saveProvider = (providerName: string, req: ProviderSaveRequest) =>
  fetchJSON<ProviderSaveResponse>(`/api/providers/${providerName}`, {
    method: "PUT",
    body: JSON.stringify(req),
  });

export const getProviderModels = (name: string, endpoint?: string) =>
  fetchJSON<ModelInfo[]>(
    `/api/providers/${name}/models${endpoint ? `?endpoint=${encodeURIComponent(endpoint)}` : ""}`,
  );

// Config

/** Fields accepted by PUT /api/config. active_provider/active_model are
 *  intentionally absent — those are set exclusively via PUT /api/providers/{name}. */
export interface ConfigUpdateRequest {
  budget?: BudgetConfig;
  sandbox?: { type: string; docker_image?: string | null };
  appearance?: { theme: string; sidebar_collapsed: boolean };
  provider_configs?: ProviderConfigEntry[];
  default_runtime_settings?: RuntimeSettings;
  mode_config?: Partial<ModeConfig>;
  chat_providers?: ChatProviderConfig[];
  judge_chat_provider_id?: string | null;
}

export const getConfig = () => fetchJSON<AppConfig>("/api/config");

export const updateConfig = (update: ConfigUpdateRequest) =>
  fetchJSON<AppConfig>("/api/config", { method: "PUT", body: JSON.stringify(update) });

// Profiles
export interface RunProfile {
  id: string;
  name: string;
  description: string;
  strategy: string;
  default_provider: string | null;
  providers_enabled: string[];
  runtime_settings: RuntimeSettings;
  budget: BudgetConfig;
  system_prompts: Record<string, string>;
  is_builtin: boolean;
}

export interface RunProfileCreate {
  name: string;
  description?: string;
  strategy?: string;
  default_provider?: string | null;
  providers_enabled?: string[];
  runtime_settings?: RuntimeSettings;
  budget?: BudgetConfig;
  system_prompts?: Record<string, string>;
}

export const getProfiles = () => fetchJSON<RunProfile[]>("/api/profiles");

export const createProfile = (req: RunProfileCreate) =>
  fetchJSON<RunProfile>("/api/profiles", { method: "POST", body: JSON.stringify(req) });

export const updateProfile = (id: string, req: Partial<RunProfileCreate>) =>
  fetchJSON<RunProfile>(`/api/profiles/${id}`, { method: "PUT", body: JSON.stringify(req) });

export const deleteProfile = (id: string) =>
  fetchJSON<void>(`/api/profiles/${id}`, { method: "DELETE" });

export const activateProfile = (id: string) =>
  fetchJSON<RunProfile>(`/api/profiles/${id}/activate`, { method: "POST" });

// System Prompts
export interface SystemPrompts {
  direct: string;
  rlm: string;
  rag: string;
}

export interface SystemPromptTemplate {
  name: string;
  description: string;
  prompts: Record<string, string>;
}

export const getSystemPrompts = () => fetchJSON<SystemPrompts>("/api/system-prompts");

export const updateSystemPrompts = (prompts: SystemPrompts) =>
  fetchJSON<SystemPrompts>("/api/system-prompts", { method: "PUT", body: JSON.stringify(prompts) });

export const getPromptTemplates = () =>
  fetchJSON<SystemPromptTemplate[]>("/api/system-prompts/templates");

// Chat Providers CRUD
export const getChatProviders = () => fetchJSON<ChatProviderConfig[]>("/api/chat-providers");

export const createChatProvider = (req: ChatProviderCreateRequest) =>
  fetchJSON<ChatProviderConfig>("/api/chat-providers", {
    method: "POST",
    body: JSON.stringify(req),
  });

export const updateChatProvider = (id: string, req: ChatProviderUpdateRequest) =>
  fetchJSON<ChatProviderConfig>(`/api/chat-providers/${id}`, {
    method: "PUT",
    body: JSON.stringify(req),
  });

export const deleteChatProvider = (id: string) =>
  fetchJSON<void>(`/api/chat-providers/${id}`, { method: "DELETE" });

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

export interface WSMessage {
  type: string;
  id?: string;
  data?: unknown;
  session_id?: string;
  mode?: ChatMode;
}

// Evaluations
export const submitThumbRating = (req: ThumbRatingRequest) =>
  fetchJSON<{ status: string }>("/api/evaluations/thumb", {
    method: "POST",
    body: JSON.stringify(req),
  });

export const removeThumbRating = (executionId: string) =>
  fetchJSON<{ status: string }>(`/api/evaluations/thumb/${executionId}`, {
    method: "DELETE",
  });

export const submitBestPick = (req: BestPickRequest) =>
  fetchJSON<{ status: string }>("/api/evaluations/best-pick", {
    method: "POST",
    body: JSON.stringify(req),
  });

export const getSessionEvaluations = (sessionId: string) =>
  fetchJSON<SessionEvaluations>(`/api/evaluations/${sessionId}`);

export const getEvaluationSummary = (sessionId: string) =>
  fetchJSON<EvaluationSummaryResponse>(`/api/evaluations/${sessionId}/summary`);

export const triggerJudge = (req: JudgeRequest) =>
  fetchJSON<{ pointwise: JudgeScoreData[]; pairwise: unknown[] }>("/api/evaluations/judge", {
    method: "POST",
    body: JSON.stringify(req),
  });
