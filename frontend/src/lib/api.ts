/**
 * Typed API client for the RLMKit backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
}

export interface ChatResponse {
  execution_id: string;
  session_id: string;
  status: string;
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
}

export interface MetricsResponse {
  session_id: string;
  summary: MetricsSummary;
  by_mode: Partial<Record<ChatMode, ModeSummary>>;
  by_provider: Record<string, { queries: number; total_tokens: number; total_cost_usd: number }>;
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
  result: { answer: string; success: boolean };
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

export interface AppConfig {
  active_provider: string;
  active_model: string;
  budget: BudgetConfig;
  sandbox: { type: string; docker_image: string | null };
  appearance: { theme: string; sidebar_collapsed: boolean };
  provider_configs: ProviderConfigEntry[];
  default_runtime_settings: RuntimeSettings;
  mode_config: ModeConfig;
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

export const getFile = (id: string) => fetchJSON<FileUploadResponse>(`/api/files/${id}`);

// Sessions
export const getSessions = (limit = 20, offset = 0) =>
  fetchJSON<SessionSummary[]>(`/api/sessions?limit=${limit}&offset=${offset}`);

export const getSession = (id: string) => fetchJSON<SessionDetail>(`/api/sessions/${id}`);

export const deleteSession = (id: string) =>
  fetchJSON<void>(`/api/sessions/${id}`, { method: "DELETE" });

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
}

export const getExecutions = (limit = 20) =>
  fetchJSON<ExecutionSummary[]>(`/api/executions?limit=${limit}`);

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

// Config
export const getConfig = () => fetchJSON<AppConfig>("/api/config");

export const updateConfig = (update: Partial<AppConfig>) =>
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

export function connectChatWS(sessionId: string): WebSocket {
  const wsBase = API_BASE.replace(/^http/, "ws");
  return new WebSocket(`${wsBase}/ws/chat/${sessionId}`);
}
