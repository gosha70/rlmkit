/**
 * Cookbook provider catalog — single source of truth for the
 * /learn/cookbook landing page and the per-provider guide pages.
 *
 * `id` is the URL slug used at /learn/cookbook/[id] and is also the
 * suffix of the doc slug (always `hosts-${id}` — see the backend
 * allowlist in src/rlmstudio/server/routes/docs.py).
 *
 * `backendKey` is the value the Settings LLM-Provider form uses for
 * its `backend` field. For most providers this equals `id`; DGX Spark
 * is the exception — it configures as a vLLM endpoint.
 *
 * `defaultBaseUrl` and `defaultModel` populate the "Open in Settings"
 * deep link (the values the confirmation banner then asks the user
 * to accept). Leave undefined when the guide does not promise a
 * single known-good default. API keys are NEVER included — the deep
 * link's core safety property.
 */

export type ProviderDifficulty = "Easy" | "Moderate" | "Advanced";

export type ProviderGroup =
  | "Easy local"
  | "Advanced local / self-hosted"
  | "Cloud";

export interface CookbookProvider {
  id: string;
  name: string;
  difficulty: ProviderDifficulty;
  bestFor: string;
  group: ProviderGroup;
  backendKey: string;
  defaultBaseUrl?: string;
  defaultModel?: string;
}

export const COOKBOOK_PROVIDERS: ReadonlyArray<CookbookProvider> = [
  {
    id: "ollama",
    name: "Ollama",
    difficulty: "Easy",
    bestFor: "Quick local start on macOS or Linux.",
    group: "Easy local",
    backendKey: "ollama",
    defaultBaseUrl: "http://localhost:11434",
    defaultModel: "llama3.1:8b",
  },
  {
    id: "lmstudio",
    name: "LM Studio",
    difficulty: "Easy",
    bestFor: "GUI-driven local inference, cross-platform.",
    group: "Easy local",
    backendKey: "lmstudio",
    defaultBaseUrl: "http://localhost:1234/v1",
  },
  {
    id: "vllm",
    name: "vLLM",
    difficulty: "Advanced",
    bestFor: "High-throughput GPU inference on Linux.",
    group: "Advanced local / self-hosted",
    backendKey: "vllm",
    defaultBaseUrl: "http://localhost:8000/v1",
  },
  {
    id: "dgx-spark",
    name: "DGX Spark",
    difficulty: "Advanced",
    bestFor: "Self-hosted Grace Blackwell workstation.",
    group: "Advanced local / self-hosted",
    backendKey: "vllm",
  },
  {
    id: "openai",
    name: "OpenAI",
    difficulty: "Moderate",
    bestFor: "Broad model selection, pay-per-use cloud.",
    group: "Cloud",
    backendKey: "openai",
    defaultModel: "gpt-4o-mini",
  },
  {
    id: "anthropic",
    name: "Anthropic",
    difficulty: "Moderate",
    bestFor: "Claude family; strong at reasoning and long context.",
    group: "Cloud",
    backendKey: "anthropic",
    defaultModel: "claude-sonnet-4-6",
  },
  // Groq intentionally omitted: the app's provider catalog
  // (src/rlmstudio/ui/data/providers_catalog.py) does not currently
  // support a `groq` backend, so a Cookbook entry for it would send
  // users to a path that /api/llm-providers can't complete. Add back
  // once a Groq ProviderEntry ships.
];

export const PROVIDER_GROUPS_IN_ORDER: ReadonlyArray<ProviderGroup> = [
  "Easy local",
  "Advanced local / self-hosted",
  "Cloud",
];

export function getProviderById(id: string): CookbookProvider | undefined {
  return COOKBOOK_PROVIDERS.find((p) => p.id === id);
}

export function docSlugForProvider(id: string): string {
  return `hosts-${id}`;
}

/**
 * Build the `/settings?…` deep link for a Cookbook provider.
 *
 * Uses `provider` for the backend key (matches Settings' query-param
 * contract from the pitch) and includes `baseUrl` / `model` only when
 * the catalog has a known-good default. API keys are deliberately
 * absent — the security boundary the pitch calls out.
 */
export function settingsDeepLinkFor(provider: CookbookProvider): string {
  const params = new URLSearchParams();
  params.set("provider", provider.backendKey);
  if (provider.defaultBaseUrl) params.set("baseUrl", provider.defaultBaseUrl);
  if (provider.defaultModel) params.set("model", provider.defaultModel);
  return `/settings?${params.toString()}`;
}
