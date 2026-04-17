/**
 * Cookbook provider catalog — single source of truth for the
 * /learn/cookbook landing page and the per-provider guide pages.
 *
 * `id` is the URL slug used at /learn/cookbook/[id] and is also the
 * suffix of the doc slug (always `hosts-${id}` — see the backend
 * allowlist in src/rlmkit/server/routes/docs.py). Keep these in sync.
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
}

export const COOKBOOK_PROVIDERS: ReadonlyArray<CookbookProvider> = [
  {
    id: "ollama",
    name: "Ollama",
    difficulty: "Easy",
    bestFor: "Quick local start on macOS or Linux.",
    group: "Easy local",
  },
  {
    id: "lmstudio",
    name: "LM Studio",
    difficulty: "Easy",
    bestFor: "GUI-driven local inference, cross-platform.",
    group: "Easy local",
  },
  {
    id: "vllm",
    name: "vLLM",
    difficulty: "Advanced",
    bestFor: "High-throughput GPU inference on Linux.",
    group: "Advanced local / self-hosted",
  },
  {
    id: "dgx-spark",
    name: "DGX Spark",
    difficulty: "Advanced",
    bestFor: "Self-hosted Grace Blackwell workstation.",
    group: "Advanced local / self-hosted",
  },
  {
    id: "openai",
    name: "OpenAI",
    difficulty: "Moderate",
    bestFor: "Broad model selection, pay-per-use cloud.",
    group: "Cloud",
  },
  {
    id: "anthropic",
    name: "Anthropic",
    difficulty: "Moderate",
    bestFor: "Claude family; strong at reasoning and long context.",
    group: "Cloud",
  },
  {
    id: "groq",
    name: "Groq",
    difficulty: "Moderate",
    bestFor: "Fastest cloud inference for supported open models.",
    group: "Cloud",
  },
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
