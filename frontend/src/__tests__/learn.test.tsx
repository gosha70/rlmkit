/**
 * Learn tab tests.
 *
 * Covers:
 *   - LearnPage landing shell (heading, diagnostics strip mount, cards region)
 *   - DiagnosticsStrip renderer (ok / warn / error states, fixUrl linking,
 *     loading state, accessibility labels)
 *   - MarkdownDoc loader (loading / error / rendered markdown, heading ids)
 *   - markdown-toc helpers (slugify, TOC extraction)
 *   - ProviderCard and CookbookPage (grouping, card rendering, link targets,
 *     diagnostics strip mount on cookbook)
 *   - ProviderGuidePage (known provider renders guide, unknown provider
 *     shows an alert, back link, Open in Settings deep link target)
 */

import { render, screen } from "@testing-library/react";
import { describe, test, expect, vi } from "vitest";
import useSWR from "swr";
import type { DiagnosticCheck, DiagnosticsResponse } from "@/lib/api";

vi.mock("@/components/shared/app-shell", () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  AppShell: ({ children }: { children: any }) => children,
}));

vi.mock("swr", () => ({
  default: vi.fn(),
}));

const mockUseSWR = vi.mocked(useSWR);

vi.mock("next/navigation", async () => {
  const actual = await vi.importActual<typeof import("next/navigation")>(
    "next/navigation",
  );
  return {
    ...actual,
    useParams: vi.fn(),
  };
});

import { useParams } from "next/navigation";

import LearnPage from "@/app/learn/page";
import CookbookPage from "@/app/learn/cookbook/page";
import ProviderGuidePage from "@/app/learn/cookbook/[provider]/page";
import TroubleshootPage from "@/app/learn/troubleshoot/page";
import { DiagnosticsStrip } from "@/components/learn/diagnostics-strip";
import { DiagnosticsPanel } from "@/components/learn/diagnostics-panel";
import { MarkdownDoc } from "@/components/learn/markdown-doc";
import { ProviderCard } from "@/components/learn/provider-card";
import { TroubleshootEntry } from "@/components/learn/troubleshoot-entry";
import { TroubleshootSearch } from "@/components/learn/troubleshoot-search";
import {
  filterTroubleshootEntries,
  TROUBLESHOOT_CATEGORIES,
} from "@/components/learn/troubleshoot-filter";
import type {
  TroubleshootEntry as TroubleshootEntryData,
  TroubleshootResponse,
} from "@/lib/api";
import {
  COOKBOOK_PROVIDERS,
  PROVIDER_GROUPS_IN_ORDER,
  docSlugForProvider,
  getProviderById,
  settingsDeepLinkFor,
} from "@/components/learn/provider-catalog";
import {
  DeepLinkBanner,
  parseDeepLinkFromParams,
} from "@/components/settings/deep-link-banner";
import {
  extractHeadings,
  slugifyHeading,
  topLevelHeadings,
} from "@/components/learn/markdown-toc";

const mockUseParams = vi.mocked(useParams);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const ok = (message = "OK"): DiagnosticCheck => ({ status: "ok", message });
const warn = (message: string, fixUrl?: string): DiagnosticCheck => ({
  status: "warn",
  message,
  fixUrl: fixUrl ?? null,
});
const err = (message: string, fixUrl?: string): DiagnosticCheck => ({
  status: "error",
  message,
  fixUrl: fixUrl ?? null,
});

const allOk: DiagnosticsResponse = {
  backend: ok("Backend reachable"),
  provider: ok("1 enabled provider(s)"),
  judge: ok("Judge configured"),
  storage: ok("Storage reachable"),
};

// ---------------------------------------------------------------------------
// LearnPage
// ---------------------------------------------------------------------------

describe("LearnPage", () => {
  test("renders heading at h2 (AppShell owns h1)", () => {
    mockUseSWR.mockReturnValue({ data: allOk } as ReturnType<typeof useSWR>);
    render(<LearnPage />);
    expect(
      screen.getByRole("heading", { level: 2, name: "Learn" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        /Understand RLM Studio, set up providers, and troubleshoot common issues\./,
      ),
    ).toBeInTheDocument();
  });

  test("renders the landing cards region as a named section", () => {
    mockUseSWR.mockReturnValue({ data: allOk } as ReturnType<typeof useSWR>);
    render(<LearnPage />);
    expect(
      screen.getByRole("region", { name: "Learn landing cards" }),
    ).toBeInTheDocument();
  });

  test("mounts the diagnostics strip region", () => {
    mockUseSWR.mockReturnValue({ data: allOk } as ReturnType<typeof useSWR>);
    render(<LearnPage />);
    expect(
      screen.getByRole("status", { name: "System diagnostics" }),
    ).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// DiagnosticsStrip
// ---------------------------------------------------------------------------

describe("DiagnosticsStrip", () => {
  test("renders loading state when data is null", () => {
    render(<DiagnosticsStrip data={null} />);
    expect(screen.getByLabelText("Backend: loading")).toBeInTheDocument();
    expect(screen.getByLabelText("Provider: loading")).toBeInTheDocument();
    expect(screen.getByLabelText("Judge: loading")).toBeInTheDocument();
    expect(screen.getByLabelText("Storage: loading")).toBeInTheDocument();
  });

  test("renders ok state for all four checks", () => {
    render(<DiagnosticsStrip data={allOk} />);
    expect(
      screen.getByLabelText(/Backend: OK — Backend reachable/),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText(/Provider: OK — 1 enabled provider/),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText(/Judge: OK — Judge configured/),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText(/Storage: OK — Storage reachable/),
    ).toBeInTheDocument();
  });

  test("renders warn state with correct label", () => {
    const data: DiagnosticsResponse = {
      ...allOk,
      judge: warn("Judge not configured", "/settings"),
    };
    render(<DiagnosticsStrip data={data} />);
    expect(
      screen.getByLabelText(/Judge: Warning — Judge not configured/),
    ).toBeInTheDocument();
  });

  test("renders error state with correct label", () => {
    const data: DiagnosticsResponse = {
      ...allOk,
      provider: err("No enabled LLM provider", "/settings"),
    };
    render(<DiagnosticsStrip data={data} />);
    expect(
      screen.getByLabelText(/Provider: Error — No enabled LLM provider/),
    ).toBeInTheDocument();
  });

  test("cell with fixUrl renders as a link", () => {
    const data: DiagnosticsResponse = {
      ...allOk,
      provider: err("No enabled LLM provider", "/settings"),
    };
    render(<DiagnosticsStrip data={data} />);
    const link = screen.getByRole("link", {
      name: /Provider: Error — No enabled LLM provider/,
    });
    expect(link).toHaveAttribute("href", "/settings");
  });

  test("cell without fixUrl does not render as a link", () => {
    // Backend is ok with no fixUrl in the default response.
    render(<DiagnosticsStrip data={allOk} />);
    const backend = screen.getByLabelText(/Backend: OK — Backend reachable/);
    expect(backend.tagName.toLowerCase()).not.toBe("a");
  });
});

// ---------------------------------------------------------------------------
// MarkdownDoc
// ---------------------------------------------------------------------------

describe("MarkdownDoc", () => {
  test("shows a loading state while data is pending", () => {
    mockUseSWR.mockReturnValue({
      data: undefined,
      error: undefined,
      isLoading: true,
    } as ReturnType<typeof useSWR>);

    render(<MarkdownDoc slug="rlm-concepts" />);
    expect(
      screen.getByRole("status", { name: "Loading document" }),
    ).toBeInTheDocument();
  });

  test("shows an error alert when fetch fails", () => {
    mockUseSWR.mockReturnValue({
      data: undefined,
      error: new Error("API error 404: not found"),
      isLoading: false,
    } as ReturnType<typeof useSWR>);

    render(<MarkdownDoc slug="rlm-concepts" />);
    expect(screen.getByRole("alert")).toHaveTextContent(
      /Couldn’t load this guide/,
    );
  });

  test("renders markdown content when data resolves", () => {
    mockUseSWR.mockReturnValue({
      data: {
        slug: "rlm-concepts",
        content: "# RLM Concepts\n\nHello **world**.",
      },
      error: undefined,
      isLoading: false,
    } as ReturnType<typeof useSWR>);

    const { container } = render(<MarkdownDoc slug="rlm-concepts" />);
    expect(
      screen.getByRole("heading", { level: 1, name: "RLM Concepts" }),
    ).toBeInTheDocument();
    expect(screen.getByText("world")).toBeInTheDocument();
    expect(container.querySelector("[data-slug='rlm-concepts']")).not.toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Provider catalog (data)
// ---------------------------------------------------------------------------

describe("provider catalog", () => {
  test("every provider has a unique id", () => {
    const ids = COOKBOOK_PROVIDERS.map((p) => p.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  test("every group in PROVIDER_GROUPS_IN_ORDER has at least one provider", () => {
    for (const group of PROVIDER_GROUPS_IN_ORDER) {
      const inGroup = COOKBOOK_PROVIDERS.filter((p) => p.group === group);
      expect(inGroup.length).toBeGreaterThan(0);
    }
  });

  test("docSlugForProvider matches the backend allowlist format", () => {
    expect(docSlugForProvider("ollama")).toBe("hosts-ollama");
    expect(docSlugForProvider("dgx-spark")).toBe("hosts-dgx-spark");
  });

  test("getProviderById returns known providers and undefined for unknown", () => {
    expect(getProviderById("ollama")?.name).toBe("Ollama");
    expect(getProviderById("does-not-exist")).toBeUndefined();
  });

  test("DGX Spark uses the vllm backend key", () => {
    // DGX Spark is configured as a vLLM endpoint in Settings — the
    // cookbook id and the backend key intentionally diverge.
    expect(getProviderById("dgx-spark")?.backendKey).toBe("vllm");
  });

  test("settingsDeepLinkFor builds a /settings URL with catalog defaults", () => {
    const ollama = getProviderById("ollama")!;
    const url = new URL(settingsDeepLinkFor(ollama), "http://localhost");
    expect(url.pathname).toBe("/settings");
    expect(url.searchParams.get("provider")).toBe("ollama");
    expect(url.searchParams.get("baseUrl")).toBe("http://localhost:11434");
    expect(url.searchParams.get("model")).toBe("llama3.1:8b");
  });

  test("settingsDeepLinkFor omits baseUrl/model when catalog has no default", () => {
    // DGX Spark leaves baseUrl and model unset in the catalog because
    // the known-good values depend on the operator's hardware.
    const dgx = getProviderById("dgx-spark")!;
    const url = new URL(settingsDeepLinkFor(dgx), "http://localhost");
    expect(url.searchParams.get("provider")).toBe("vllm");
    expect(url.searchParams.get("baseUrl")).toBeNull();
    expect(url.searchParams.get("model")).toBeNull();
  });

  test("settingsDeepLinkFor never emits an api_key param", () => {
    // The pitch's deep-link security boundary: secrets must never
    // travel in the URL. Verify the builder doesn't sneak one in.
    for (const provider of COOKBOOK_PROVIDERS) {
      const url = new URL(settingsDeepLinkFor(provider), "http://localhost");
      expect(url.searchParams.get("api_key")).toBeNull();
      expect(url.searchParams.get("apiKey")).toBeNull();
    }
  });
});

// ---------------------------------------------------------------------------
// ProviderCard
// ---------------------------------------------------------------------------

describe("ProviderCard", () => {
  test("renders provider name, difficulty, and best-for copy", () => {
    const ollama = getProviderById("ollama")!;
    render(<ProviderCard provider={ollama} />);
    expect(screen.getByText("Ollama")).toBeInTheDocument();
    expect(screen.getByText("Easy")).toBeInTheDocument();
    expect(screen.getByText(/Quick local start/)).toBeInTheDocument();
  });

  test("renders a link that points at the provider guide route", () => {
    const openai = getProviderById("openai")!;
    render(<ProviderCard provider={openai} />);
    const link = screen.getByRole("link", {
      name: /Open OpenAI guide \(Moderate\)/,
    });
    expect(link).toHaveAttribute("href", "/learn/cookbook/openai");
  });
});

// ---------------------------------------------------------------------------
// CookbookPage
// ---------------------------------------------------------------------------

describe("CookbookPage", () => {
  const allOkDiagnostics: DiagnosticsResponse = {
    backend: ok("Backend reachable"),
    provider: ok("1 LLM provider(s) configured"),
    judge: ok("Judge configured"),
    storage: ok("Storage reachable"),
  };

  test("renders heading and subtitle", () => {
    mockUseSWR.mockReturnValue({
      data: allOkDiagnostics,
    } as ReturnType<typeof useSWR>);
    render(<CookbookPage />);
    expect(
      screen.getByRole("heading", { level: 2, name: "Cookbook" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Connect a local or cloud model provider."),
    ).toBeInTheDocument();
  });

  test("mounts the diagnostics strip", () => {
    mockUseSWR.mockReturnValue({
      data: allOkDiagnostics,
    } as ReturnType<typeof useSWR>);
    render(<CookbookPage />);
    expect(
      screen.getByRole("status", { name: "System diagnostics" }),
    ).toBeInTheDocument();
  });

  test("renders every provider in COOKBOOK_PROVIDERS exactly once", () => {
    mockUseSWR.mockReturnValue({
      data: allOkDiagnostics,
    } as ReturnType<typeof useSWR>);
    render(<CookbookPage />);
    for (const p of COOKBOOK_PROVIDERS) {
      const matches = screen.getAllByRole("link", {
        name: new RegExp(`Open ${p.name} guide`),
      });
      expect(matches).toHaveLength(1);
      expect(matches[0]).toHaveAttribute(
        "href",
        `/learn/cookbook/${p.id}`,
      );
    }
  });

  test("renders grouped regions in the spec order", () => {
    mockUseSWR.mockReturnValue({
      data: allOkDiagnostics,
    } as ReturnType<typeof useSWR>);
    render(<CookbookPage />);
    const regions = screen.getAllByRole("region");
    const names = regions.map((r) => r.getAttribute("aria-labelledby"));
    // First region is the diagnostics strip is role=status (not region),
    // so we only see the grouped sections here.
    expect(names).toEqual([
      "cookbook-group-easy-local",
      "cookbook-group-advanced-local-self-hosted",
      "cookbook-group-cloud",
    ]);
  });
});

// ---------------------------------------------------------------------------
// markdown-toc helpers
// ---------------------------------------------------------------------------

describe("markdown-toc", () => {
  test("slugifyHeading lowercases and hyphenates non-word characters", () => {
    expect(slugifyHeading("1. Install")).toBe("1-install");
    expect(slugifyHeading("Add to RLM Studio")).toBe("add-to-rlm-studio");
    expect(slugifyHeading("  5. Test connection  ")).toBe("5-test-connection");
  });

  test("extractHeadings picks up H2 and H3, skipping fenced code", () => {
    const source = `# Top\n\n## Install\n\n\`\`\`\n## fake-heading-in-code\n\`\`\`\n\n### Detail\n\n## Start\n`;
    const headings = extractHeadings(source);
    expect(headings.map((h) => [h.level, h.text])).toEqual([
      [2, "Install"],
      [3, "Detail"],
      [2, "Start"],
    ]);
  });

  test("topLevelHeadings returns only H2s", () => {
    const source = "## Install\n### Detail\n## Test\n";
    const tops = topLevelHeadings(source);
    expect(tops.map((h) => h.text)).toEqual(["Install", "Test"]);
    expect(tops.map((h) => h.id)).toEqual(["install", "test"]);
  });
});

// ---------------------------------------------------------------------------
// MarkdownDoc heading ids
// ---------------------------------------------------------------------------

describe("MarkdownDoc heading ids", () => {
  test("H2 and H3 elements receive slugified ids", () => {
    mockUseSWR.mockReturnValue({
      data: {
        slug: "rlm-concepts",
        content:
          "## Install\n\nbody\n\n### Detailed notes\n\nmore body\n",
      },
      error: undefined,
      isLoading: false,
    } as ReturnType<typeof useSWR>);

    render(<MarkdownDoc slug="rlm-concepts" />);
    const h2 = screen.getByRole("heading", { level: 2, name: "Install" });
    const h3 = screen.getByRole("heading", { level: 3, name: "Detailed notes" });
    expect(h2).toHaveAttribute("id", "install");
    expect(h3).toHaveAttribute("id", "detailed-notes");
  });
});

// ---------------------------------------------------------------------------
// ProviderGuidePage
// ---------------------------------------------------------------------------

describe("ProviderGuidePage", () => {
  const sampleDoc = {
    slug: "hosts-ollama",
    content: "## Install\n\nbody\n\n## Start the server\n\nmore body\n",
  };

  test("renders the guide for a known provider with TOC and markdown", () => {
    mockUseParams.mockReturnValue({ provider: "ollama" });
    // LearnPage/Guide both call SWR — data will flow to both calls
    // because they share the same cache key via the mock's single
    // return value. Use a flexible default.
    mockUseSWR.mockImplementation(
      (key: unknown) => {
        if (Array.isArray(key) && key[0] === "learn-doc") {
          return { data: sampleDoc } as ReturnType<typeof useSWR>;
        }
        return { data: undefined } as ReturnType<typeof useSWR>;
      },
    );

    render(<ProviderGuidePage />);
    expect(
      screen.getByRole("heading", { level: 2, name: "Ollama" }),
    ).toBeInTheDocument();
    // Left-rail anchors use the topLevelHeadings output.
    expect(screen.getByRole("link", { name: "Install" })).toHaveAttribute(
      "href",
      "#install",
    );
    expect(
      screen.getByRole("link", { name: "Start the server" }),
    ).toHaveAttribute("href", "#start-the-server");
    // Markdown body rendered (H2 from content, with id set by MarkdownDoc).
    expect(
      screen.getByRole("heading", { level: 2, name: "Install" }),
    ).toBeInTheDocument();
  });

  test("Open in Settings link carries backend key + catalog defaults", () => {
    mockUseParams.mockReturnValue({ provider: "ollama" });
    mockUseSWR.mockReturnValue({ data: sampleDoc } as ReturnType<typeof useSWR>);
    render(<ProviderGuidePage />);
    const link = screen.getByRole("link", {
      name: "Open Ollama in Settings",
    });
    const href = link.getAttribute("href") ?? "";
    const url = new URL(href, "http://localhost");
    expect(url.pathname).toBe("/settings");
    expect(url.searchParams.get("provider")).toBe("ollama");
    expect(url.searchParams.get("baseUrl")).toBe("http://localhost:11434");
    expect(url.searchParams.get("model")).toBe("llama3.1:8b");
  });

  test("DGX Spark's Open in Settings uses vllm backend key without defaults", () => {
    mockUseParams.mockReturnValue({ provider: "dgx-spark" });
    mockUseSWR.mockReturnValue({ data: sampleDoc } as ReturnType<typeof useSWR>);
    render(<ProviderGuidePage />);
    const link = screen.getByRole("link", {
      name: "Open DGX Spark in Settings",
    });
    const url = new URL(link.getAttribute("href") ?? "", "http://localhost");
    expect(url.searchParams.get("provider")).toBe("vllm");
    expect(url.searchParams.get("baseUrl")).toBeNull();
    expect(url.searchParams.get("model")).toBeNull();
  });

  test("unknown provider id shows an alert, not a crash", () => {
    mockUseParams.mockReturnValue({ provider: "definitely-not-real" });
    mockUseSWR.mockReturnValue({ data: undefined } as ReturnType<typeof useSWR>);
    render(<ProviderGuidePage />);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent(/Provider not found/);
    expect(alert).toHaveTextContent(/definitely-not-real/);
    // Back link inside the alert goes home.
    const backLinks = screen.getAllByRole("link", { name: /Back to Cookbook/ });
    expect(backLinks.length).toBeGreaterThan(0);
    for (const l of backLinks) {
      expect(l).toHaveAttribute("href", "/learn/cookbook");
    }
  });
});

// ---------------------------------------------------------------------------
// DeepLinkBanner
// ---------------------------------------------------------------------------

describe("DeepLinkBanner", () => {
  const baseValues = {
    provider: "ollama",
    baseUrl: "http://localhost:11434",
    model: "llama3.1:8b",
  };

  test("renders heading, fields, and the required secrets reassurance", () => {
    render(
      <DeepLinkBanner
        values={baseValues}
        providerDisplayName="Ollama"
        onCancel={() => {}}
        onUseValues={() => {}}
      />,
    );
    expect(
      screen.getByRole("region", {
        name: "Review provider values from this guide",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("Ollama")).toBeInTheDocument();
    expect(screen.getByText("http://localhost:11434")).toBeInTheDocument();
    expect(screen.getByText("llama3.1:8b")).toBeInTheDocument();
    // Pitch §Resolved Decisions #4 — this copy is load-bearing.
    expect(
      screen.getByText(
        "No API keys or secrets will be filled in automatically.",
      ),
    ).toBeInTheDocument();
  });

  test("Cancel is the safe default and fires onCancel", async () => {
    const onCancel = vi.fn();
    const onUseValues = vi.fn();
    render(
      <DeepLinkBanner
        values={baseValues}
        providerDisplayName="Ollama"
        onCancel={onCancel}
        onUseValues={onUseValues}
      />,
    );
    const cancel = screen.getByRole("button", { name: "Cancel" });
    const { default: userEvent } = await import("@testing-library/user-event");
    await userEvent.setup().click(cancel);
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onUseValues).not.toHaveBeenCalled();
  });

  test("Use values fires onUseValues", async () => {
    const onCancel = vi.fn();
    const onUseValues = vi.fn();
    render(
      <DeepLinkBanner
        values={baseValues}
        providerDisplayName="Ollama"
        onCancel={onCancel}
        onUseValues={onUseValues}
      />,
    );
    const use = screen.getByRole("button", { name: "Use values" });
    const { default: userEvent } = await import("@testing-library/user-event");
    await userEvent.setup().click(use);
    expect(onUseValues).toHaveBeenCalledTimes(1);
    expect(onCancel).not.toHaveBeenCalled();
  });

  test("omits optional rows when baseUrl or model are missing", () => {
    render(
      <DeepLinkBanner
        values={{ provider: "anthropic" }}
        providerDisplayName="Anthropic"
        onCancel={() => {}}
        onUseValues={() => {}}
      />,
    );
    expect(screen.getByText("Anthropic")).toBeInTheDocument();
    expect(screen.queryByText("Base URL:")).not.toBeInTheDocument();
    expect(screen.queryByText("Model:")).not.toBeInTheDocument();
  });

  test("falls back to raw provider key when no display name is given", () => {
    render(
      <DeepLinkBanner
        values={{ provider: "vllm" }}
        onCancel={() => {}}
        onUseValues={() => {}}
      />,
    );
    expect(screen.getByText("vllm")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// parseDeepLinkFromParams — security-critical parsing
// ---------------------------------------------------------------------------

describe("parseDeepLinkFromParams", () => {
  const ALLOWED = new Set(["openai", "anthropic", "ollama", "lmstudio", "vllm"]);
  const BASE_URL_ALLOWED = new Set(["ollama", "lmstudio", "vllm"]);

  test("returns null when provider is missing", () => {
    const params = new URLSearchParams();
    expect(parseDeepLinkFromParams(params, ALLOWED, BASE_URL_ALLOWED)).toBeNull();
  });

  test("returns null when provider is not allowlisted", () => {
    const params = new URLSearchParams("provider=evil");
    expect(parseDeepLinkFromParams(params, ALLOWED, BASE_URL_ALLOWED)).toBeNull();
  });

  test("accepts baseUrl for local backends", () => {
    const params = new URLSearchParams(
      "provider=ollama&baseUrl=http%3A%2F%2Flocalhost%3A11434&model=llama3.1:8b",
    );
    const values = parseDeepLinkFromParams(params, ALLOWED, BASE_URL_ALLOWED);
    expect(values).toEqual({
      provider: "ollama",
      baseUrl: "http://localhost:11434",
      model: "llama3.1:8b",
    });
  });

  test("strips baseUrl for cloud backends (P1 phishing defense)", () => {
    // Core attack scenario: attacker crafts a link that points
    // provider=openai at their host. If baseUrl were honored here it
    // would end up in hidden form state and the user's API key would
    // be routed to the attacker on save.
    const params = new URLSearchParams(
      "provider=openai&baseUrl=https%3A%2F%2Fevil.example.com&model=gpt-4o",
    );
    const values = parseDeepLinkFromParams(params, ALLOWED, BASE_URL_ALLOWED);
    expect(values).toEqual({
      provider: "openai",
      baseUrl: undefined,
      model: "gpt-4o",
    });
  });

  test("strips baseUrl for Anthropic too", () => {
    const params = new URLSearchParams(
      "provider=anthropic&baseUrl=https%3A%2F%2Fevil.example.com",
    );
    const values = parseDeepLinkFromParams(params, ALLOWED, BASE_URL_ALLOWED);
    expect(values?.baseUrl).toBeUndefined();
  });

  test("never reads api_key or apiKey params", () => {
    // The pitch's security boundary: secrets must never ride in the
    // URL. The parser must not even surface them into the DeepLinkValues
    // shape, even if a pathological caller adds them to the shape later.
    const params = new URLSearchParams(
      "provider=openai&api_key=sk-leak&apiKey=sk-leak2",
    );
    const values = parseDeepLinkFromParams(params, ALLOWED, BASE_URL_ALLOWED);
    expect(values).toEqual({
      provider: "openai",
      baseUrl: undefined,
      model: undefined,
    });
    const serialized = JSON.stringify(values);
    expect(serialized).not.toMatch(/sk-leak/);
  });
});

// ---------------------------------------------------------------------------
// filterTroubleshootEntries
// ---------------------------------------------------------------------------

describe("filterTroubleshootEntries", () => {
  const entries: TroubleshootEntryData[] = [
    {
      id: "a",
      title: "Empty Anthropic response",
      symptom: "Request succeeds but empty",
      cause: "temperature + top_p",
      category: "Provider",
      fix: ["Remove one"],
      seealso: ["cookbook/anthropic"],
    },
    {
      id: "b",
      title: "Judge flat at 5.0",
      symptom: "Every run scores 5",
      cause: "rubric v2 missing",
      category: "Judge",
      fix: [],
      seealso: [],
    },
    {
      id: "c",
      title: "Ollama model not found",
      symptom: "Ollama rejects",
      cause: "pull missing",
      category: "Setup",
      fix: [],
      seealso: ["cookbook/ollama"],
    },
  ];

  test("empty options returns every entry", () => {
    expect(filterTroubleshootEntries(entries)).toHaveLength(entries.length);
    expect(filterTroubleshootEntries(entries, {})).toHaveLength(entries.length);
    expect(
      filterTroubleshootEntries(entries, { query: "   " }),
    ).toHaveLength(entries.length);
  });

  test("query matches title, symptom, and cause", () => {
    expect(
      filterTroubleshootEntries(entries, { query: "anthropic" }).map((e) => e.id),
    ).toEqual(["a"]);
    expect(
      filterTroubleshootEntries(entries, { query: "EVERY RUN" }).map((e) => e.id),
    ).toEqual(["b"]);
    expect(
      filterTroubleshootEntries(entries, { query: "pull" }).map((e) => e.id),
    ).toEqual(["c"]);
  });

  test("category filter narrows regardless of query", () => {
    const setupOnly = new Set<typeof entries[number]["category"]>(["Setup"]);
    expect(
      filterTroubleshootEntries(entries, { categories: setupOnly }).map(
        (e) => e.id,
      ),
    ).toEqual(["c"]);
  });

  test("query and categories compose with AND semantics", () => {
    const providerOnly = new Set<typeof entries[number]["category"]>([
      "Provider",
    ]);
    expect(
      filterTroubleshootEntries(entries, {
        query: "anthropic",
        categories: providerOnly,
      }).map((e) => e.id),
    ).toEqual(["a"]);
    // Same query, wrong category = empty.
    expect(
      filterTroubleshootEntries(entries, {
        query: "anthropic",
        categories: new Set(["Judge"]),
      }),
    ).toEqual([]);
  });

  test("TROUBLESHOOT_CATEGORIES enumerates six spec values", () => {
    expect(TROUBLESHOOT_CATEGORIES).toEqual([
      "Setup",
      "Provider",
      "Compare",
      "Judge",
      "Budget",
      "Runtime",
    ]);
  });
});

// ---------------------------------------------------------------------------
// TroubleshootSearch
// ---------------------------------------------------------------------------

describe("TroubleshootSearch", () => {
  test("renders search input and one chip per category", () => {
    render(
      <TroubleshootSearch
        query=""
        onQueryChange={() => {}}
        categories={new Set()}
        onToggleCategory={() => {}}
      />,
    );
    expect(screen.getByRole("searchbox")).toBeInTheDocument();
    for (const cat of TROUBLESHOOT_CATEGORIES) {
      expect(screen.getByRole("button", { name: cat })).toBeInTheDocument();
    }
  });

  test("typing in the input fires onQueryChange", async () => {
    const onQueryChange = vi.fn();
    render(
      <TroubleshootSearch
        query=""
        onQueryChange={onQueryChange}
        categories={new Set()}
        onToggleCategory={() => {}}
      />,
    );
    const { default: userEvent } = await import("@testing-library/user-event");
    // Controlled component with query="" never accumulates — each keystroke
    // fires a change event with the single typed character since the parent
    // state never updates in the test.
    await userEvent.setup().type(screen.getByRole("searchbox"), "ab");
    expect(onQueryChange).toHaveBeenCalledTimes(2);
    expect(onQueryChange).toHaveBeenNthCalledWith(1, "a");
    expect(onQueryChange).toHaveBeenNthCalledWith(2, "b");
  });

  test("chip click fires onToggleCategory", async () => {
    const onToggle = vi.fn();
    render(
      <TroubleshootSearch
        query=""
        onQueryChange={() => {}}
        categories={new Set()}
        onToggleCategory={onToggle}
      />,
    );
    const { default: userEvent } = await import("@testing-library/user-event");
    await userEvent.setup().click(screen.getByRole("button", { name: "Provider" }));
    expect(onToggle).toHaveBeenCalledWith("Provider");
  });

  test("active chip carries aria-pressed=true", () => {
    render(
      <TroubleshootSearch
        query=""
        onQueryChange={() => {}}
        categories={new Set(["Judge"])}
        onToggleCategory={() => {}}
      />,
    );
    expect(screen.getByRole("button", { name: "Judge" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: "Setup" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });
});

// ---------------------------------------------------------------------------
// TroubleshootEntry (card)
// ---------------------------------------------------------------------------

describe("TroubleshootEntry card", () => {
  const entry: TroubleshootEntryData = {
    id: "anthropic-empty-response",
    title: "Empty Anthropic response",
    symptom: "Request succeeds but returns no useful output",
    cause: "Unsupported parameter combination",
    category: "Provider",
    fix: ["Remove one of temperature or top_p", "Retry"],
    seealso: ["cookbook/anthropic", "unknown/shape"],
  };

  test("renders title, category, symptom, cause, fix list", () => {
    render(<TroubleshootEntry entry={entry} />);
    expect(
      screen.getByRole("heading", { level: 3, name: entry.title }),
    ).toBeInTheDocument();
    expect(screen.getByText("Provider")).toBeInTheDocument();
    expect(screen.getByText(/Request succeeds but returns no useful output/))
      .toBeInTheDocument();
    expect(
      screen.getByText(/Unsupported parameter combination/),
    ).toBeInTheDocument();
    expect(screen.getByText("Retry")).toBeInTheDocument();
  });

  test("cookbook seealso links resolve, unknown shapes stay as text", () => {
    render(<TroubleshootEntry entry={entry} />);
    const cookbookLink = screen.getByRole("link", {
      name: /Cookbook: anthropic/,
    });
    expect(cookbookLink).toHaveAttribute("href", "/learn/cookbook/anthropic");
    expect(screen.getByText("unknown/shape")).toBeInTheDocument();
  });

  test("cookbook seealso for a provider NOT in the catalog stays as text", () => {
    // Regression guard for the Groq removal (commit 5d1c29c): a YAML
    // entry that references cookbook/groq must not render a live link
    // because the /learn/cookbook/groq route now lands on the
    // "Provider not found" alert. The card renders the raw ref as
    // plain text instead.
    const stale: TroubleshootEntryData = {
      ...entry,
      id: "stale-ref",
      seealso: ["cookbook/groq", "cookbook/anthorpic"],
    };
    render(<TroubleshootEntry entry={stale} />);
    expect(
      screen.queryByRole("link", { name: /Cookbook: groq/ }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: /Cookbook: anthorpic/ }),
    ).not.toBeInTheDocument();
    expect(screen.getByText("cookbook/groq")).toBeInTheDocument();
    expect(screen.getByText("cookbook/anthorpic")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// DiagnosticsPanel
// ---------------------------------------------------------------------------

describe("DiagnosticsPanel", () => {
  const sample: DiagnosticsResponse = {
    backend: ok("Backend reachable"),
    provider: err("No LLM provider configured", "/settings"),
    judge: warn("Judge not configured", "/settings"),
    storage: ok("Storage reachable"),
  };

  test("renders one row per check with a named region", () => {
    render(<DiagnosticsPanel data={sample} />);
    expect(
      screen.getByRole("region", { name: "Diagnostics" }),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText(/Backend: OK — Backend reachable/),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText(/Provider: Error — No LLM provider configured/),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText(/Judge: Warning — Judge not configured/),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText(/Storage: OK — Storage reachable/),
    ).toBeInTheDocument();
  });

  test("fixUrl rows render a Go-to CTA link", () => {
    render(<DiagnosticsPanel data={sample} />);
    expect(
      screen.getByRole("link", { name: /Go to Settings/ }),
    ).toHaveAttribute("href", "/settings");
  });

  test("rows without fixUrl do not render a CTA link", () => {
    render(<DiagnosticsPanel data={sample} />);
    const links = screen.getAllByRole("link");
    // Provider (error → /settings) + Judge (warn → /settings) only.
    expect(links).toHaveLength(2);
  });

  test("null data renders four loading rows", () => {
    render(<DiagnosticsPanel data={null} />);
    expect(screen.getByLabelText("Backend: loading")).toBeInTheDocument();
    expect(screen.getByLabelText("Provider: loading")).toBeInTheDocument();
    expect(screen.getByLabelText("Judge: loading")).toBeInTheDocument();
    expect(screen.getByLabelText("Storage: loading")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// TroubleshootPage
// ---------------------------------------------------------------------------

describe("TroubleshootPage", () => {
  const allOkDiagnostics: DiagnosticsResponse = {
    backend: ok("Backend reachable"),
    provider: ok("1 LLM provider(s) configured"),
    judge: ok("Judge configured"),
    storage: ok("Storage reachable"),
  };
  const troubleshootData: TroubleshootResponse = {
    entries: [
      {
        id: "a",
        title: "Empty Anthropic response",
        symptom: "Empty output",
        cause: "temperature + top_p",
        category: "Provider",
        fix: ["Remove one"],
        seealso: ["cookbook/anthropic"],
      },
      {
        id: "b",
        title: "Judge flat at 5.0",
        symptom: "Scores all 5",
        cause: "rubric v2",
        category: "Judge",
        fix: ["Update prompt"],
        seealso: [],
      },
    ],
  };

  const swrByKey = (
    troubleshoot: TroubleshootResponse | undefined,
    diagnostics: DiagnosticsResponse | undefined,
    troubleshootError?: Error,
  ) =>
    ((key: unknown) => {
      if (key === "learn-troubleshoot") {
        return {
          data: troubleshoot,
          error: troubleshootError,
        } as ReturnType<typeof useSWR>;
      }
      if (key === "learn-diagnostics") {
        return { data: diagnostics } as ReturnType<typeof useSWR>;
      }
      return { data: undefined } as ReturnType<typeof useSWR>;
    }) as unknown as typeof useSWR;

  test("renders heading, search, diagnostics strip, diagnostics panel", () => {
    mockUseSWR.mockImplementation(
      swrByKey(troubleshootData, allOkDiagnostics),
    );
    render(<TroubleshootPage />);
    expect(
      screen.getByRole("heading", { level: 2, name: "Troubleshoot" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("searchbox")).toBeInTheDocument();
    expect(
      screen.getByRole("status", { name: "System diagnostics" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("region", { name: "Diagnostics" }),
    ).toBeInTheDocument();
  });

  test("renders every entry by default", () => {
    mockUseSWR.mockImplementation(
      swrByKey(troubleshootData, allOkDiagnostics),
    );
    render(<TroubleshootPage />);
    expect(
      screen.getByRole("heading", { level: 3, name: "Empty Anthropic response" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { level: 3, name: "Judge flat at 5.0" }),
    ).toBeInTheDocument();
  });

  test("typing narrows results", async () => {
    mockUseSWR.mockImplementation(
      swrByKey(troubleshootData, allOkDiagnostics),
    );
    render(<TroubleshootPage />);
    const { default: userEvent } = await import("@testing-library/user-event");
    await userEvent.setup().type(screen.getByRole("searchbox"), "anthropic");
    expect(
      screen.getByRole("heading", { level: 3, name: "Empty Anthropic response" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { level: 3, name: "Judge flat at 5.0" }),
    ).not.toBeInTheDocument();
  });

  test("quick-filter chip narrows results by category", async () => {
    mockUseSWR.mockImplementation(
      swrByKey(troubleshootData, allOkDiagnostics),
    );
    render(<TroubleshootPage />);
    const { default: userEvent } = await import("@testing-library/user-event");
    await userEvent.setup().click(screen.getByRole("button", { name: "Judge" }));
    expect(
      screen.getByRole("heading", { level: 3, name: "Judge flat at 5.0" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", {
        level: 3,
        name: "Empty Anthropic response",
      }),
    ).not.toBeInTheDocument();
  });

  test("no-match state renders an empty-state message", async () => {
    mockUseSWR.mockImplementation(
      swrByKey(troubleshootData, allOkDiagnostics),
    );
    render(<TroubleshootPage />);
    const { default: userEvent } = await import("@testing-library/user-event");
    await userEvent
      .setup()
      .type(screen.getByRole("searchbox"), "definitely-not-a-match-zzz");
    expect(screen.getByText(/No entries match this filter/)).toBeInTheDocument();
  });

  test("error state renders an alert", () => {
    mockUseSWR.mockImplementation(
      swrByKey(undefined, allOkDiagnostics, new Error("500")),
    );
    render(<TroubleshootPage />);
    expect(screen.getByRole("alert")).toHaveTextContent(
      /Couldn’t load troubleshoot entries/,
    );
  });
});
