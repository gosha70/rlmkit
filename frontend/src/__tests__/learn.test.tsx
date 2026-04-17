/**
 * Learn tab tests.
 *
 * Covers:
 *   - LearnPage landing shell (heading, diagnostics strip mount, cards region)
 *   - DiagnosticsStrip renderer (ok / warn / error states, fixUrl linking,
 *     loading state, accessibility labels)
 *   - MarkdownDoc loader (loading / error / rendered markdown)
 *   - ProviderCard and CookbookPage (grouping, card rendering, link targets,
 *     diagnostics strip mount on cookbook)
 *
 * Sub-routes for per-provider guides populate the next step and get
 * their own tests then.
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

import LearnPage from "@/app/learn/page";
import CookbookPage from "@/app/learn/cookbook/page";
import { DiagnosticsStrip } from "@/components/learn/diagnostics-strip";
import { MarkdownDoc } from "@/components/learn/markdown-doc";
import { ProviderCard } from "@/components/learn/provider-card";
import {
  COOKBOOK_PROVIDERS,
  PROVIDER_GROUPS_IN_ORDER,
  docSlugForProvider,
  getProviderById,
} from "@/components/learn/provider-catalog";

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
