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
import { DiagnosticsStrip } from "@/components/learn/diagnostics-strip";
import { MarkdownDoc } from "@/components/learn/markdown-doc";
import { ProviderCard } from "@/components/learn/provider-card";
import {
  COOKBOOK_PROVIDERS,
  PROVIDER_GROUPS_IN_ORDER,
  docSlugForProvider,
  getProviderById,
} from "@/components/learn/provider-catalog";
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

  test("Open in Settings link preserves the provider id", () => {
    mockUseParams.mockReturnValue({ provider: "dgx-spark" });
    mockUseSWR.mockReturnValue({ data: sampleDoc } as ReturnType<typeof useSWR>);
    render(<ProviderGuidePage />);
    const link = screen.getByRole("link", {
      name: "Open DGX Spark in Settings",
    });
    expect(link).toHaveAttribute("href", "/settings?provider=dgx-spark");
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
