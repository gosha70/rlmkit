/**
 * Learn tab tests.
 *
 * Covers:
 *   - LearnPage landing shell (heading, diagnostics strip mount, cards region)
 *   - DiagnosticsStrip renderer (ok / warn / error states, fixUrl linking,
 *     loading state, accessibility labels)
 *
 * Sub-routes and cards populate later steps and get their own tests then.
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
import { DiagnosticsStrip } from "@/components/learn/diagnostics-strip";

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
