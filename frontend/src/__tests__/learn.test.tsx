/**
 * Learn tab smoke tests.
 *
 * Covers the top-level /learn landing page shell. Landing cards,
 * diagnostics strip, and sub-routes land in later steps and get
 * their own tests then.
 */

import { render, screen } from "@testing-library/react";
import { describe, test, expect, vi } from "vitest";

vi.mock("@/components/shared/app-shell", () => ({
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  AppShell: ({ children }: { children: any }) => children,
}));

import LearnPage from "@/app/learn/page";

describe("LearnPage", () => {
  test("renders title and subtitle", () => {
    render(<LearnPage />);
    expect(
      screen.getByRole("heading", { level: 1, name: "Learn" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        /Understand RLM Studio, set up providers, and troubleshoot common issues\./,
      ),
    ).toBeInTheDocument();
  });

  test("renders labeled landing cards container", () => {
    render(<LearnPage />);
    expect(screen.getByLabelText("Learn landing cards")).toBeInTheDocument();
  });
});
