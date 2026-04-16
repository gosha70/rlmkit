"use client";

import { AppShell } from "@/components/shared/app-shell";

export default function LearnPage() {
  return (
    <AppShell>
      <div className="mx-auto max-w-5xl px-6 py-8">
        <header className="mb-6">
          <h1 className="text-2xl font-semibold tracking-tight">Learn</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Understand RLM Studio, set up providers, and troubleshoot common issues.
          </p>
        </header>

        <div
          className="grid grid-cols-1 gap-4 sm:grid-cols-2"
          aria-label="Learn landing cards"
        />
      </div>
    </AppShell>
  );
}
