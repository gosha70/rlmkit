"use client";

import useSWR from "swr";
import { AppShell } from "@/components/shared/app-shell";
import { DiagnosticsStrip } from "@/components/learn/diagnostics-strip";
import { getDiagnostics } from "@/lib/api";

export default function LearnPage() {
  const { data } = useSWR("learn-diagnostics", getDiagnostics, {
    refreshInterval: 30_000,
    dedupingInterval: 30_000,
    errorRetryCount: 2,
  });

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl px-6 py-8">
        <header className="mb-4">
          <h2 className="text-2xl font-semibold tracking-tight">Learn</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Understand RLM Studio, set up providers, and troubleshoot common issues.
          </p>
        </header>

        <DiagnosticsStrip data={data ?? null} className="mb-6" />

        <section
          className="grid grid-cols-1 gap-4 sm:grid-cols-2"
          aria-label="Learn landing cards"
        />
      </div>
    </AppShell>
  );
}
