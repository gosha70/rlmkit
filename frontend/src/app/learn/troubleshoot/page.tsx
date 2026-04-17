"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import { AppShell } from "@/components/shared/app-shell";
import { BackToLearn } from "@/components/learn/back-to-learn";
import { DiagnosticsStrip } from "@/components/learn/diagnostics-strip";
import { DiagnosticsPanel } from "@/components/learn/diagnostics-panel";
import { TroubleshootEntry } from "@/components/learn/troubleshoot-entry";
import { TroubleshootSearch } from "@/components/learn/troubleshoot-search";
import { filterTroubleshootEntries } from "@/components/learn/troubleshoot-filter";
import {
  getDiagnostics,
  getTroubleshoot,
  type TroubleshootCategory,
  type TroubleshootResponse,
} from "@/lib/api";

export default function TroubleshootPage() {
  const { data: diagnostics } = useSWR("learn-diagnostics", getDiagnostics, {
    refreshInterval: 30_000,
    dedupingInterval: 30_000,
    errorRetryCount: 2,
  });
  const { data: troubleshoot, error: troubleshootError } =
    useSWR<TroubleshootResponse>("learn-troubleshoot", getTroubleshoot, {
      revalidateOnFocus: false,
    });

  const [query, setQuery] = useState("");
  const [categories, setCategories] = useState<ReadonlySet<TroubleshootCategory>>(
    () => new Set<TroubleshootCategory>(),
  );

  const visibleEntries = useMemo(() => {
    if (!troubleshoot) return [];
    return filterTroubleshootEntries(troubleshoot.entries, {
      query,
      categories,
    });
  }, [troubleshoot, query, categories]);

  const toggleCategory = (cat: TroubleshootCategory) => {
    setCategories((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  };

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl px-6 py-8">
        <BackToLearn className="mb-4" />
        <header className="mb-4">
          <h2 className="text-2xl font-semibold tracking-tight">
            Troubleshoot
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Search known issues, check diagnostics, jump to a fix.
          </p>
        </header>

        <DiagnosticsStrip data={diagnostics ?? null} className="mb-6" />

        <TroubleshootSearch
          query={query}
          onQueryChange={setQuery}
          categories={categories}
          onToggleCategory={toggleCategory}
          className="mb-6"
        />

        <DiagnosticsPanel data={diagnostics ?? null} className="mb-6" />

        <section aria-label="Troubleshoot entries" className="flex flex-col gap-3">
          {troubleshootError ? (
            <div
              role="alert"
              className="rounded-md border border-destructive/40 bg-destructive/5 px-4 py-3 text-sm text-destructive"
            >
              Couldn’t load troubleshoot entries.
            </div>
          ) : !troubleshoot ? (
            <p className="text-sm text-muted-foreground">Loading entries…</p>
          ) : visibleEntries.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No entries match this filter.
            </p>
          ) : (
            visibleEntries.map((e) => <TroubleshootEntry key={e.id} entry={e} />)
          )}
        </section>
      </div>
    </AppShell>
  );
}
