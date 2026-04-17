"use client";

import useSWR from "swr";
import { AppShell } from "@/components/shared/app-shell";
import { DiagnosticsStrip } from "@/components/learn/diagnostics-strip";
import { ProviderCard } from "@/components/learn/provider-card";
import {
  COOKBOOK_PROVIDERS,
  PROVIDER_GROUPS_IN_ORDER,
  type CookbookProvider,
  type ProviderGroup,
} from "@/components/learn/provider-catalog";
import { getDiagnostics } from "@/lib/api";

export default function CookbookPage() {
  const { data } = useSWR("learn-diagnostics", getDiagnostics, {
    refreshInterval: 30_000,
    dedupingInterval: 30_000,
    errorRetryCount: 2,
  });

  const providersByGroup = groupProviders(COOKBOOK_PROVIDERS);

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl px-6 py-8">
        <header className="mb-4">
          <h2 className="text-2xl font-semibold tracking-tight">Cookbook</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Connect a local or cloud model provider.
          </p>
        </header>

        <DiagnosticsStrip data={data ?? null} className="mb-6" />

        <div className="flex flex-col gap-8">
          {PROVIDER_GROUPS_IN_ORDER.map((group) => {
            const providers = providersByGroup[group] ?? [];
            if (providers.length === 0) return null;
            return (
              <section
                key={group}
                aria-labelledby={groupHeadingId(group)}
                className="flex flex-col gap-3"
              >
                <h3
                  id={groupHeadingId(group)}
                  className="text-sm font-semibold uppercase tracking-wide text-muted-foreground"
                >
                  {group}
                </h3>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {providers.map((p) => (
                    <ProviderCard key={p.id} provider={p} />
                  ))}
                </div>
              </section>
            );
          })}
        </div>
      </div>
    </AppShell>
  );
}

function groupProviders(
  providers: ReadonlyArray<CookbookProvider>,
): Record<ProviderGroup, CookbookProvider[]> {
  const result: Record<ProviderGroup, CookbookProvider[]> = {
    "Easy local": [],
    "Advanced local / self-hosted": [],
    Cloud: [],
  };
  for (const p of providers) {
    result[p.group].push(p);
  }
  return result;
}

function groupHeadingId(group: ProviderGroup): string {
  return `cookbook-group-${group.replace(/[^a-z0-9]+/gi, "-").toLowerCase()}`;
}
