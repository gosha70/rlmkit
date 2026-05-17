"use client";

/**
 * Cookbook tab — provider catalog plus per-provider guides.
 *
 * Two views share this route, switched by the ``?provider=`` query
 * param:
 *
 * * Without ``?provider=`` — the catalog landing page that groups all
 *   Cookbook entries by category and renders one ``ProviderCard``
 *   per provider.
 * * With ``?provider=<id>`` — the per-provider guide page. Shows the
 *   ``ProviderGuide`` for the matched provider, or an alert with a
 *   back link when the id is unknown.
 *
 * The page uses a query param (and not a dynamic ``[provider]``
 * segment) so the bundled-UI build (``npm run build:bundle``) can
 * produce a static export. Dynamic-segment routes can't be enumerated
 * at build time and would force a per-route server.
 */

import { Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import useSWR from "swr";

import { AppShell } from "@/components/shared/app-shell";
import { BackToLearn } from "@/components/learn/back-to-learn";
import { DiagnosticsStrip } from "@/components/learn/diagnostics-strip";
import { ProviderCard } from "@/components/learn/provider-card";
import { ProviderGuide } from "@/components/learn/provider-guide";
import {
  COOKBOOK_PROVIDERS,
  PROVIDER_GROUPS_IN_ORDER,
  getProviderById,
  type CookbookProvider,
  type ProviderGroup,
} from "@/components/learn/provider-catalog";
import { getDiagnostics } from "@/lib/api";

function CatalogView() {
  const { data } = useSWR("learn-diagnostics", getDiagnostics, {
    refreshInterval: 30_000,
    dedupingInterval: 30_000,
    errorRetryCount: 2,
  });

  const providersByGroup = groupProviders(COOKBOOK_PROVIDERS);

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl px-6 py-8">
        <BackToLearn className="mb-4" />
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

function GuideView({ providerId }: { providerId: string }) {
  const provider = getProviderById(providerId);

  const { data: diagnostics } = useSWR("learn-diagnostics", getDiagnostics, {
    refreshInterval: 30_000,
    dedupingInterval: 30_000,
    errorRetryCount: 2,
  });

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl px-6 py-8">
        {/* Mini breadcrumb: Learn / Cookbook. "Back to Learn" is the
            primary nav (consistent with other Learn sub-pages); the
            Cookbook step sits next to it for one-click return to the
            provider list. */}
        <div className="mb-4 flex items-center gap-2 text-sm text-muted-foreground">
          <BackToLearn />
          <span aria-hidden="true">/</span>
          <Link
            href="/learn/cookbook"
            className="hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
          >
            Cookbook
          </Link>
        </div>

        <DiagnosticsStrip data={diagnostics ?? null} className="mb-6" />

        {provider ? (
          <ProviderGuide provider={provider} />
        ) : (
          <div
            role="alert"
            className="rounded-md border border-destructive/40 bg-destructive/5 px-4 py-3 text-sm text-destructive"
          >
            <p className="font-medium">Provider not found</p>
            <p className="mt-1 text-destructive/80">
              {providerId
                ? `No Cookbook guide exists for "${providerId}".`
                : "No provider specified."}{" "}
              <Link href="/learn/cookbook" className="underline">
                Back to Cookbook
              </Link>
            </p>
          </div>
        )}
      </div>
    </AppShell>
  );
}

function CookbookPageInner() {
  const searchParams = useSearchParams();
  const providerId = searchParams.get("provider");
  return providerId ? <GuideView providerId={providerId} /> : <CatalogView />;
}

export default function CookbookPage() {
  return (
    <Suspense>
      <CookbookPageInner />
    </Suspense>
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
