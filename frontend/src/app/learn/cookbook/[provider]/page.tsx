"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import useSWR from "swr";
import { AppShell } from "@/components/shared/app-shell";
import { DiagnosticsStrip } from "@/components/learn/diagnostics-strip";
import { ProviderGuide } from "@/components/learn/provider-guide";
import { getProviderById } from "@/components/learn/provider-catalog";
import { getDiagnostics } from "@/lib/api";

export default function ProviderGuidePage() {
  const params = useParams<{ provider: string }>();
  const providerId = Array.isArray(params?.provider)
    ? params.provider[0]
    : params?.provider ?? "";
  const provider = getProviderById(providerId);

  const { data: diagnostics } = useSWR("learn-diagnostics", getDiagnostics, {
    refreshInterval: 30_000,
    dedupingInterval: 30_000,
    errorRetryCount: 2,
  });

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl px-6 py-8">
        <Link
          href="/learn/cookbook"
          className="mb-4 inline-block text-sm text-muted-foreground hover:text-foreground"
        >
          ← Back to Cookbook
        </Link>

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
