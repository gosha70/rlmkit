"use client";

/**
 * Confirmation banner for the Cookbook → Settings deep link.
 *
 * Pitch decision #4: safety-oriented copy, Cancel as the safe
 * default, explicit "No API keys or secrets will be filled in
 * automatically." reassurance. The banner is purely presentational —
 * the parent (SettingsPage) owns the query-param state and decides
 * what "Use values" actually applies.
 *
 * See doc_internal/specs/learn-tab/pitch-learn-tab.md §Deep-link
 * security and §Resolved Decisions #4 for the security rationale.
 */

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface DeepLinkValues {
  provider: string;
  baseUrl?: string;
  model?: string;
}

interface DeepLinkBannerProps {
  values: DeepLinkValues;
  providerDisplayName?: string;
  onCancel: () => void;
  onUseValues: () => void;
  className?: string;
}

export function DeepLinkBanner({
  values,
  providerDisplayName,
  onCancel,
  onUseValues,
  className,
}: DeepLinkBannerProps) {
  const rows: Array<[string, string]> = [
    ["Provider", providerDisplayName ?? values.provider],
  ];
  if (values.baseUrl) rows.push(["Base URL", values.baseUrl]);
  if (values.model) rows.push(["Model", values.model]);

  return (
    <section
      role="region"
      aria-labelledby="deep-link-banner-heading"
      className={cn(
        "rounded-lg border bg-card p-4 shadow-sm",
        className,
      )}
    >
      <h3
        id="deep-link-banner-heading"
        className="text-base font-semibold tracking-tight"
      >
        Review provider values from this guide
      </h3>

      <dl className="mt-3 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
        {rows.map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="text-muted-foreground">{label}:</dt>
            <dd className="font-mono text-foreground break-all">{value}</dd>
          </div>
        ))}
      </dl>

      <p className="mt-3 text-xs text-muted-foreground">
        No API keys or secrets will be filled in automatically.
      </p>

      <div className="mt-4 flex items-center justify-end gap-2">
        <Button variant="outline" size="sm" onClick={onCancel}>
          Cancel
        </Button>
        <Button size="sm" onClick={onUseValues}>
          Use values
        </Button>
      </div>
    </section>
  );
}
