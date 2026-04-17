"use client";

/**
 * Full Diagnostics Panel — the expanded counterpart to
 * DiagnosticsStrip. Shows one row per check with status icon,
 * human-readable message, and an optional "Go to fix" deep link.
 *
 * Rendered inside the Troubleshoot page (spec §5 Diagnostics Panel).
 */

import Link from "next/link";
import {
  AlertTriangle,
  CheckCircle2,
  ExternalLink,
  XCircle,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import type {
  DiagnosticCheck,
  DiagnosticStatus,
  DiagnosticsResponse,
} from "@/lib/api";

const STATUS_CONFIG: Record<
  DiagnosticStatus,
  { icon: LucideIcon; color: string; label: string }
> = {
  ok: {
    icon: CheckCircle2,
    color: "text-emerald-600 dark:text-emerald-400",
    label: "OK",
  },
  warn: {
    icon: AlertTriangle,
    color: "text-amber-600 dark:text-amber-400",
    label: "Warning",
  },
  error: {
    icon: XCircle,
    color: "text-red-600 dark:text-red-400",
    label: "Error",
  },
};

const ROW_LABEL_AND_CTA: Array<{
  key: keyof DiagnosticsResponse;
  label: string;
  ctaLabel: string;
}> = [
  { key: "backend", label: "Backend", ctaLabel: "View backend" },
  { key: "provider", label: "Provider", ctaLabel: "Go to Settings" },
  { key: "judge", label: "Judge", ctaLabel: "Open judge setup" },
  { key: "storage", label: "Storage", ctaLabel: "View storage" },
];

interface DiagnosticsPanelProps {
  data: DiagnosticsResponse | null;
  className?: string;
}

export function DiagnosticsPanel({ data, className }: DiagnosticsPanelProps) {
  return (
    <section
      role="region"
      aria-label="Diagnostics"
      className={cn(
        "rounded-lg border bg-card p-4 shadow-sm",
        className,
      )}
    >
      <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
        Diagnostics
      </h3>
      <ul className="mt-3 flex flex-col divide-y">
        {ROW_LABEL_AND_CTA.map(({ key, label, ctaLabel }) => (
          <DiagnosticsRow
            key={key}
            label={label}
            ctaLabel={ctaLabel}
            check={data?.[key] ?? null}
          />
        ))}
      </ul>
    </section>
  );
}

interface DiagnosticsRowProps {
  label: string;
  ctaLabel: string;
  check: DiagnosticCheck | null;
}

function DiagnosticsRow({ label, ctaLabel, check }: DiagnosticsRowProps) {
  if (!check) {
    return (
      <li
        aria-label={`${label}: loading`}
        className="flex items-center gap-3 py-2 text-sm text-muted-foreground"
      >
        <span className="inline-block h-3 w-3 animate-pulse rounded-full bg-muted-foreground/40" />
        <span className="font-medium text-foreground">{label}</span>
        <span>Loading…</span>
      </li>
    );
  }

  const { icon: Icon, color, label: statusLabel } = STATUS_CONFIG[check.status];
  return (
    <li
      aria-label={`${label}: ${statusLabel} — ${check.message}`}
      className="flex items-center justify-between gap-3 py-2 text-sm"
    >
      <div className="flex min-w-0 items-center gap-2">
        <Icon aria-hidden="true" className={cn("h-4 w-4 shrink-0", color)} />
        <span className="font-medium">{label}</span>
        <span className="truncate text-muted-foreground">{check.message}</span>
      </div>
      {check.fixUrl ? (
        <Link
          href={check.fixUrl}
          className="inline-flex shrink-0 items-center gap-1 text-xs font-medium text-primary hover:underline"
        >
          {ctaLabel}
          <ExternalLink className="h-3 w-3" aria-hidden="true" />
        </Link>
      ) : null}
    </li>
  );
}
