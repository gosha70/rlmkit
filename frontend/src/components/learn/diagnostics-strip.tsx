"use client";

/**
 * Persistent diagnostics strip — rendered at the top of every Learn page.
 *
 * Shows four compact status cells (backend, provider, judge, storage)
 * backed by GET /api/diagnostics. Cells with a fixUrl are clickable and
 * route to Settings or a Troubleshoot entry. Icon + text carry the
 * status (not colour alone) per §11 Accessibility.
 */

import Link from "next/link";
import { CheckCircle2, AlertTriangle, XCircle } from "lucide-react";
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

const CHECKS: ReadonlyArray<{
  key: keyof DiagnosticsResponse;
  label: string;
}> = [
  { key: "backend", label: "Backend" },
  { key: "provider", label: "Provider" },
  { key: "judge", label: "Judge" },
  { key: "storage", label: "Storage" },
];

interface DiagnosticsStripProps {
  data: DiagnosticsResponse | null;
  className?: string;
}

export function DiagnosticsStrip({ data, className }: DiagnosticsStripProps) {
  return (
    <div
      role="status"
      aria-label="System diagnostics"
      className={cn(
        "flex flex-wrap items-center gap-3 rounded-md border bg-muted/30 px-3 py-2 text-xs",
        className,
      )}
    >
      {CHECKS.map(({ key, label }) => (
        <DiagnosticCell
          key={key}
          label={label}
          check={data?.[key] ?? null}
        />
      ))}
    </div>
  );
}

interface DiagnosticCellProps {
  label: string;
  check: DiagnosticCheck | null;
}

function DiagnosticCell({ label, check }: DiagnosticCellProps) {
  if (!check) {
    return (
      <span
        aria-label={`${label}: loading`}
        className="flex items-center gap-1.5 text-muted-foreground"
      >
        <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-muted-foreground" />
        <span>{label}</span>
      </span>
    );
  }

  const { icon: Icon, color, label: statusLabel } = STATUS_CONFIG[check.status];
  const body = (
    <>
      <Icon aria-hidden="true" className={cn("h-3.5 w-3.5", color)} />
      <span className="font-medium">{label}</span>
      <span className={cn("sr-only")}>{`: ${statusLabel}`}</span>
      <span className={cn("hidden sm:inline text-muted-foreground")}>
        {check.message}
      </span>
    </>
  );

  const ariaLabel = `${label}: ${statusLabel} — ${check.message}`;

  if (check.fixUrl) {
    return (
      <Link
        href={check.fixUrl}
        aria-label={ariaLabel}
        className="flex items-center gap-1.5 rounded-sm px-1.5 py-0.5 hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {body}
      </Link>
    );
  }

  return (
    <span aria-label={ariaLabel} className="flex items-center gap-1.5 px-1.5 py-0.5">
      {body}
    </span>
  );
}
