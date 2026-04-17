"use client";

/**
 * Bottom tray of the Concepts §C walkthrough — expandable Advanced
 * details for the current step. Shows prompt excerpt, generated
 * code, REPL output, and metrics.
 *
 * Starts collapsed on every step change so the "default view is
 * educational" contract (spec §3) holds per-step, not globally.
 * The parent passes a fresh key={step.id} which forces React to
 * remount this component on step change, resetting local state.
 *
 * If the current step has no details and no metrics, the tray
 * renders nothing — no empty toggle, no layout shift.
 */

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { LearnReplayStep } from "@/lib/api";

interface ReplayAdvancedTrayProps {
  step: LearnReplayStep;
  className?: string;
}

export function ReplayAdvancedTray({
  step,
  className,
}: ReplayAdvancedTrayProps) {
  const [open, setOpen] = useState(false);
  const hasDetails =
    !!step.details &&
    (!!step.details.prompt || !!step.details.code || !!step.details.output);
  const hasMetrics = !!step.metrics && Object.keys(step.metrics).length > 0;
  if (!hasDetails && !hasMetrics) return null;

  return (
    <section
      aria-label="Advanced details"
      className={cn(
        "rounded-lg border bg-card shadow-sm",
        className,
      )}
    >
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1 px-4 py-2 text-xs font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
      >
        {open ? (
          <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
        )}
        {open ? "Hide advanced details" : "Show advanced details"}
      </button>

      {open && (
        <div className="flex flex-col gap-3 border-t px-4 py-3 text-xs">
          {step.details?.prompt && (
            <DetailBlock label="Prompt excerpt" body={step.details.prompt} />
          )}
          {step.details?.code && (
            <DetailBlock label="Generated code" body={step.details.code} mono />
          )}
          {step.details?.output && (
            <DetailBlock label="REPL output" body={step.details.output} mono />
          )}
          {hasMetrics && step.metrics && (
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Metrics
              </p>
              <dl className="mt-1 grid grid-cols-[max-content_1fr] gap-x-3 gap-y-0.5 text-xs">
                {step.metrics.tokensIn !== undefined && (
                  <MetricRow
                    label="Tokens in"
                    value={String(step.metrics.tokensIn)}
                  />
                )}
                {step.metrics.tokensOut !== undefined && (
                  <MetricRow
                    label="Tokens out"
                    value={String(step.metrics.tokensOut)}
                  />
                )}
                {step.metrics.latencyMs !== undefined && (
                  <MetricRow
                    label="Latency"
                    value={`${step.metrics.latencyMs} ms`}
                  />
                )}
                {step.metrics.costUsd !== undefined && (
                  <MetricRow
                    label="Cost"
                    value={`$${step.metrics.costUsd.toFixed(4)}`}
                  />
                )}
              </dl>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

interface DetailBlockProps {
  label: string;
  body: string;
  mono?: boolean;
}

function DetailBlock({ label, body, mono }: DetailBlockProps) {
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {label}
      </p>
      <pre
        className={cn(
          "mt-1 max-h-48 overflow-auto rounded-md bg-muted p-2 text-foreground whitespace-pre-wrap break-words",
          mono ? "font-mono text-[11px]" : "text-xs",
        )}
      >
        {body}
      </pre>
    </div>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="text-muted-foreground">{label}:</dt>
      <dd className="font-mono text-foreground">{value}</dd>
    </>
  );
}
