"use client";

/**
 * Right pane of the Concepts §C walkthrough — explanation of the
 * currently selected step plus a collapsible "Advanced details"
 * tray with prompt excerpt, generated code, REPL output, and
 * metrics.
 *
 * Keeps default view educational: only title + summary visible
 * unless the user opens the tray.
 */

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { LearnReplayStep } from "@/lib/api";

interface ReplayStepDetailProps {
  step: LearnReplayStep;
  className?: string;
}

export function ReplayStepDetail({ step, className }: ReplayStepDetailProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const hasDetails =
    !!step.details &&
    (!!step.details.prompt || !!step.details.code || !!step.details.output);
  const hasMetrics = !!step.metrics && Object.keys(step.metrics).length > 0;

  return (
    <article
      aria-label={`Step detail: ${step.title}`}
      className={cn(
        "flex flex-col gap-3 rounded-lg border bg-card p-4 shadow-sm",
        className,
      )}
    >
      <header>
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {step.kind}
        </p>
        <h4 className="mt-1 text-base font-semibold leading-tight">
          {step.title}
        </h4>
        <p className="mt-2 text-sm text-foreground/90">{step.summary}</p>
      </header>

      {(hasDetails || hasMetrics) && (
        <div>
          <button
            type="button"
            aria-expanded={showAdvanced}
            onClick={() => setShowAdvanced((v) => !v)}
            className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
          >
            {showAdvanced ? (
              <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
            )}
            {showAdvanced ? "Hide advanced details" : "Show advanced details"}
          </button>

          {showAdvanced && (
            <div className="mt-3 flex flex-col gap-3 border-t pt-3 text-xs">
              {step.details?.prompt && (
                <DetailBlock label="Prompt excerpt" body={step.details.prompt} />
              )}
              {step.details?.code && (
                <DetailBlock
                  label="Generated code"
                  body={step.details.code}
                  mono
                />
              )}
              {step.details?.output && (
                <DetailBlock
                  label="REPL output"
                  body={step.details.output}
                  mono
                />
              )}
              {hasMetrics && step.metrics && (
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Metrics
                  </p>
                  <dl className="mt-1 grid grid-cols-[max-content_1fr] gap-x-3 gap-y-0.5 text-xs">
                    {step.metrics.tokensIn !== undefined && (
                      <MetricRow label="Tokens in" value={String(step.metrics.tokensIn)} />
                    )}
                    {step.metrics.tokensOut !== undefined && (
                      <MetricRow label="Tokens out" value={String(step.metrics.tokensOut)} />
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
        </div>
      )}
    </article>
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
