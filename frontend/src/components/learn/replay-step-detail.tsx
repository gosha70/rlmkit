"use client";

/**
 * Right pane of the Concepts §C walkthrough — title / kind / summary
 * only. Stateless.
 *
 * V2-step-4 had this component own an embedded Advanced-details
 * toggle; the shaped spec puts those details in a separate bottom
 * tray that spans all three panes. The tray component is
 * ReplayAdvancedTray; keeping the two responsibilities split lets
 * the tray own its own collapsed state (keyed by step id, so it
 * resets on step change per spec §3's "default view is educational"
 * contract) without coupling it to the explanation pane.
 */

import { cn } from "@/lib/utils";
import type { LearnReplayStep } from "@/lib/api";

interface ReplayStepDetailProps {
  step: LearnReplayStep;
  className?: string;
}

export function ReplayStepDetail({ step, className }: ReplayStepDetailProps) {
  return (
    <article
      aria-label={`Step explanation: ${step.title}`}
      className={cn(
        "flex flex-col gap-2 rounded-lg border bg-card p-4 shadow-sm",
        className,
      )}
    >
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {step.kind}
      </p>
      <h4 className="text-base font-semibold leading-tight">{step.title}</h4>
      <p className="text-sm text-foreground/90">{step.summary}</p>
    </article>
  );
}
