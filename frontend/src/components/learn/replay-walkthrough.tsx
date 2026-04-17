"use client";

/**
 * Concepts §C — replay walkthrough composite.
 *
 * Layout:
 *
 *   header   : title + description + optional truncation note
 *   controls : Play / Pause / Step / Reset / Speed strip
 *   diagram  : 6-node SVG (full width of the walkthrough)
 *   grid     : left rail (step list) │ right pane (title + summary)
 *   tray     : bottom Advanced details (expandable)
 *
 * Deliberate trade-off against a strict 3-column "three-pane" grid.
 * At typical desktop widths the Learn surface's middle column
 * compresses a 6-node horizontal diagram to ~480px — far too narrow
 * for readable labels. Giving the diagram its own full-width row
 * keeps it legible without losing the three distinct panes (rail,
 * diagram, explanation) the spec calls for — they're stacked, not
 * columns. The bottom Advanced tray addressed in the V2 P2 review
 * is unchanged: separate region, keyed by step id so it resets on
 * step change per the "default view is educational" contract.
 */

import { useReplayControls } from "./use-replay-controls";
import { ReplayAdvancedTray } from "./replay-advanced-tray";
import { ReplayControls } from "./replay-controls";
import { ReplayDiagram } from "./replay-diagram";
import { ReplayStepList } from "./replay-step-list";
import { ReplayStepDetail } from "./replay-step-detail";
import { cn } from "@/lib/utils";
import type { LearnReplay } from "@/lib/api";

interface ReplayWalkthroughProps {
  replay: LearnReplay;
  className?: string;
}

export function ReplayWalkthrough({
  replay,
  className,
}: ReplayWalkthroughProps) {
  const controls = useReplayControls(replay.steps.length);
  const currentStep = replay.steps[controls.currentStep] ?? replay.steps[0];

  return (
    <section
      aria-label={`Replay walkthrough: ${replay.title}`}
      className={cn("flex flex-col gap-4", className)}
    >
      <header className="flex flex-col gap-1">
        <h4 className="text-base font-semibold leading-tight">{replay.title}</h4>
        <p className="text-sm text-muted-foreground">{replay.description}</p>
        {replay.metadata.truncated && (
          <p
            role="note"
            className="text-xs text-amber-700 dark:text-amber-300"
          >
            Showing {replay.steps.length} of{" "}
            {replay.metadata.originalStepCount ?? replay.steps.length} steps —
            the middle of the run was truncated for display.
          </p>
        )}
      </header>

      <ReplayControls controls={controls} />

      <ReplayDiagram activeKind={currentStep.kind} />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-[14rem_minmax(0,1fr)]">
        <ReplayStepList
          steps={replay.steps}
          currentStep={controls.currentStep}
          onSelect={controls.goTo}
        />
        <ReplayStepDetail step={currentStep} />
      </div>

      {/* key={step.id} remounts the tray on step change so it
          collapses back to the educational default, per spec §3. */}
      <ReplayAdvancedTray key={currentStep.id} step={currentStep} />
    </section>
  );
}
