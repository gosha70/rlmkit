"use client";

/**
 * Concepts §C — replay walkthrough composite.
 *
 * Layout matches the shaped three-pane + bottom-tray spec:
 *
 *   header   : title + description + optional truncation note
 *   controls : Play / Pause / Step / Reset / Speed strip
 *   grid     : left rail (step list)
 *              │ center (SVG diagram)
 *              │ right pane (title + summary)
 *   tray     : bottom Advanced details (expandable)
 *
 * On narrow screens the three panes stack vertically (list → diagram
 * → explanation). At md+ they form a 3-column grid: 12rem rail /
 * flexible diagram center / 20rem right pane. The diagram's internal
 * SVG is already overflow-x-auto so shrinking the center cell is safe.
 *
 * The Advanced tray carries `key={step.id}` so React remounts it on
 * every step change — that resets the tray to collapsed per the
 * spec's "default view is educational" contract (if a user opens the
 * tray on step 1, navigating to step 3 starts collapsed, not open).
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

      <div className="grid grid-cols-1 gap-4 md:grid-cols-[12rem_minmax(0,1fr)_20rem]">
        <ReplayStepList
          steps={replay.steps}
          currentStep={controls.currentStep}
          onSelect={controls.goTo}
        />
        <div className="flex min-w-0 items-center">
          <ReplayDiagram activeKind={currentStep.kind} />
        </div>
        <ReplayStepDetail step={currentStep} />
      </div>

      {/* key={step.id} remounts the tray on step change so it
          collapses back to the educational default, per spec §3. */}
      <ReplayAdvancedTray key={currentStep.id} step={currentStep} />
    </section>
  );
}
