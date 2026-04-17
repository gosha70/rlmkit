"use client";

/**
 * Concepts §C — replay walkthrough composite.
 *
 * Layout:
 *
 *   header   : title + description + optional truncation note
 *   controls : Play / Pause / Step / Reset / Speed strip
 *   diagram  : 6-node SVG (full width of the walkthrough)
 *   grid     : left rail (step list) │ right pane (title + summary
 *              + in-pane Advanced details)
 *
 * Advanced details live INSIDE the right pane rather than in a
 * separate bottom tray. The pane had plenty of vertical room on the
 * default step (no details = empty space); folding Advanced back
 * into the same card keeps the reader's focus in one place and
 * removes the redundant second region below the grid.
 *
 * The right pane carries `key={currentStep.id}` which forces React
 * to remount it on step change. That resets its local `showAdvanced`
 * state to collapsed per the spec §3 "default view is educational"
 * contract — switching steps always lands on a clean collapsed
 * default, so the V2 P2 review invariant is preserved.
 */

import { useReplayControls } from "./use-replay-controls";
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
        {/* key={step.id} remounts the detail pane on step change so
            its in-pane Advanced section collapses back to the
            educational default (V2 P2 review invariant). */}
        <ReplayStepDetail key={currentStep.id} step={currentStep} />
      </div>
    </section>
  );
}
