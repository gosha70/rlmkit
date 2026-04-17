"use client";

/**
 * Concepts §C — three-pane replay walkthrough composite.
 *
 * Layout:
 *   header   : controls strip
 *   center   : SVG diagram (Query → … → Answer)
 *   below    : grid — left rail (step list) + right pane (step detail)
 *
 * Pure presentation; data comes in as a LearnReplay prop. Owns the
 * replay state machine via useReplayControls. The detail pane's
 * Advanced details tray collapses by default per spec §3.
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

      <div className="grid grid-cols-1 gap-4 md:grid-cols-[14rem_1fr]">
        <ReplayStepList
          steps={replay.steps}
          currentStep={controls.currentStep}
          onSelect={controls.goTo}
        />
        <ReplayStepDetail step={currentStep} />
      </div>
    </section>
  );
}
