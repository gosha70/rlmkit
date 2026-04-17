"use client";

/**
 * Numbered step list — left rail of the Concepts §C walkthrough.
 *
 * Click a step to jump (parent typically forwards to controls.goTo,
 * which also pauses autoplay). The list is the canonical accessible
 * navigation for the walkthrough; the SVG diagram is supplementary.
 */

import { cn } from "@/lib/utils";
import type { LearnReplayStep } from "@/lib/api";

interface ReplayStepListProps {
  steps: ReadonlyArray<LearnReplayStep>;
  currentStep: number;
  onSelect: (index: number) => void;
  className?: string;
}

export function ReplayStepList({
  steps,
  currentStep,
  onSelect,
  className,
}: ReplayStepListProps) {
  return (
    <nav aria-label="Replay steps" className={className}>
      <ol className="flex flex-col gap-1 text-sm">
        {steps.map((step, index) => {
          const isActive = index === currentStep;
          return (
            <li key={step.id}>
              <button
                type="button"
                onClick={() => onSelect(index)}
                aria-current={isActive ? "step" : undefined}
                className={cn(
                  "flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  isActive
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                <span
                  aria-hidden="true"
                  className={cn(
                    "mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-xs font-semibold",
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-muted-foreground",
                  )}
                >
                  {index + 1}
                </span>
                <span className="min-w-0">
                  <span className="block font-medium leading-tight">
                    {step.title}
                  </span>
                  <span className="block text-xs uppercase tracking-wide text-muted-foreground/80">
                    {step.kind}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
