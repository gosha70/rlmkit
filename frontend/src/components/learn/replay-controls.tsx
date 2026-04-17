"use client";

/**
 * Play / Pause / Step / Reset / Speed controls for the Concepts §C
 * walkthrough. Stateless — receives a ReplayControls instance from
 * useReplayControls and dispatches actions back through it.
 *
 * Speed cycles through REPLAY_SPEEDS (1× → 1.5× → 2× → 1×). One
 * button rather than three keeps the strip compact and avoids
 * duplicate aria-pressed bookkeeping.
 *
 * Buttons use the default size (h-10) rather than `sm` so they're
 * comfortable primary actions on the Concepts page, not compact
 * toolbar buttons.
 */

import { Pause, Play, RotateCcw, SkipForward } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  REPLAY_SPEEDS,
  type ReplayControls as ReplayControlsHandle,
  type ReplaySpeed,
} from "./use-replay-controls";

interface ReplayControlsProps {
  controls: ReplayControlsHandle;
  className?: string;
}

function nextSpeed(current: ReplaySpeed): ReplaySpeed {
  const i = REPLAY_SPEEDS.indexOf(current);
  return REPLAY_SPEEDS[(i + 1) % REPLAY_SPEEDS.length];
}

export function ReplayControls({
  controls,
  className,
}: ReplayControlsProps) {
  const { isPlaying, isAtEnd, speed, play, pause, step, reset, setSpeed } =
    controls;

  return (
    <div
      role="group"
      aria-label="Replay controls"
      className={cn("flex flex-wrap items-center gap-2", className)}
    >
      {isPlaying ? (
        <Button
          type="button"
          variant="default"
          onClick={pause}
          aria-label="Pause replay"
        >
          <Pause className="mr-1.5 h-4 w-4" aria-hidden="true" />
          Pause
        </Button>
      ) : (
        <Button
          type="button"
          variant="default"
          onClick={play}
          aria-label={isAtEnd ? "Replay from start" : "Play replay"}
        >
          <Play className="mr-1.5 h-4 w-4" aria-hidden="true" />
          {isAtEnd ? "Replay" : "Play"}
        </Button>
      )}

      <Button
        type="button"
        variant="outline"
        onClick={step}
        disabled={isAtEnd}
        aria-label="Step forward"
      >
        <SkipForward className="mr-1.5 h-4 w-4" aria-hidden="true" />
        Step
      </Button>

      <Button
        type="button"
        variant="outline"
        onClick={reset}
        aria-label="Reset replay"
      >
        <RotateCcw className="mr-1.5 h-4 w-4" aria-hidden="true" />
        Reset
      </Button>

      <Button
        type="button"
        variant="outline"
        onClick={() => setSpeed(nextSpeed(speed))}
        aria-label={`Speed: ${speed}x (click to change)`}
      >
        Speed: {speed}×
      </Button>
    </div>
  );
}
