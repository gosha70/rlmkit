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
 * Sized as primary actions — lg variant plus a text-base override
 * so the strip reads as the main affordance on the Concepts page,
 * not a compact toolbar.
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

// Shared class override for every control button in this strip.
// shadcn's `lg` size is h-11 px-8; we also bump the text from the
// baseline text-sm to text-base and give the icons more breathing
// room. Applied via cn() so every button shares one source of truth.
const CONTROL_BUTTON_CLASSES = "h-12 px-6 text-base";
const CONTROL_ICON_CLASSES = "mr-2 h-5 w-5";

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
          size="lg"
          onClick={pause}
          aria-label="Pause replay"
          className={CONTROL_BUTTON_CLASSES}
        >
          <Pause className={CONTROL_ICON_CLASSES} aria-hidden="true" />
          Pause
        </Button>
      ) : (
        <Button
          type="button"
          variant="default"
          size="lg"
          onClick={play}
          aria-label={isAtEnd ? "Replay from start" : "Play replay"}
          className={CONTROL_BUTTON_CLASSES}
        >
          <Play className={CONTROL_ICON_CLASSES} aria-hidden="true" />
          {isAtEnd ? "Replay" : "Play"}
        </Button>
      )}

      <Button
        type="button"
        variant="outline"
        size="lg"
        onClick={step}
        disabled={isAtEnd}
        aria-label="Step forward"
        className={CONTROL_BUTTON_CLASSES}
      >
        <SkipForward className={CONTROL_ICON_CLASSES} aria-hidden="true" />
        Step
      </Button>

      <Button
        type="button"
        variant="outline"
        size="lg"
        onClick={reset}
        aria-label="Reset replay"
        className={CONTROL_BUTTON_CLASSES}
      >
        <RotateCcw className={CONTROL_ICON_CLASSES} aria-hidden="true" />
        Reset
      </Button>

      <Button
        type="button"
        variant="outline"
        size="lg"
        onClick={() => setSpeed(nextSpeed(speed))}
        aria-label={`Speed: ${speed}x (click to change)`}
        className={CONTROL_BUTTON_CLASSES}
      >
        Speed: {speed}×
      </Button>
    </div>
  );
}
