"use client";

/**
 * Replay state machine for the Concepts §C walkthrough widget.
 *
 * Owns four pieces of state: currentStep (index), isPlaying flag,
 * speed multiplier, and the autoplay timer. Exposes Play / Pause /
 * Step / Reset / setSpeed. Bounds: never advance past the final
 * step (autoplay auto-pauses at the end), never step back below 0.
 *
 * Speed semantics (NEXT.md §1 open question, pinned here): speed
 * controls the autoplay step interval only, not any motion-tween
 * the SVG diagram chooses to apply. The hook does not animate.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

export type ReplaySpeed = 1 | 1.5 | 2;

export const REPLAY_SPEEDS: ReadonlyArray<ReplaySpeed> = [1, 1.5, 2];

/** Base autoplay interval in ms at 1× speed. */
export const REPLAY_BASE_INTERVAL_MS = 2000;

export interface ReplayControlsState {
  currentStep: number;
  totalSteps: number;
  isPlaying: boolean;
  speed: ReplaySpeed;
  isAtStart: boolean;
  isAtEnd: boolean;
}

export interface ReplayControls extends ReplayControlsState {
  play: () => void;
  pause: () => void;
  step: () => void;
  stepBack: () => void;
  reset: () => void;
  goTo: (index: number) => void;
  setSpeed: (speed: ReplaySpeed) => void;
}

interface UseReplayControlsOptions {
  /** Override the base interval; useful in tests. */
  baseIntervalMs?: number;
}

export function useReplayControls(
  totalSteps: number,
  options: UseReplayControlsOptions = {},
): ReplayControls {
  const baseIntervalMs = options.baseIntervalMs ?? REPLAY_BASE_INTERVAL_MS;
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeedState] = useState<ReplaySpeed>(1);

  const lastIndex = Math.max(0, totalSteps - 1);

  const step = useCallback(() => {
    setCurrentStep((s) => Math.min(s + 1, lastIndex));
  }, [lastIndex]);

  const stepBack = useCallback(() => {
    setCurrentStep((s) => Math.max(s - 1, 0));
  }, []);

  const play = useCallback(() => {
    // Restart from 0 if we're at the end so Play after auto-pause
    // is intuitive (otherwise user clicks Play and nothing happens).
    setCurrentStep((s) => (s >= lastIndex ? 0 : s));
    setIsPlaying(true);
  }, [lastIndex]);

  const pause = useCallback(() => setIsPlaying(false), []);

  const reset = useCallback(() => {
    setIsPlaying(false);
    setCurrentStep(0);
  }, []);

  const goTo = useCallback(
    (index: number) => {
      setIsPlaying(false);
      setCurrentStep(Math.max(0, Math.min(index, lastIndex)));
    },
    [lastIndex],
  );

  const setSpeed = useCallback((next: ReplaySpeed) => setSpeedState(next), []);

  // Autoplay tick. Each step schedules a single setTimeout, so the
  // effect re-binds on every advance — no stale-ref hazard, no
  // accumulating interval. State updates only happen inside the
  // async setTimeout callback (not synchronously in the effect
  // body), which keeps the React 19 set-state-in-effect lint happy.
  useEffect(() => {
    if (!isPlaying || currentStep >= lastIndex) return;
    const intervalMs = baseIntervalMs / speed;
    const id = setTimeout(() => {
      const next = Math.min(currentStep + 1, lastIndex);
      setCurrentStep(next);
      // Auto-pause when we land on the final step so the user
      // returns to a clean paused state at the end.
      if (next >= lastIndex) {
        setIsPlaying(false);
      }
    }, intervalMs);
    return () => clearTimeout(id);
  }, [isPlaying, speed, baseIntervalMs, lastIndex, currentStep]);

  return useMemo<ReplayControls>(
    () => ({
      currentStep,
      totalSteps,
      isPlaying,
      speed,
      isAtStart: currentStep === 0,
      isAtEnd: currentStep === lastIndex,
      play,
      pause,
      step,
      stepBack,
      reset,
      goTo,
      setSpeed,
    }),
    [
      currentStep,
      totalSteps,
      isPlaying,
      speed,
      lastIndex,
      play,
      pause,
      step,
      stepBack,
      reset,
      goTo,
      setSpeed,
    ],
  );
}
