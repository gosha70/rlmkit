/**
 * Learn tab — replay walkthrough tests (V2 §C).
 *
 * Lives in its own file rather than extending learn.test.tsx because
 * V2 introduces a separate state-machine surface (controls hook,
 * SVG diagram, three-pane layout) and bundled-asset assertions that
 * deserve their own home.
 */

import { describe, test, expect } from "vitest";

import bundledReplay from "../../public/learn/replays/bundled-rlm-demo.json";
import type { LearnReplay, LearnReplayStepKind } from "@/lib/api";

describe("bundled replay JSON", () => {
  const replay = bundledReplay as unknown as LearnReplay;

  test("matches the LearnReplay shape at top level", () => {
    expect(typeof replay.id).toBe("string");
    expect(typeof replay.title).toBe("string");
    expect(typeof replay.description).toBe("string");
    expect(Array.isArray(replay.steps)).toBe(true);
    expect(replay.metadata.source).toBe("bundled");
    expect(typeof replay.metadata.convertorVersion).toBe("number");
  });

  test("opens with question and ends with answer", () => {
    expect(replay.steps[0].kind).toBe("question");
    expect(replay.steps[replay.steps.length - 1].kind).toBe("answer");
  });

  test("every step kind is in the discriminator union", () => {
    const allowed = new Set<LearnReplayStepKind>([
      "question",
      "plan",
      "code",
      "result",
      "decision",
      "answer",
    ]);
    for (const step of replay.steps) {
      expect(allowed.has(step.kind)).toBe(true);
    }
  });

  test("every code step is followed by a result step (no orphans)", () => {
    // Spec §3 invariant — same rule the V2b trace converter will
    // enforce under truncation. Bundled replays must satisfy it
    // natively or the truncation contract drifts.
    for (let i = 0; i < replay.steps.length; i++) {
      if (replay.steps[i].kind === "code") {
        expect(replay.steps[i + 1]?.kind).toBe("result");
      }
      if (replay.steps[i].kind === "result") {
        expect(replay.steps[i - 1]?.kind).toBe("code");
      }
    }
  });

  test("every step has a unique id", () => {
    const ids = replay.steps.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

// ---------------------------------------------------------------------------
// useReplayControls — replay state machine (V2 §C controls)
// ---------------------------------------------------------------------------

import { act, renderHook } from "@testing-library/react";
import {
  REPLAY_BASE_INTERVAL_MS,
  REPLAY_SPEEDS,
  useReplayControls,
} from "@/components/learn/use-replay-controls";
import { vi } from "vitest";

describe("useReplayControls", () => {
  test("starts paused at step 0", () => {
    const { result } = renderHook(() => useReplayControls(6));
    expect(result.current.currentStep).toBe(0);
    expect(result.current.totalSteps).toBe(6);
    expect(result.current.isPlaying).toBe(false);
    expect(result.current.speed).toBe(1);
    expect(result.current.isAtStart).toBe(true);
    expect(result.current.isAtEnd).toBe(false);
  });

  test("step() advances by one and clamps at the final index", () => {
    const { result } = renderHook(() => useReplayControls(3));
    act(() => result.current.step());
    expect(result.current.currentStep).toBe(1);
    act(() => result.current.step());
    expect(result.current.currentStep).toBe(2);
    expect(result.current.isAtEnd).toBe(true);
    act(() => result.current.step()); // no-op past final
    expect(result.current.currentStep).toBe(2);
  });

  test("stepBack() never goes below 0", () => {
    const { result } = renderHook(() => useReplayControls(3));
    act(() => result.current.stepBack());
    expect(result.current.currentStep).toBe(0);
    act(() => result.current.step());
    act(() => result.current.stepBack());
    expect(result.current.currentStep).toBe(0);
  });

  test("play() then pause() toggles isPlaying", () => {
    const { result } = renderHook(() => useReplayControls(3));
    act(() => result.current.play());
    expect(result.current.isPlaying).toBe(true);
    act(() => result.current.pause());
    expect(result.current.isPlaying).toBe(false);
  });

  test("reset() returns to step 0 and pauses", () => {
    const { result } = renderHook(() => useReplayControls(5));
    act(() => result.current.step());
    act(() => result.current.step());
    act(() => result.current.play());
    act(() => result.current.reset());
    expect(result.current.currentStep).toBe(0);
    expect(result.current.isPlaying).toBe(false);
  });

  test("goTo() clamps and pauses", () => {
    const { result } = renderHook(() => useReplayControls(5));
    act(() => result.current.play());
    act(() => result.current.goTo(3));
    expect(result.current.currentStep).toBe(3);
    expect(result.current.isPlaying).toBe(false);
    act(() => result.current.goTo(99));
    expect(result.current.currentStep).toBe(4);
    act(() => result.current.goTo(-2));
    expect(result.current.currentStep).toBe(0);
  });

  test("setSpeed() accepts each declared speed", () => {
    const { result } = renderHook(() => useReplayControls(3));
    for (const s of REPLAY_SPEEDS) {
      act(() => result.current.setSpeed(s));
      expect(result.current.speed).toBe(s);
    }
  });

  test("autoplay advances on the timer at 1× speed", () => {
    vi.useFakeTimers();
    try {
      const { result } = renderHook(() =>
        useReplayControls(4, { baseIntervalMs: 100 }),
      );
      act(() => result.current.play());
      expect(result.current.currentStep).toBe(0);
      act(() => vi.advanceTimersByTime(100));
      expect(result.current.currentStep).toBe(1);
      act(() => vi.advanceTimersByTime(100));
      expect(result.current.currentStep).toBe(2);
    } finally {
      vi.useRealTimers();
    }
  });

  test("autoplay halves the interval at 2× speed", () => {
    vi.useFakeTimers();
    try {
      const { result } = renderHook(() =>
        useReplayControls(4, { baseIntervalMs: 100 }),
      );
      act(() => result.current.setSpeed(2));
      act(() => result.current.play());
      // At 2× the interval is 50ms; one tick advances exactly one step.
      act(() => vi.advanceTimersByTime(50));
      expect(result.current.currentStep).toBe(1);
      act(() => vi.advanceTimersByTime(50));
      expect(result.current.currentStep).toBe(2);
    } finally {
      vi.useRealTimers();
    }
  });

  test("autoplay auto-pauses at the final step", () => {
    vi.useFakeTimers();
    try {
      const { result } = renderHook(() =>
        useReplayControls(3, { baseIntervalMs: 100 }),
      );
      act(() => result.current.play());
      // The autoplay loop schedules a fresh setTimeout per step so the
      // React flush between ticks lands cleanly. Advance a step at a
      // time so the test mirrors that cadence.
      act(() => vi.advanceTimersByTime(100));
      expect(result.current.currentStep).toBe(1);
      act(() => vi.advanceTimersByTime(100));
      expect(result.current.currentStep).toBe(2);
      // After landing on the final step the next effect cycle flips
      // isPlaying to false; flush by advancing zero ms inside act.
      act(() => vi.advanceTimersByTime(0));
      expect(result.current.isPlaying).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  test("play() at the end restarts from step 0", () => {
    const { result } = renderHook(() => useReplayControls(3));
    act(() => result.current.goTo(2));
    expect(result.current.currentStep).toBe(2);
    act(() => result.current.play());
    expect(result.current.currentStep).toBe(0);
    expect(result.current.isPlaying).toBe(true);
  });

  test("REPLAY_BASE_INTERVAL_MS is a positive number", () => {
    // Pinned default; if anyone changes it, the polish that follows
    // (UX feel, autoplay snap) needs a deliberate revisit.
    expect(REPLAY_BASE_INTERVAL_MS).toBeGreaterThan(0);
  });
});
