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

// ---------------------------------------------------------------------------
// ReplayDiagram — 6-node SVG (V2 §C diagram)
// ---------------------------------------------------------------------------

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ReplayDiagram } from "@/components/learn/replay-diagram";

describe("ReplayDiagram", () => {
  test("renders all six nodes in spec order", () => {
    const { container } = render(<ReplayDiagram activeKind="question" />);
    const groups = container.querySelectorAll("g[data-kind]");
    const kinds = Array.from(groups).map((g) => g.getAttribute("data-kind"));
    expect(kinds).toEqual([
      "question",
      "plan",
      "code",
      "result",
      "decision",
      "answer",
    ]);
  });

  test("highlights the active node and only the active node", () => {
    const { container } = render(<ReplayDiagram activeKind="code" />);
    const active = container.querySelectorAll('g[data-active="true"]');
    expect(active.length).toBe(1);
    expect(active[0].getAttribute("data-kind")).toBe("code");
  });

  test("aria-label names the active node label, not the kind", () => {
    render(<ReplayDiagram activeKind="result" />);
    expect(
      screen.getByRole("img", { name: /active node: Sandbox/ }),
    ).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// ReplayStepList — left rail
// ---------------------------------------------------------------------------

import { ReplayStepList } from "@/components/learn/replay-step-list";
import type { LearnReplayStep } from "@/lib/api";

const SAMPLE_STEPS: LearnReplayStep[] = [
  { id: "q", kind: "question", title: "Question received", summary: "..." },
  { id: "p", kind: "plan", title: "Plan formed", summary: "..." },
  { id: "c", kind: "code", title: "Code generated", summary: "..." },
];

describe("ReplayStepList", () => {
  test("renders one button per step with 1-indexed numbering", () => {
    render(
      <ReplayStepList
        steps={SAMPLE_STEPS}
        currentStep={0}
        onSelect={() => {}}
      />,
    );
    const buttons = screen.getAllByRole("button");
    expect(buttons).toHaveLength(SAMPLE_STEPS.length);
    expect(buttons[0]).toHaveTextContent("Question received");
    expect(buttons[2]).toHaveTextContent("Code generated");
  });

  test("active item carries aria-current=step", () => {
    render(
      <ReplayStepList
        steps={SAMPLE_STEPS}
        currentStep={1}
        onSelect={() => {}}
      />,
    );
    const buttons = screen.getAllByRole("button");
    expect(buttons[1]).toHaveAttribute("aria-current", "step");
    expect(buttons[0]).not.toHaveAttribute("aria-current");
    expect(buttons[2]).not.toHaveAttribute("aria-current");
  });

  test("clicking a step fires onSelect with its index", async () => {
    const onSelect = vi.fn();
    render(
      <ReplayStepList
        steps={SAMPLE_STEPS}
        currentStep={0}
        onSelect={onSelect}
      />,
    );
    await userEvent.setup().click(screen.getAllByRole("button")[2]);
    expect(onSelect).toHaveBeenCalledWith(2);
  });
});

// ---------------------------------------------------------------------------
// ReplayStepDetail — right pane + Advanced details tray
// ---------------------------------------------------------------------------

import { ReplayStepDetail } from "@/components/learn/replay-step-detail";

describe("ReplayStepDetail", () => {
  const richStep: LearnReplayStep = {
    id: "c1",
    kind: "code",
    title: "Code generated",
    summary: "The model writes a small Python action.",
    details: {
      prompt: "PROMPT_EXCERPT_HERE",
      code: "hits = grep(prompt, pattern=\"x\")",
      output: "OUTPUT_HERE",
    },
    metrics: {
      tokensIn: 380,
      tokensOut: 140,
      latencyMs: 1820,
      costUsd: 0.0021,
    },
  };

  test("shows title and summary by default; advanced details hidden", () => {
    render(<ReplayStepDetail step={richStep} />);
    expect(
      screen.getByRole("heading", { level: 4, name: "Code generated" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/The model writes a small Python action/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/PROMPT_EXCERPT_HERE/)).not.toBeInTheDocument();
    expect(screen.queryByText(/OUTPUT_HERE/)).not.toBeInTheDocument();
  });

  test("toggle reveals prompt, code, output, and metrics", async () => {
    render(<ReplayStepDetail step={richStep} />);
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: /Show advanced details/ }));
    expect(screen.getByText("PROMPT_EXCERPT_HERE")).toBeInTheDocument();
    expect(screen.getByText(/grep\(prompt, pattern=/)).toBeInTheDocument();
    expect(screen.getByText("OUTPUT_HERE")).toBeInTheDocument();
    expect(screen.getByText("Tokens in:")).toBeInTheDocument();
    expect(screen.getByText("380")).toBeInTheDocument();
    expect(screen.getByText("Tokens out:")).toBeInTheDocument();
    expect(screen.getByText("140")).toBeInTheDocument();
    expect(screen.getByText("Latency:")).toBeInTheDocument();
    expect(screen.getByText("1820 ms")).toBeInTheDocument();
    expect(screen.getByText("Cost:")).toBeInTheDocument();
    expect(screen.getByText("$0.0021")).toBeInTheDocument();
  });

  test("step without details/metrics omits the tray entirely", () => {
    const bareStep: LearnReplayStep = {
      id: "q",
      kind: "question",
      title: "Question received",
      summary: "...",
    };
    render(<ReplayStepDetail step={bareStep} />);
    expect(
      screen.queryByRole("button", { name: /advanced details/i }),
    ).not.toBeInTheDocument();
  });
});
