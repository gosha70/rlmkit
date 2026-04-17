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
