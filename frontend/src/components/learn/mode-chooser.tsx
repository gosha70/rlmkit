"use client";

/**
 * Concepts §B mode chooser — replaces the V1 static decision strip.
 *
 * Pure client-side lookup. Selecting a scenario in the dropdown
 * highlights the recommended mode and surfaces a one-line rationale
 * directly under the strip. The dropdown opens with the
 * "Not sure…" scenario selected so the chooser is never an empty
 * prompt and Auto is the safe-default reveal.
 */

import { useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

export type LearnMode = "Direct" | "RLM" | "Compare" | "Auto";

interface ModeRow {
  mode: LearnMode;
  body: string;
  dot: string;
}

const MODE_ROWS: ReadonlyArray<ModeRow> = [
  {
    mode: "Direct",
    body: "small, self-contained input that fits in one prompt.",
    dot: "bg-emerald-500",
  },
  {
    mode: "RLM",
    body:
      "large or complex content; the model inspects and reasons step-by-step through a sandboxed Python REPL.",
    dot: "bg-blue-500",
  },
  {
    mode: "Compare",
    body: "run strategies, providers, or profiles side-by-side for a benchmark.",
    dot: "bg-purple-500",
  },
  {
    mode: "Auto",
    body:
      "Studio picks Direct or RLM for you based on input size. Good default when you're not sure.",
    dot: "bg-slate-500",
  },
];

interface Scenario {
  id: string;
  prompt: string;
  recommendedMode: LearnMode;
  rationale: string;
}

const SCENARIOS: ReadonlyArray<Scenario> = [
  {
    id: "not-sure",
    prompt: "Not sure how large or complex the input is",
    recommendedMode: "Auto",
    rationale:
      "Auto routes short inputs through Direct and longer ones through RLM, so you don't have to predict the shape of the task.",
  },
  {
    id: "short-note",
    prompt: "Summarize a short note",
    recommendedMode: "Direct",
    rationale:
      "Direct is one-shot and cheapest — the right call when the input fits comfortably in a single prompt.",
  },
  {
    id: "long-doc",
    prompt: "Answer questions about a long design doc",
    recommendedMode: "RLM",
    rationale:
      "RLM lets the model inspect the document iteratively instead of trying to fit it all into one prompt.",
  },
  {
    id: "compare-prompts",
    prompt: "Compare prompt strategies",
    recommendedMode: "Compare",
    rationale:
      "Compare runs the same input through multiple strategies, providers, or profiles side-by-side so you can pick by evidence.",
  },
];

const SCENARIO_BY_ID: ReadonlyMap<string, Scenario> = new Map(
  SCENARIOS.map((s) => [s.id, s] as const),
);

export function getModeForScenario(scenarioId: string): LearnMode | undefined {
  return SCENARIO_BY_ID.get(scenarioId)?.recommendedMode;
}

export function ModeChooser({ className }: { className?: string }) {
  const [scenarioId, setScenarioId] = useState<string>(SCENARIOS[0].id);
  const scenario = SCENARIO_BY_ID.get(scenarioId);
  const recommendedMode = scenario?.recommendedMode;

  return (
    <div className={cn("rounded-lg border bg-card p-4 shadow-sm", className)}>
      <div className="flex flex-col gap-2">
        <label
          htmlFor="mode-chooser-scenario"
          className="text-sm font-medium"
        >
          What does your task look like?
        </label>
        <Select value={scenarioId} onValueChange={setScenarioId}>
          <SelectTrigger
            id="mode-chooser-scenario"
            aria-label="Pick a scenario"
            className="w-full"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SCENARIOS.map((s) => (
              <SelectItem key={s.id} value={s.id}>
                {s.prompt}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <ul
        aria-label="Mode options"
        className="mt-4 flex flex-col gap-2 text-sm"
      >
        {MODE_ROWS.map((row) => {
          const isActive = recommendedMode === row.mode;
          return (
            <li
              key={row.mode}
              data-mode={row.mode}
              data-active={isActive ? "true" : "false"}
              aria-current={isActive ? "true" : undefined}
              className={cn(
                "flex items-start gap-3 rounded-md border px-3 py-2 transition-colors",
                isActive
                  ? "border-primary bg-primary/5"
                  : "border-transparent",
              )}
            >
              <span
                aria-hidden="true"
                className={cn(
                  "mt-1 inline-block h-2 w-2 shrink-0 rounded-full",
                  row.dot,
                )}
              />
              <span>
                <span className="font-semibold">{row.mode}</span> — {row.body}
              </span>
            </li>
          );
        })}
      </ul>

      {scenario && (
        <p
          aria-live="polite"
          className="mt-3 text-sm text-muted-foreground"
        >
          <span className="font-semibold text-foreground">
            Recommendation: {scenario.recommendedMode}.
          </span>{" "}
          {scenario.rationale}
        </p>
      )}
    </div>
  );
}
