"use client";

/**
 * Six-node SVG diagram for the Concepts §C replay walkthrough.
 *
 * Static layout: Query → Plan → Code → Sandbox → Decision → Answer.
 * The active node corresponds to the *kind* of the current replay
 * step (question→Query, plan→Plan, code→Code, result→Sandbox,
 * decision→Decision, answer→Answer). For long replays where multiple
 * consecutive steps share a kind, the same node stays highlighted —
 * which is semantically correct.
 *
 * No motion tween. Per the NEXT.md §1 speed contract, the autoplay
 * speed control governs step interval only; the diagram swaps the
 * active fill on render with no transition. That keeps the visual
 * stable at high speeds and the state-machine tests time-independent.
 */

import { cn } from "@/lib/utils";
import type { LearnReplayStepKind } from "@/lib/api";

interface ReplayDiagramProps {
  activeKind: LearnReplayStepKind;
  className?: string;
}

interface DiagramNode {
  kind: LearnReplayStepKind;
  label: string;
}

// Spec §3 Replay diagram — node order is fixed.
const NODES: ReadonlyArray<DiagramNode> = [
  { kind: "question", label: "Query" },
  { kind: "plan", label: "Plan" },
  { kind: "code", label: "Code" },
  { kind: "result", label: "Sandbox" },
  { kind: "decision", label: "Decision" },
  { kind: "answer", label: "Answer" },
];

const NODE_WIDTH = 120;
const NODE_HEIGHT = 72;
const NODE_GAP = 28;
const NODE_LABEL_FONT_SIZE = 15;
const PADDING_X = 16;
const SVG_WIDTH =
  PADDING_X * 2 + NODES.length * NODE_WIDTH + (NODES.length - 1) * NODE_GAP;
const SVG_HEIGHT = NODE_HEIGHT + 40;

function nodeX(index: number): number {
  return PADDING_X + index * (NODE_WIDTH + NODE_GAP);
}

export function ReplayDiagram({ activeKind, className }: ReplayDiagramProps) {
  const activeIndex = NODES.findIndex((n) => n.kind === activeKind);

  return (
    <div className={cn("w-full overflow-x-auto", className)}>
      <svg
        role="img"
        aria-label={`Replay diagram, active node: ${
          NODES[activeIndex]?.label ?? "unknown"
        }`}
        viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
        className="block w-full max-w-full"
      >
        <defs>
          <marker
            id="replay-arrow-head"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="8"
            markerHeight="8"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" className="fill-muted-foreground" />
          </marker>
        </defs>

        {/* Arrows between consecutive nodes. */}
        {NODES.slice(0, -1).map((_, i) => {
          const startX = nodeX(i) + NODE_WIDTH;
          const endX = nodeX(i + 1);
          const y = SVG_HEIGHT / 2;
          return (
            <line
              key={`arrow-${i}`}
              x1={startX}
              y1={y}
              x2={endX - 2}
              y2={y}
              strokeWidth={1.5}
              className="stroke-muted-foreground/60"
              markerEnd="url(#replay-arrow-head)"
            />
          );
        })}

        {/* Nodes. */}
        {NODES.map((n, i) => {
          const isActive = i === activeIndex;
          const x = nodeX(i);
          const y = (SVG_HEIGHT - NODE_HEIGHT) / 2;
          return (
            <g
              key={n.kind}
              data-kind={n.kind}
              data-active={isActive ? "true" : "false"}
            >
              <rect
                x={x}
                y={y}
                rx={8}
                ry={8}
                width={NODE_WIDTH}
                height={NODE_HEIGHT}
                strokeWidth={isActive ? 2 : 1}
                className={cn(
                  isActive
                    ? "fill-primary/15 stroke-primary"
                    : "fill-card stroke-border",
                )}
              />
              <text
                x={x + NODE_WIDTH / 2}
                y={y + NODE_HEIGHT / 2 + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={NODE_LABEL_FONT_SIZE}
                className={cn(
                  "font-semibold",
                  isActive ? "fill-foreground" : "fill-muted-foreground",
                )}
              >
                {n.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
