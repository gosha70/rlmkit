"use client";

import { ChartContainer } from "./chart-container";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Scale } from "lucide-react";
import type { JudgeScoreData } from "@/lib/api";
import { useChartTooltipStyle } from "./use-chart-tooltip";

const DIM_LABELS: Record<string, string> = {
  relevance: "Relevance",
  correctness: "Correctness",
  completeness: "Completeness",
  coherence: "Coherence",
  conciseness: "Conciseness",
};

const PROVIDER_COLORS = [
  "#f59e0b", // amber-400
  "#3b82f6", // blue-500
  "#10b981", // emerald-500
  "#8b5cf6", // violet-500
  "#ef4444", // red-500
];

interface JudgeDimensionsChartProps {
  judgeScores: JudgeScoreData[];
  /** Map of chat_provider_id → display name */
  providerNames: Record<string, string>;
}

export function JudgeDimensionsChart({ judgeScores, providerNames }: JudgeDimensionsChartProps) {
  const tooltipStyle = useChartTooltipStyle();
  if (judgeScores.length === 0) return null;

  // Collect all providers and dimension names from the data
  const providerIds = [...new Set(judgeScores.map((s) => s.chat_provider_id))];
  const allDims = [
    ...new Set(judgeScores.flatMap((s) => Object.keys(s.dimensions))),
  ].sort();

  // Average per provider per dimension
  const avgByProviderDim: Record<string, Record<string, number>> = {};
  for (const pid of providerIds) {
    const provScores = judgeScores.filter((s) => s.chat_provider_id === pid);
    avgByProviderDim[pid] = {};
    for (const dim of allDims) {
      const vals = provScores.map((s) => s.dimensions[dim]).filter((v) => v != null);
      avgByProviderDim[pid][dim] = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    }
  }

  // Build recharts data: one entry per dimension
  const chartData = allDims.map((dim) => {
    const entry: Record<string, string | number> = {
      dim: DIM_LABELS[dim] ?? dim,
    };
    for (const pid of providerIds) {
      entry[providerNames[pid] ?? pid] = Number((avgByProviderDim[pid][dim] ?? 0).toFixed(2));
    }
    return entry;
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Scale className="h-4 w-4" />
          Judge Dimensions (avg per provider)
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer className="h-56" role="img" aria-label="Grouped bar chart of judge dimension scores per provider">
          {(w, h) => (
            <BarChart width={w} height={h} data={chartData} barCategoryGap="25%" barGap={2}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
              <XAxis
                dataKey="dim"
                tick={{ fill: "var(--color-muted-foreground)", fontSize: 12 }}
              />
              <YAxis
                domain={[0, 5]}
                ticks={[1, 2, 3, 4, 5]}
                tick={{ fill: "var(--color-muted-foreground)", fontSize: 12 }}
              />
              <ReferenceLine y={3} stroke="var(--color-muted-foreground)" strokeDasharray="4 4" strokeOpacity={0.5} />
              <Tooltip
                contentStyle={tooltipStyle}
                formatter={(v) => (typeof v === "number" ? v.toFixed(2) : v)}
              />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              {providerIds.map((pid, i) => (
                <Bar
                  key={pid}
                  dataKey={providerNames[pid] ?? pid}
                  fill={PROVIDER_COLORS[i % PROVIDER_COLORS.length]}
                  radius={[3, 3, 0, 0]}
                />
              ))}
            </BarChart>
          )}
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
