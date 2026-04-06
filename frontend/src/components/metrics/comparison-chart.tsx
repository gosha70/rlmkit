"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ModeSummary } from "@/lib/api";
import { useChartTooltipStyle } from "./use-chart-tooltip";

interface ComparisonChartProps {
  rlmData: ModeSummary | undefined;
  directData: ModeSummary | undefined;
}

export function ComparisonChart({ rlmData, directData }: ComparisonChartProps) {
  if (!rlmData && !directData) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">RLM vs Direct</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="flex h-48 items-center justify-center text-sm text-muted-foreground">
            Run queries in both RLM and Direct modes to see comparison.
          </p>
        </CardContent>
      </Card>
    );
  }

  const tokenData = [
    { name: "Tokens", RLM: rlmData?.total_tokens ?? 0, Direct: directData?.total_tokens ?? 0 },
  ];
  const costData = [
    { name: "Cost ($)", RLM: rlmData?.total_cost_usd ?? 0, Direct: directData?.total_cost_usd ?? 0 },
  ];

  const tooltipStyle = useChartTooltipStyle();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">RLM vs Direct</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="h-48" role="img" aria-label="Bar chart comparing RLM and Direct mode token usage">
            <ResponsiveContainer width="100%" height="100%" minWidth={0}>
              <BarChart data={tokenData} barCategoryGap="20%">
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis dataKey="name" className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
                <YAxis className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Bar dataKey="RLM" fill="#2563eb" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Direct" fill="#7c3aed" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="h-48" role="img" aria-label="Bar chart comparing RLM and Direct mode cost">
            <ResponsiveContainer width="100%" height="100%" minWidth={0}>
              <BarChart data={costData} barCategoryGap="20%">
                <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                <XAxis dataKey="name" className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
                <YAxis className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => `$${Number(v).toFixed(4)}`} />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Bar dataKey="RLM" fill="#2563eb" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Direct" fill="#7c3aed" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
