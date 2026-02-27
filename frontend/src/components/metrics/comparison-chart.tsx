"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ModeSummary } from "@/lib/api";

interface ComparisonChartProps {
  rlmData: ModeSummary | undefined;
  directData: ModeSummary | undefined;
}

export function ComparisonChart({ rlmData, directData }: ComparisonChartProps) {
  const data = [
    {
      name: "Tokens",
      RLM: rlmData?.total_tokens ?? 0,
      Direct: directData?.total_tokens ?? 0,
    },
    {
      name: "Cost ($)",
      RLM: rlmData?.total_cost_usd ?? 0,
      Direct: directData?.total_cost_usd ?? 0,
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">RLM vs Direct</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64" role="img" aria-label="Bar chart comparing RLM and Direct mode token usage and cost">
          <ResponsiveContainer width="100%" height="100%" minWidth={0}>
            <BarChart data={data} barCategoryGap="20%">
              <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
              <XAxis dataKey="name" className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
              <YAxis className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--color-card)",
                  border: "1px solid var(--color-border)",
                  borderRadius: "6px",
                  fontSize: "12px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Bar dataKey="RLM" fill="#2563eb" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Direct" fill="#7c3aed" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
