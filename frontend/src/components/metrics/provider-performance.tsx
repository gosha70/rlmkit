"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import { ChartContainer } from "./chart-container";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { displayProviderName } from "@/lib/constants";
import { useChartTooltipStyle } from "./use-chart-tooltip";

interface ProviderPerformanceProps {
  data: Record<string, { queries: number; total_tokens: number; total_cost_usd: number; avg_latency_seconds: number }>;
  title?: string;
}

const COLORS = { tokens: "#2563eb", latency: "#d97706" };

export function ProviderPerformance({ data, title = "Provider Performance" }: ProviderPerformanceProps) {
  const chartData = Object.entries(data).map(([name, values]) => ({
    name: displayProviderName(name),
    tokens: values.total_tokens,
    latency: values.avg_latency_seconds,
    queries: values.queries,
  }));

  const tooltipStyle = useChartTooltipStyle();
  if (chartData.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer className="h-64" role="img" aria-label="Bar chart comparing provider performance by tokens and latency">
          {(w, h) => (
            <BarChart width={w} height={h} data={chartData} barCategoryGap="20%">
              <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
              <XAxis dataKey="name" className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
              <YAxis yAxisId="left" className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
              <YAxis yAxisId="right" orientation="right" className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Bar yAxisId="left" dataKey="tokens" fill={COLORS.tokens} radius={[4, 4, 0, 0]} name="Tokens" />
              <Bar yAxisId="right" dataKey="latency" fill={COLORS.latency} radius={[4, 4, 0, 0]} name="Avg Latency (s)" />
            </BarChart>
          )}
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
