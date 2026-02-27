"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface ProviderPerformanceProps {
  data: Record<string, { queries: number; total_tokens: number; total_cost_usd: number; avg_latency_seconds: number }>;
}

const COLORS = { tokens: "#2563eb", latency: "#d97706", cost: "#7c3aed" };

export function ProviderPerformance({ data }: ProviderPerformanceProps) {
  const chartData = Object.entries(data).map(([name, values]) => ({
    name,
    tokens: values.total_tokens,
    latency: values.avg_latency_seconds,
    cost: values.total_cost_usd,
    queries: values.queries,
  }));

  if (chartData.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Provider Performance</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64" role="img" aria-label="Bar chart comparing provider performance by tokens, latency, and cost">
          <ResponsiveContainer width="100%" height="100%" minWidth={0}>
            <BarChart data={chartData} barCategoryGap="20%">
              <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
              <XAxis dataKey="name" className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
              <YAxis yAxisId="left" className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
              <YAxis yAxisId="right" orientation="right" className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "var(--color-card)",
                  border: "1px solid var(--color-border)",
                  borderRadius: "6px",
                  fontSize: "12px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Bar yAxisId="left" dataKey="tokens" fill={COLORS.tokens} radius={[4, 4, 0, 0]} name="Tokens" />
              <Bar yAxisId="right" dataKey="latency" fill={COLORS.latency} radius={[4, 4, 0, 0]} name="Avg Latency (s)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
