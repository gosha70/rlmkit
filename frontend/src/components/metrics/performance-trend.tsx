"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { TimelineEntry } from "@/lib/api";

interface PerformanceTrendProps {
  timeline: TimelineEntry[];
}

export function PerformanceTrend({ timeline }: PerformanceTrendProps) {
  const data = timeline.map((entry, i) => ({
    index: i + 1,
    tokens: entry.tokens,
    cost: entry.cost_usd,
    latency: entry.latency_seconds,
    mode: entry.mode,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Performance Over Time</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
              <XAxis dataKey="index" label={{ value: "Query #", position: "insideBottomRight", offset: -5 }} className="text-xs" tick={{ fill: "var(--color-muted-foreground)" }} />
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
              <Line yAxisId="left" type="monotone" dataKey="tokens" stroke="#2563eb" strokeWidth={2} dot={false} name="Tokens" />
              <Line yAxisId="right" type="monotone" dataKey="cost" stroke="#7c3aed" strokeWidth={2} dot={false} name="Cost ($)" />
              <Line yAxisId="right" type="monotone" dataKey="latency" stroke="#d97706" strokeWidth={2} dot={false} name="Latency (s)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
