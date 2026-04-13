"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import { ChartContainer } from "./chart-container";
import { CheckCircle2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useChartTooltipStyle } from "./use-chart-tooltip";
import type { FailureMetricsResponse, FailureCategory } from "@/lib/api";

interface FailureBreakdownProps {
  data: FailureMetricsResponse;
}

const CATEGORY_LABELS: Record<FailureCategory, string> = {
  timeout: "Timeout",
  budget_exhausted: "Budget Exhausted",
  context_overflow: "Context Overflow",
  general_error: "General Error",
};

const ALL_CATEGORIES: FailureCategory[] = [
  "timeout",
  "budget_exhausted",
  "context_overflow",
  "general_error",
];

// Provider colors — same palette as judge-dimensions-chart
const PROVIDER_COLORS = [
  "#10b981", // emerald
  "#6366f1", // indigo
  "#a855f7", // purple
  "#f97316", // orange
  "#3b82f6", // blue
  "#ec4899", // pink
  "#14b8a6", // teal
  "#eab308", // yellow
];

export function FailureBreakdown({ data }: FailureBreakdownProps) {
  const tooltipStyle = useChartTooltipStyle();
  const { total_runs, total_failures, failure_rate, by_provider } = data;

  // Build grouped chart data: one row per error category, one bar per provider.
  // Shape: [{ category: "Timeout", "vLLM-RLM": 1, "DIRECT-ANTHROPIC": 0, ... }, ...]
  const providers = Object.keys(by_provider);

  const chartData = ALL_CATEGORIES
    .map((cat) => {
      const row: Record<string, string | number> = {
        category: CATEGORY_LABELS[cat],
      };
      let hasAny = false;
      for (const provider of providers) {
        const entry = by_provider[provider]?.find((c) => c.category === cat);
        const count = entry?.count ?? 0;
        row[provider] = count;
        if (count > 0) hasAny = true;
      }
      return hasAny ? row : null;
    })
    .filter(Boolean) as Record<string, string | number>[];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">⚠️ Failure Analysis (by Chat Provider)</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {total_failures === 0 ? (
          <div className="flex items-center gap-2 py-6 text-sm text-green-600 dark:text-green-400">
            <CheckCircle2 className="h-5 w-5" />
            <span>No failures in this session</span>
          </div>
        ) : (
          <>
            <p className="text-sm text-muted-foreground">
              {total_failures} of {total_runs} runs failed ({(failure_rate * 100).toFixed(0)}%)
            </p>

            {chartData.length > 0 && (
              <ChartContainer className="h-64" role="img" aria-label="Grouped bar chart showing failures per category per provider">
                {(w, h) => (
                  <BarChart width={w} height={h} data={chartData} barCategoryGap="20%">
                    <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                    <XAxis
                      dataKey="category"
                      className="text-xs"
                      tick={{ fill: "var(--color-muted-foreground)" }}
                    />
                    <YAxis
                      allowDecimals={false}
                      className="text-xs"
                      tick={{ fill: "var(--color-muted-foreground)" }}
                    />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Legend />
                    {providers.map((provider, i) => (
                      <Bar
                        key={provider}
                        dataKey={provider}
                        name={provider}
                        fill={PROVIDER_COLORS[i % PROVIDER_COLORS.length]}
                        radius={[4, 4, 0, 0]}
                      />
                    ))}
                  </BarChart>
                )}
              </ChartContainer>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
