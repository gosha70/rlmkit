"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface CostBreakdownProps {
  data: Record<string, { queries: number; total_tokens: number; total_cost_usd: number }>;
}

const COLORS = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626", "#6366f1"];

export function CostBreakdown({ data }: CostBreakdownProps) {
  const chartData = Object.entries(data).map(([name, values]) => ({
    name,
    value: values.total_cost_usd,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Cost by Provider</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64" role="img" aria-label="Pie chart showing cost breakdown by provider">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={80}
                dataKey="value"
                label={({ name, value }) => `${name}: $${(value as number).toFixed(2)}`}
                labelLine={false}
              >
                {chartData.map((_, index) => (
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value) => `$${Number(value).toFixed(4)}`}
                contentStyle={{
                  backgroundColor: "var(--color-card)",
                  border: "1px solid var(--color-border)",
                  borderRadius: "6px",
                  fontSize: "12px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
