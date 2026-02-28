"use client";

import { useEffect, useState } from "react";
import useSWR from "swr";
import { Hash, DollarSign, Clock, Activity } from "lucide-react";
import { AppShell } from "@/components/shared/app-shell";
import { MetricCard } from "@/components/metrics/metric-card";
import { ComparisonChart } from "@/components/metrics/comparison-chart";
import { CostBreakdown } from "@/components/metrics/cost-breakdown";
import { ProviderPerformance } from "@/components/metrics/provider-performance";
import { PerformanceTrend } from "@/components/metrics/performance-trend";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { getSessions, getMetrics, type SessionSummary, type MetricsResponse } from "@/lib/api";

export default function DashboardPage() {
  const [selectedSession, setSelectedSession] = useState<string>("");

  const { data: sessions = [] } = useSWR<SessionSummary[]>("sessions", () => getSessions());
  const { data: metrics, isLoading } = useSWR<MetricsResponse>(
    selectedSession ? `metrics-${selectedSession}` : null,
    () => getMetrics(selectedSession),
  );

  // Auto-select the most recent session
  useEffect(() => {
    if (!selectedSession && sessions.length > 0) {
      setSelectedSession(sessions[0].id);
    }
  }, [sessions, selectedSession]);

  const summary = metrics?.summary;

  return (
    <AppShell>
      <div className="mx-auto max-w-[1200px] space-y-6 p-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-semibold">Dashboard</h2>
          <Select value={selectedSession} onValueChange={setSelectedSession}>
            <SelectTrigger className="w-56" aria-label="Select session">
              <SelectValue placeholder="Select session..." />
            </SelectTrigger>
            <SelectContent>
              {sessions.map((s) => (
                <SelectItem key={s.id} value={s.id}>
                  {s.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Summary cards */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            label="Total Tokens"
            value={summary ? summary.total_tokens.toLocaleString() : "0"}
            icon={Hash}
          />
          <MetricCard
            label="Total Cost"
            value={summary ? `$${summary.total_cost_usd.toFixed(2)}` : "$0.00"}
            icon={DollarSign}
          />
          <MetricCard
            label="Avg Latency"
            value={summary ? summary.avg_latency_seconds.toFixed(1) : "0.0"}
            unit="s"
            icon={Clock}
          />
          <MetricCard
            label="Token Savings"
            value={summary ? `${summary.avg_token_savings_percent.toFixed(0)}%` : "0%"}
            icon={Activity}
          />
        </div>

        {isLoading && (
          <div className="grid grid-cols-2 gap-4">
            <Skeleton className="h-72" />
            <Skeleton className="h-72" />
          </div>
        )}

        {/* Charts row */}
        {metrics && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <ComparisonChart
              rlmData={metrics.by_mode.rlm}
              directData={metrics.by_mode.direct}
            />
            <CostBreakdown data={metrics.by_provider} />
          </div>
        )}

        {/* Provider performance */}
        {metrics && Object.keys(metrics.by_provider).length > 0 && (
          <ProviderPerformance data={metrics.by_provider} />
        )}

        {/* Chat Provider performance */}
        {metrics && metrics.by_chat_provider && Object.keys(metrics.by_chat_provider).length > 0 && (
          <>
            <ProviderPerformance data={metrics.by_chat_provider} title="Chat Provider Performance" />
            <CostBreakdown data={metrics.by_chat_provider} title="Cost by Chat Provider" />
          </>
        )}

        {/* Performance trend */}
        {metrics && metrics.timeline.length > 0 && (
          <PerformanceTrend timeline={metrics.timeline} />
        )}

        {/* Recent executions table */}
        {metrics && metrics.timeline.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Recent Executions</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>#</TableHead>
                    <TableHead>Mode</TableHead>
                    <TableHead>Provider</TableHead>
                    <TableHead className="text-right">Tokens</TableHead>
                    <TableHead className="text-right">Cost</TableHead>
                    <TableHead className="text-right">Latency</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {metrics.timeline.map((entry, i) => (
                    <TableRow key={i}>
                      <TableCell>{i + 1}</TableCell>
                      <TableCell>
                        <Badge variant="outline">{entry.mode.toUpperCase()}</Badge>
                      </TableCell>
                      <TableCell className="text-muted-foreground text-sm">
                        {entry.provider || "—"}
                      </TableCell>
                      <TableCell className="text-right">
                        {entry.tokens.toLocaleString()}
                      </TableCell>
                      <TableCell className="text-right">
                        ${entry.cost_usd.toFixed(4)}
                      </TableCell>
                      <TableCell className="text-right">
                        {entry.latency_seconds.toFixed(1)}s
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}

        {!selectedSession && (
          <div className="py-20 text-center text-muted-foreground">
            <p>Select a session to view metrics.</p>
          </div>
        )}
      </div>
    </AppShell>
  );
}
