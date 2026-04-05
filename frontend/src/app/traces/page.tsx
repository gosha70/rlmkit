"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import useSWR from "swr";
import { AppShell } from "@/components/shared/app-shell";
import { Timeline } from "@/components/trace/timeline";
import { TraceTree } from "@/components/trace/trace-tree";
import { StepDetail } from "@/components/trace/step-detail";
import { CodeBlock } from "@/components/trace/code-block";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  getExecutions,
  getSessions,
  getTrace,
  type ExecutionSummary,
  type SessionSummary,
  type TraceResponse,
  type TraceStep,
} from "@/lib/api";

function TracesPageInner() {
  const searchParams = useSearchParams();
  const [trace, setTrace] = useState<TraceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedStep, setSelectedStep] = useState<TraceStep | null>(null);
  const [filterProviderId, setFilterProviderId] = useState<string>("");
  const [filterSessionId, setFilterSessionId] = useState<string>("");
  const [limit, setLimit] = useState(20);

  const { data: sessions = [] } = useSWR<SessionSummary[]>("sessions", getSessions);

  // Unfiltered fetch — drives the chat-provider filter dropdown so it never disappears
  const { data: allExecutions = [] } = useSWR<ExecutionSummary[]>(
    ["executions-all", filterSessionId],
    () => getExecutions(50, undefined, filterSessionId || undefined),
    { refreshInterval: 5000 },
  );

  // Filtered fetch — drives the table
  const { data: executions = [] } = useSWR<ExecutionSummary[]>(
    ["executions", filterProviderId, filterSessionId, limit],
    () => getExecutions(limit, filterProviderId || undefined, filterSessionId || undefined),
    { refreshInterval: 5000 },
  );

  // Unique chat providers from ALL executions (within selected session) for the filter dropdown
  const chatProviderOptions = Array.from(
    new Map(
      allExecutions
        .filter((e) => e.chat_provider_id && e.chat_provider_name)
        .map((e) => [e.chat_provider_id!, e.chat_provider_name!]),
    ),
  );

  const handleSelectExecution = async (executionId: string) => {
    setError(null);
    setLoading(true);
    setSelectedStep(null);
    try {
      const data = await getTrace(executionId);
      setTrace(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load trace");
      setTrace(null);
    } finally {
      setLoading(false);
    }
  };

  // Auto-load trace when navigated from Dashboard with ?exec=<id>
  useEffect(() => {
    const execId = searchParams.get("exec");
    if (execId) {
      handleSelectExecution(execId);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <AppShell>
      <div className="mx-auto max-w-[1200px] space-y-6 p-6">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <h2 className="text-2xl font-semibold">Traces</h2>
          <div className="flex items-center gap-2">
            {sessions.length > 0 && (
              <Select
                value={filterSessionId}
                onValueChange={(v) => { setFilterSessionId(v === "all" ? "" : v); setFilterProviderId(""); setLimit(20); }}
              >
                <SelectTrigger className="w-48" aria-label="Filter by Session">
                  <SelectValue placeholder="All Sessions" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Sessions</SelectItem>
                  {sessions.map((s) => (
                    <SelectItem key={s.id} value={s.id}>{s.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            {chatProviderOptions.length > 0 && (
              <Select
                value={filterProviderId}
                onValueChange={(v) => { setFilterProviderId(v === "all" ? "" : v); setLimit(20); }}
              >
                <SelectTrigger className="w-56" aria-label="Filter by Chat Provider">
                  <SelectValue placeholder="All Chat Providers" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Chat Providers</SelectItem>
                  {chatProviderOptions.map(([id, name]) => (
                    <SelectItem key={id} value={id}>{name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>
        </div>

        {/* Execution list */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Recent Executions</CardTitle>
          </CardHeader>
          <CardContent>
            {executions.length === 0 ? (
              <p className="py-8 text-center text-sm text-muted-foreground">
                No executions yet. Send a message in Chat to see traces here.
              </p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Query</TableHead>
                    <TableHead>Chat Provider</TableHead>
                    <TableHead>Mode</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Tokens</TableHead>
                    <TableHead className="text-right">Cost</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {executions.map((exec) => (
                    <TableRow
                      key={exec.execution_id}
                      className="cursor-pointer hover:bg-muted/50"
                      onClick={() => handleSelectExecution(exec.execution_id)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          handleSelectExecution(exec.execution_id);
                        }
                      }}
                      aria-label={`View trace for: ${exec.query}`}
                    >
                      <TableCell className="max-w-[300px] truncate font-medium">
                        {exec.query}
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {exec.chat_provider_name || "—"}
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">{exec.mode.toUpperCase()}</Badge>
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={
                            exec.status === "complete"
                              ? "success"
                              : exec.status === "running"
                                ? "default"
                                : "destructive"
                          }
                        >
                          {exec.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        {exec.total_tokens.toLocaleString()}
                      </TableCell>
                      <TableCell className="text-right">
                        ${exec.total_cost.toFixed(4)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
            {executions.length === limit && (
              <div className="mt-3 flex justify-center">
                <Button variant="outline" size="sm" onClick={() => setLimit((l) => l + 20)}>
                  Load more
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {error && <p className="text-sm text-destructive" role="alert">{error}</p>}
        {loading && <p className="text-sm text-muted-foreground">Loading trace...</p>}

        {trace && (
          <>
            {/* Summary bar */}
            <Card>
              <CardContent className="flex flex-wrap items-center gap-4 p-4">
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-muted-foreground">Query</p>
                  <p className="truncate font-medium">{trace.query}</p>
                </div>
                <Badge variant="outline">{trace.mode.toUpperCase()}</Badge>
                {trace.chat_provider_name && (
                  <>
                    <Separator orientation="vertical" className="h-8" />
                    <div className="text-sm">
                      <span className="text-muted-foreground">Chat Provider: </span>
                      <span className="font-medium">{trace.chat_provider_name}</span>
                    </div>
                  </>
                )}
                <Separator orientation="vertical" className="h-8" />
                <div className="text-sm">
                  <span className="text-muted-foreground">Steps: </span>
                  <span className="font-medium">{trace.budget.steps_used}/{trace.budget.steps_limit}</span>
                </div>
                <Separator orientation="vertical" className="h-8" />
                <div className="text-sm">
                  <span className="text-muted-foreground">Tokens: </span>
                  <span className="font-medium">{trace.budget.tokens_used.toLocaleString()}</span>
                </div>
                <Separator orientation="vertical" className="h-8" />
                <div className="text-sm">
                  <span className="text-muted-foreground">Cost: </span>
                  <span className="font-medium">${trace.budget.cost_used.toFixed(4)}</span>
                </div>
                <Badge variant={trace.result.success ? "success" : "destructive"}>
                  {trace.status}
                </Badge>
              </CardContent>
            </Card>

            {/* Tabbed views */}
            <Tabs defaultValue="timeline">
              <TabsList>
                <TabsTrigger value="timeline">Timeline</TabsTrigger>
                <TabsTrigger value="tree">Tree</TabsTrigger>
                <TabsTrigger value="code">Code</TabsTrigger>
              </TabsList>

              <TabsContent value="timeline">
                <Timeline
                  steps={trace.steps}
                  onSelect={setSelectedStep}
                  selectedIndex={selectedStep?.index}
                />
              </TabsContent>

              <TabsContent value="tree">
                <TraceTree
                  steps={trace.steps}
                  onSelect={setSelectedStep}
                  selectedIndex={selectedStep?.index}
                />
              </TabsContent>

              <TabsContent value="code">
                <div className="space-y-4">
                  {trace.steps
                    .filter((s) => s.code)
                    .map((step) => (
                      <div key={step.index}>
                        <p className="mb-1 text-sm font-medium">
                          Step {step.index + 1}: {step.action_type}
                        </p>
                        <CodeBlock code={step.code!} output={step.output} />
                      </div>
                    ))}
                  {trace.steps.filter((s) => s.code).length === 0 && (
                    <p className="text-sm text-muted-foreground">No code steps in this trace.</p>
                  )}
                </div>
              </TabsContent>
            </Tabs>

            {/* Step detail panel */}
            {selectedStep && (
              <div className="mt-4">
                <StepDetail step={selectedStep} />
              </div>
            )}
          </>
        )}
      </div>
    </AppShell>
  );
}

export default function TracesPage() {
  return (
    <Suspense>
      <TracesPageInner />
    </Suspense>
  );
}
