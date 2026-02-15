"use client";

import { useState } from "react";
import useSWR from "swr";
import { AppShell } from "@/components/shared/app-shell";
import { Timeline } from "@/components/trace/timeline";
import { TraceTree } from "@/components/trace/trace-tree";
import { StepDetail } from "@/components/trace/step-detail";
import { CodeBlock } from "@/components/trace/code-block";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  getExecutions,
  getTrace,
  type ExecutionSummary,
  type TraceResponse,
  type TraceStep,
} from "@/lib/api";

export default function TracesPage() {
  const [trace, setTrace] = useState<TraceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedStep, setSelectedStep] = useState<TraceStep | null>(null);

  const { data: executions = [] } = useSWR<ExecutionSummary[]>(
    "executions",
    () => getExecutions(),
    { refreshInterval: 5000 },
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

  return (
    <AppShell>
      <div className="mx-auto max-w-[1200px] space-y-6 p-6">
        <h2 className="text-2xl font-semibold">Traces</h2>

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
