"use client";

import { useState } from "react";
import { AppShell } from "@/components/shared/app-shell";
import { Timeline } from "@/components/trace/timeline";
import { TraceTree } from "@/components/trace/trace-tree";
import { StepDetail } from "@/components/trace/step-detail";
import { CodeBlock } from "@/components/trace/code-block";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getTrace, type TraceResponse, type TraceStep } from "@/lib/api";

export default function TracesPage() {
  const [executionId, setExecutionId] = useState("");
  const [trace, setTrace] = useState<TraceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedStep, setSelectedStep] = useState<TraceStep | null>(null);

  const handleLoad = async () => {
    const id = executionId.trim();
    if (!id) return;
    setError(null);
    setLoading(true);
    setSelectedStep(null);
    try {
      const data = await getTrace(id);
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
        <h2 className="text-2xl font-semibold">Trace Viewer</h2>

        {/* Execution ID input */}
        <div className="flex items-center gap-2">
          <Input
            value={executionId}
            onChange={(e) => setExecutionId(e.target.value)}
            placeholder="Enter execution ID..."
            className="flex-1"
            onKeyDown={(e) => e.key === "Enter" && handleLoad()}
            aria-label="Execution ID"
          />
          <Button onClick={handleLoad} disabled={loading || !executionId.trim()}>
            {loading ? "Loading..." : "Load Trace"}
          </Button>
        </div>

        {error && <p className="text-sm text-destructive" role="alert">{error}</p>}

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

        {!trace && !error && !loading && (
          <div className="py-20 text-center text-muted-foreground">
            <p>Enter an execution ID to view its trace.</p>
          </div>
        )}
      </div>
    </AppShell>
  );
}
