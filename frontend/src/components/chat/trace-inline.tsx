"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import type { TraceStep } from "@/lib/api";

interface TraceInlineProps {
  steps: TraceStep[];
  expanded?: boolean;
  className?: string;
}

export function TraceInline({ steps, expanded: initialExpanded = false, className }: TraceInlineProps) {
  const [expanded, setExpanded] = useState(initialExpanded);

  if (steps.length === 0) return null;

  return (
    <div className={cn("rounded-md border text-xs", className)}>
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-1.5 text-muted-foreground hover:bg-muted/50 transition-colors"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
        aria-label={`Trace: ${steps.length} steps. Click to ${expanded ? "collapse" : "expand"}`}
      >
        {expanded ? <ChevronDown className="h-3 w-3" aria-hidden="true" /> : <ChevronRight className="h-3 w-3" aria-hidden="true" />}
        <span>{steps.length} steps</span>
      </button>

      {expanded && (
        <div className="border-t px-3 py-2 space-y-1">
          {steps.map((step) => (
            <div key={step.index} className="flex items-center gap-2">
              <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                {step.action_type}
              </Badge>
              <span className="truncate flex-1 font-mono">{step.code ?? step.output ?? "--"}</span>
              <span className="text-muted-foreground shrink-0">
                {step.duration_seconds.toFixed(1)}s
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
