"use client";

import { cn } from "@/lib/utils";
import type { ChatMode } from "@/lib/api";

const MODES = [
  { value: "auto", label: "Auto" },
  { value: "rlm", label: "RLM" },
  { value: "direct", label: "Direct" },
  { value: "rag", label: "RAG" },
  { value: "compare", label: "Compare" },
] as const;

interface ModeSelectorProps {
  value: ChatMode;
  onChange: (mode: ChatMode) => void;
}

export function ModeSelector({ value, onChange }: ModeSelectorProps) {
  return (
    <div className="flex gap-1 rounded-lg border bg-muted p-1" role="radiogroup" aria-label="Execution mode">
      {MODES.map((mode) => (
        <button
          key={mode.value}
          role="radio"
          aria-checked={value === mode.value}
          onClick={() => onChange(mode.value)}
          className={cn(
            "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
            value === mode.value
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {mode.label}
        </button>
      ))}
    </div>
  );
}
