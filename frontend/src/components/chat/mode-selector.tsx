"use client";

import { cn } from "@/lib/utils";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ProviderInfo } from "@/lib/api";

const STRATEGIES = [
  { value: "rlm", label: "RLM" },
  { value: "direct", label: "Direct" },
  { value: "rag", label: "RAG" },
] as const;

export type Strategy = "rlm" | "direct" | "rag";

interface ModeSelectorProps {
  modes: Strategy[];
  onModesChange: (modes: Strategy[]) => void;
  providers: ProviderInfo[];
  selectedProvider: string;
  onProviderChange: (provider: string) => void;
}

export function ModeSelector({
  modes,
  onModesChange,
  providers,
  selectedProvider,
  onProviderChange,
}: ModeSelectorProps) {
  const toggle = (strategy: Strategy) => {
    if (modes.includes(strategy)) {
      // Don't allow deselecting the last mode
      if (modes.length === 1) return;
      onModesChange(modes.filter((m) => m !== strategy));
    } else {
      onModesChange([...modes, strategy]);
    }
  };

  const connectedProviders = providers.filter(
    (p) => p.status === "connected" || p.status === "configured",
  );

  return (
    <div className="flex items-center gap-3">
      {/* Strategy toggles */}
      <div
        className="flex gap-1 rounded-lg border bg-muted p-1"
        role="group"
        aria-label="Execution strategies"
      >
        {STRATEGIES.map((s) => (
          <button
            key={s.value}
            role="checkbox"
            aria-checked={modes.includes(s.value)}
            onClick={() => toggle(s.value)}
            className={cn(
              "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
              modes.includes(s.value)
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {s.label}
          </button>
        ))}
      </div>

      {/* Provider selector */}
      {connectedProviders.length > 0 && (
        <Select value={selectedProvider} onValueChange={onProviderChange}>
          <SelectTrigger className="h-8 w-auto min-w-[160px] text-xs" aria-label="LLM provider">
            <SelectValue placeholder="Select provider" />
          </SelectTrigger>
          <SelectContent>
            {connectedProviders.map((p) => (
              <SelectItem key={p.name} value={p.name} className="text-xs">
                <span className="flex items-center gap-1.5">
                  <span
                    className={cn(
                      "inline-block h-2 w-2 rounded-full",
                      p.status === "connected" ? "bg-emerald-500" : "bg-amber-500",
                    )}
                  />
                  {p.default_model} via {p.display_name}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </div>
  );
}
