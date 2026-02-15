"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { type ModelInfo } from "@/lib/api";

interface ModelSelectorProps {
  models: ModelInfo[];
  value: string | null;
  onChange: (model: string) => void;
  /** For local providers: show free-text input instead of dropdown */
  freeText?: boolean;
  placeholder?: string;
}

function formatCost(cost: number): string {
  if (cost === 0) return "free";
  if (cost < 0.001) return `$${cost.toFixed(5)}`;
  return `$${cost}`;
}

export function ModelSelector({
  models,
  value,
  onChange,
  freeText = false,
  placeholder = "Select model",
}: ModelSelectorProps) {
  if (freeText) {
    return (
      <Input
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        aria-label="Model name"
      />
    );
  }

  return (
    <Select value={value ?? undefined} onValueChange={onChange}>
      <SelectTrigger className="w-full" aria-label="Select model">
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {models.map((model) => (
          <SelectItem key={model.name} value={model.name}>
            <span className="flex items-center justify-between gap-3 w-full">
              <span>{model.name}</span>
              {(model.input_cost_per_1k > 0 || model.output_cost_per_1k > 0) && (
                <span className="text-xs text-muted-foreground ml-auto">
                  {formatCost(model.input_cost_per_1k)}/{formatCost(model.output_cost_per_1k)} per 1K
                </span>
              )}
            </span>
          </SelectItem>
        ))}
        {models.length === 0 && (
          <div className="px-3 py-2 text-sm text-muted-foreground">No models available</div>
        )}
      </SelectContent>
    </Select>
  );
}
