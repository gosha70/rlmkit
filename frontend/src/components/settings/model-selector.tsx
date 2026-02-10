"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface ModelSelectorProps {
  models: string[];
  value: string | null;
  onChange: (model: string) => void;
}

export function ModelSelector({ models, value, onChange }: ModelSelectorProps) {
  return (
    <Select value={value ?? undefined} onValueChange={onChange}>
      <SelectTrigger className="w-full" aria-label="Select model">
        <SelectValue placeholder="Select model" />
      </SelectTrigger>
      <SelectContent>
        {models.map((model) => (
          <SelectItem key={model} value={model}>
            {model}
          </SelectItem>
        ))}
        {models.length === 0 && (
          <div className="px-3 py-2 text-sm text-muted-foreground">No models available</div>
        )}
      </SelectContent>
    </Select>
  );
}
