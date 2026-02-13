"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import type { BudgetConfig as BudgetConfigType } from "@/lib/api";

interface BudgetConfigProps {
  config: BudgetConfigType;
  onChange: (config: BudgetConfigType) => void;
}

const DEFAULTS: BudgetConfigType = {
  max_steps: 16,
  max_tokens: 50000,
  max_cost_usd: 2.0,
  max_time_seconds: 30,
  max_recursion_depth: 5,
};

export function BudgetConfig({ config, onChange }: BudgetConfigProps) {
  const [local, setLocal] = useState(config);

  const update = <K extends keyof BudgetConfigType>(key: K, value: BudgetConfigType[K]) => {
    setLocal((prev) => ({ ...prev, [key]: value }));
  };

  const handleSave = () => {
    onChange(local);
  };

  const handleReset = () => {
    setLocal(DEFAULTS);
    onChange(DEFAULTS);
  };

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Label htmlFor="max-steps">Max Steps: {local.max_steps}</Label>
        <Slider
          id="max-steps"
          min={1}
          max={50}
          step={1}
          value={[local.max_steps]}
          onValueChange={([v]) => update("max_steps", v)}
          aria-label="Max steps"
          aria-valuetext={`${local.max_steps} steps`}
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="max-tokens">Max Tokens</Label>
        <Input
          id="max-tokens"
          type="number"
          min={1000}
          max={500000}
          value={local.max_tokens}
          onChange={(e) => update("max_tokens", Number(e.target.value))}
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="max-cost">Max Cost (USD)</Label>
        <Input
          id="max-cost"
          type="number"
          min={0.01}
          max={100}
          step={0.01}
          value={local.max_cost_usd}
          onChange={(e) => update("max_cost_usd", Number(e.target.value))}
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="max-time">Max Time (seconds): {local.max_time_seconds}</Label>
        <Slider
          id="max-time"
          min={5}
          max={120}
          step={5}
          value={[local.max_time_seconds]}
          onValueChange={([v]) => update("max_time_seconds", v)}
          aria-label="Max time in seconds"
          aria-valuetext={`${local.max_time_seconds} seconds`}
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="max-depth">Max Recursion Depth: {local.max_recursion_depth}</Label>
        <Slider
          id="max-depth"
          min={1}
          max={20}
          step={1}
          value={[local.max_recursion_depth]}
          onValueChange={([v]) => update("max_recursion_depth", v)}
          aria-label="Max recursion depth"
          aria-valuetext={`${local.max_recursion_depth} levels`}
        />
      </div>

      <div className="flex gap-2">
        <Button onClick={handleSave}>Save</Button>
        <Button variant="outline" onClick={handleReset}>
          Reset to Defaults
        </Button>
      </div>
    </div>
  );
}
