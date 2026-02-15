"use client";

import { useState } from "react";
import useSWR from "swr";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  getSystemPrompts,
  updateSystemPrompts,
  getPromptTemplates,
  type SystemPrompts,
  type SystemPromptTemplate,
} from "@/lib/api";
import { Save, RotateCcw } from "lucide-react";

const MODES = [
  { key: "direct" as const, label: "Direct" },
  { key: "rlm" as const, label: "RLM" },
  { key: "rag" as const, label: "RAG" },
];

export function SystemPromptEditor() {
  const { data: prompts, mutate: mutatePrompts } = useSWR("system-prompts", getSystemPrompts);
  const { data: templates = [] } = useSWR<SystemPromptTemplate[]>("prompt-templates", getPromptTemplates);
  const [local, setLocal] = useState<SystemPrompts | null>(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const current = local ?? prompts ?? { direct: "", rlm: "", rag: "" };

  const handleChange = (mode: keyof SystemPrompts, value: string) => {
    setLocal({ ...current, [mode]: value });
    setMessage(null);
  };

  const handleSave = async () => {
    setSaving(true);
    setMessage(null);
    try {
      const updated = await updateSystemPrompts(current);
      mutatePrompts(updated, false);
      setLocal(null);
      setMessage("Prompts saved");
    } catch {
      setMessage("Failed to save prompts");
    } finally {
      setSaving(false);
    }
  };

  const handleApplyTemplate = (templateName: string) => {
    const template = templates.find((t) => t.name === templateName);
    if (!template) return;
    setLocal({
      direct: template.prompts.direct ?? "",
      rlm: template.prompts.rlm ?? "",
      rag: template.prompts.rag ?? "",
    });
    setMessage(null);
  };

  const handleReset = () => {
    setLocal(null);
    setMessage(null);
  };

  const hasChanges = local !== null;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">System Prompts</CardTitle>
          <div className="flex items-center gap-2">
            {templates.length > 0 && (
              <Select onValueChange={handleApplyTemplate}>
                <SelectTrigger className="h-8 w-auto min-w-[160px] text-xs" aria-label="Apply template">
                  <SelectValue placeholder="Apply template..." />
                </SelectTrigger>
                <SelectContent>
                  {templates.map((t) => (
                    <SelectItem key={t.name} value={t.name} className="text-xs">
                      <div>
                        <span>{t.name}</span>
                        {t.description && (
                          <span className="text-muted-foreground ml-2">- {t.description}</span>
                        )}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {MODES.map(({ key, label }) => (
          <div key={key} className="space-y-2">
            <Label htmlFor={`prompt-${key}`}>{label} Mode</Label>
            <Textarea
              id={`prompt-${key}`}
              value={current[key]}
              onChange={(e) => handleChange(key, e.target.value)}
              placeholder={`System prompt for ${label} mode (leave empty to use built-in default)`}
              rows={4}
              className="text-sm font-mono"
              aria-label={`System prompt for ${label} mode`}
            />
          </div>
        ))}

        <div className="flex items-center gap-2">
          <Button
            onClick={handleSave}
            disabled={saving || !hasChanges}
            size="sm"
          >
            <Save className="mr-1 h-4 w-4" aria-hidden="true" />
            {saving ? "Saving..." : "Save Prompts"}
          </Button>
          <Button variant="ghost" size="sm" onClick={handleReset} disabled={!hasChanges}>
            <RotateCcw className="mr-1 h-4 w-4" aria-hidden="true" />
            Reset
          </Button>
          {message && (
            <span className="text-xs text-green-600 dark:text-green-400" role="status">
              {message}
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
