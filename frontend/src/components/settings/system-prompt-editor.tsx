"use client";

import { useState } from "react";
import useSWR from "swr";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
  const [activeTab, setActiveTab] = useState<string>("direct");

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
    <Card className="flex flex-col" style={{ minHeight: "calc(100vh - 240px)" }}>
      <CardHeader className="shrink-0">
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
      <CardContent className="flex flex-1 flex-col gap-3">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex flex-1 flex-col">
          <TabsList className="shrink-0">
            {MODES.map(({ key, label }) => (
              <TabsTrigger key={key} value={key}>
                {label} Mode
              </TabsTrigger>
            ))}
          </TabsList>

          {MODES.map(({ key, label }) => (
            <TabsContent key={key} value={key} className="flex-1 mt-3">
              <Textarea
                id={`prompt-${key}`}
                value={current[key]}
                onChange={(e) => handleChange(key, e.target.value)}
                placeholder={`System prompt for ${label} mode (leave empty to use built-in default)`}
                className="h-full min-h-[300px] text-sm font-mono resize-y"
                aria-label={`System prompt for ${label} mode`}
              />
            </TabsContent>
          ))}
        </Tabs>

        <div className="flex items-center gap-2 shrink-0">
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
