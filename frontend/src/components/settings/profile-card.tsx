"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { FieldLabel } from "./field-label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  createProfile,
  deleteProfile,
  updateProfile,
  getPromptTemplates,
  getSystemPrompts,
  type RunProfile,
  type ChatProviderConfig,
  type SystemPromptTemplate,
  type SystemPrompts,
} from "@/lib/api";
import useSWR from "swr";
import { Trash2, Lock, Edit2, Copy, Download, ChevronDown, ChevronUp } from "lucide-react";

interface ProfileCardProps {
  profile: RunProfile;
  chatProviders?: ChatProviderConfig[];
  onDeleted?: () => void;
  onUpdated?: () => void;
  onCloned?: () => void;
}

export function ProfileCard({
  profile,
  chatProviders = [],
  onDeleted,
  onUpdated,
  onCloned,
}: ProfileCardProps) {
  const [deleting, setDeleting] = useState(false);
  const [cloning, setCloning] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const { data: templates = [] } = useSWR<SystemPromptTemplate[]>("prompt-templates", getPromptTemplates);
  const { data: livePrompts } = useSWR<SystemPrompts>("system-prompts", getSystemPrompts);
  const [showCustomEditors, setShowCustomEditors] = useState(false);

  // Derive the current template name from stored system_prompts content.
  // Matches against built-in templates + live Prompts-tab values.
  const deriveTemplateName = (prompts: Record<string, string>): string => {
    const hasContent = Object.values(prompts).some((v) => v && v.trim());
    if (!hasContent) return "__global__";
    // Check if it matches the live Prompts tab state
    if (livePrompts) {
      const matchesLive = (["direct", "rlm", "rag"] as const).every(
        (mode) => (livePrompts[mode] ?? "") === (prompts[mode] ?? ""),
      );
      if (matchesLive) return "__live__";
    }
    // Check built-in templates
    for (const t of templates) {
      const match = (["direct", "rlm", "rag"] as const).every(
        (mode) => (t.prompts[mode] ?? "") === (prompts[mode] ?? ""),
      );
      if (match) return t.name;
    }
    return "__custom__";
  };

  const [editData, setEditData] = useState({
    name: profile.name,
    strategy: profile.strategy,
    description: profile.description,
    temperature: profile.runtime_settings.temperature,
    top_p: profile.runtime_settings.top_p,
    max_output_tokens: profile.runtime_settings.max_output_tokens,
    timeout_seconds: profile.runtime_settings.timeout_seconds,
    max_steps: profile.budget.max_steps,
    max_tokens: profile.budget.max_tokens,
    max_cost_usd: profile.budget.max_cost_usd,
    max_time_seconds: profile.budget.max_time_seconds,
    max_recursion_depth: profile.budget.max_recursion_depth,
    repeat_limit: profile.budget.repeat_limit ?? 2,
    nudge_at_fraction: profile.budget.nudge_at_fraction ?? 0.6,
    system_prompts: { ...profile.system_prompts } as Record<string, string>,
  });

  const usedBy = chatProviders.filter((cp) => cp.profile_id === profile.id);

  const handleDelete = async () => {
    if (usedBy.length > 0) {
      const names = usedBy.map((cp) => cp.name).join(", ");
      setMessage(`Cannot delete: used by ${names}`);
      return;
    }
    if (!confirm(`Delete profile "${profile.name}"?`)) return;
    setDeleting(true);
    try {
      await deleteProfile(profile.id);
      onDeleted?.();
    } catch {
      setMessage("Failed to delete profile");
      setDeleting(false);
    }
  };

  const handleClone = async () => {
    setCloning(true);
    setMessage(null);
    try {
      await createProfile({
        name: `Copy of ${profile.name}`,
        description: profile.description,
        strategy: profile.strategy,
        runtime_settings: { ...profile.runtime_settings },
        budget: { ...profile.budget },
      });
      onCloned?.();
    } catch {
      setMessage("Failed to clone profile");
    } finally {
      setCloning(false);
    }
  };

  const handleStartEdit = () => {
    setEditData({
      name: profile.name,
      strategy: profile.strategy,
      description: profile.description,
      temperature: profile.runtime_settings.temperature,
      top_p: profile.runtime_settings.top_p,
      max_output_tokens: profile.runtime_settings.max_output_tokens,
      timeout_seconds: profile.runtime_settings.timeout_seconds,
      max_steps: profile.budget.max_steps,
      max_tokens: profile.budget.max_tokens,
      max_cost_usd: profile.budget.max_cost_usd,
      max_time_seconds: profile.budget.max_time_seconds,
      max_recursion_depth: profile.budget.max_recursion_depth,
      repeat_limit: profile.budget.repeat_limit ?? 2,
      nudge_at_fraction: profile.budget.nudge_at_fraction ?? 0.6,
      system_prompts: { ...profile.system_prompts } as Record<string, string>,
    });
    setEditing(true);
    setMessage(null);
    // Auto-open custom editors if the profile has non-template prompts
    const hasCustom = Object.values(profile.system_prompts).some((v) => v && v.trim());
    setShowCustomEditors(hasCustom && deriveTemplateName({ ...profile.system_prompts } as Record<string, string>) === "__custom__");
  };

  const handleExport = () => {
    const exportData = {
      name: profile.name,
      description: profile.description,
      strategy: profile.strategy,
      runtime_settings: profile.runtime_settings,
      budget: profile.budget,
      system_prompts: profile.system_prompts,
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `profile-${profile.name.toLowerCase().replace(/\s+/g, "-")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleSaveEdit = async () => {
    setSaving(true);
    setMessage(null);
    try {
      await updateProfile(profile.id, {
        name: editData.name.trim() || profile.name,
        strategy: editData.strategy,
        description: editData.description,
        runtime_settings: {
          temperature: editData.temperature,
          top_p: editData.top_p,
          max_output_tokens: editData.max_output_tokens,
          timeout_seconds: editData.timeout_seconds,
        },
        budget: {
          max_steps: editData.max_steps,
          max_tokens: editData.max_tokens,
          max_cost_usd: editData.max_cost_usd,
          max_time_seconds: editData.max_time_seconds,
          max_recursion_depth: editData.max_recursion_depth,
          repeat_limit: editData.repeat_limit,
          nudge_at_fraction: editData.nudge_at_fraction,
        },
        system_prompts: editData.system_prompts,
      });
      setEditing(false);
      onUpdated?.();
    } catch {
      setMessage("Failed to update profile");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle className="text-sm">{profile.name}</CardTitle>
            {profile.is_builtin && (
              <Badge variant="secondary" className="text-xs">
                <Lock className="mr-1 h-3 w-3" aria-hidden="true" />
                Built-in
              </Badge>
            )}
            <Badge variant="outline" className="text-xs capitalize">
              {profile.strategy}
            </Badge>
          </div>
          <div className="flex gap-1">
            {!profile.is_builtin && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleStartEdit}
                disabled={editing}
                aria-label={`Edit ${profile.name} profile`}
              >
                <Edit2 className="h-4 w-4" aria-hidden="true" />
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={handleExport}
              aria-label={`Export ${profile.name} profile`}
            >
              <Download className="h-4 w-4" aria-hidden="true" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClone}
              disabled={cloning}
              aria-label={`Clone ${profile.name} profile`}
            >
              <Copy className="h-4 w-4" aria-hidden="true" />
            </Button>
            {!profile.is_builtin && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleDelete}
                disabled={deleting}
                aria-label={`Delete ${profile.name} profile`}
                className="text-destructive hover:text-destructive"
              >
                <Trash2 className="h-4 w-4" aria-hidden="true" />
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="pb-3">
        {profile.description && (
          <p className="text-xs text-muted-foreground mb-2">{profile.description}</p>
        )}
        <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
          <span>Temp: {profile.runtime_settings.temperature}</span>
          <span>Max tokens: {profile.runtime_settings.max_output_tokens}</span>
          <span>Steps: {profile.budget.max_steps}</span>
        </div>

        {usedBy.length > 0 ? (
          <p className="text-xs text-muted-foreground mt-2">
            Used by: {usedBy.map((cp) => cp.name).join(", ")}
          </p>
        ) : (
          <p className="text-xs text-muted-foreground/60 mt-2">
            Not used by any Chat Provider
          </p>
        )}

        {editing && (
          <div className="mt-4 space-y-3 rounded-lg border border-muted p-3">
            <div className="space-y-1">
              <Label htmlFor={`edit-name-${profile.id}`} className="text-xs">Name</Label>
              <Input
                id={`edit-name-${profile.id}`}
                className="h-8 text-xs"
                value={editData.name}
                onChange={(e) => setEditData({ ...editData, name: e.target.value })}
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor={`edit-strategy-${profile.id}`} className="text-xs">Strategy</Label>
              <Select
                value={editData.strategy}
                onValueChange={(v) => setEditData({ ...editData, strategy: v })}
              >
                <SelectTrigger id={`edit-strategy-${profile.id}`} className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="direct">Direct</SelectItem>
                  <SelectItem value="rlm">RLM</SelectItem>
                  <SelectItem value="rag">RAG</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <Label htmlFor={`edit-desc-${profile.id}`} className="text-xs">Description</Label>
              <Input
                id={`edit-desc-${profile.id}`}
                className="h-8 text-xs"
                value={editData.description}
                onChange={(e) => setEditData({ ...editData, description: e.target.value })}
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-temp-${profile.id}`} className="text-xs" tooltip="Controls randomness. Lower values (0.1-0.3) give focused, deterministic outputs; higher values (0.7-1.0) increase creativity.">Temperature</FieldLabel>
                <Input
                  id={`edit-temp-${profile.id}`}
                  type="number"
                  step="0.1"
                  min="0"
                  max="2"
                  className="h-8 text-xs"
                  value={editData.temperature}
                  onChange={(e) => setEditData({ ...editData, temperature: parseFloat(e.target.value) || 0 })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-topp-${profile.id}`} className="text-xs" tooltip="Nucleus sampling: only consider tokens whose cumulative probability reaches this threshold. 1.0 = no filtering.">Top P</FieldLabel>
                <Input
                  id={`edit-topp-${profile.id}`}
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  className="h-8 text-xs"
                  value={editData.top_p}
                  onChange={(e) => setEditData({ ...editData, top_p: parseFloat(e.target.value) || 0 })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-maxtokens-${profile.id}`} className="text-xs" tooltip="Maximum output tokens per LLM call (not the total budget). Controls response length per step.">Max Tokens</FieldLabel>
                <Input
                  id={`edit-maxtokens-${profile.id}`}
                  type="number"
                  min="1"
                  className="h-8 text-xs"
                  value={editData.max_output_tokens}
                  onChange={(e) => setEditData({ ...editData, max_output_tokens: parseInt(e.target.value) || 1 })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-timeout-${profile.id}`} className="text-xs" tooltip="Per-step timeout in seconds. If a single LLM call exceeds this, it is aborted.">Timeout (s)</FieldLabel>
                <Input
                  id={`edit-timeout-${profile.id}`}
                  type="number"
                  min="1"
                  className="h-8 text-xs"
                  value={editData.timeout_seconds}
                  onChange={(e) => setEditData({ ...editData, timeout_seconds: parseInt(e.target.value) || 1 })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-steps-${profile.id}`} className="text-xs" tooltip="Maximum reasoning steps in RLM mode. More steps allow deeper document exploration but cost more tokens.">Max Steps</FieldLabel>
                <Input
                  id={`edit-steps-${profile.id}`}
                  type="number"
                  min="1"
                  className="h-8 text-xs"
                  value={editData.max_steps}
                  onChange={(e) => setEditData({ ...editData, max_steps: parseInt(e.target.value) || 1 })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-cost-${profile.id}`} className="text-xs" tooltip="Dollar limit per query. The execution is stopped if this cost is exceeded.">Max Cost ($)</FieldLabel>
                <Input
                  id={`edit-cost-${profile.id}`}
                  type="number"
                  step="0.1"
                  min="0"
                  className="h-8 text-xs"
                  value={editData.max_cost_usd}
                  onChange={(e) => setEditData({ ...editData, max_cost_usd: parseFloat(e.target.value) || 0 })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-repeat-limit-${profile.id}`} className="text-xs" tooltip="RLM convergence: how many duplicate inspect actions before forcing finalization. Lower = faster convergence, higher = more retries.">Repeat Limit</FieldLabel>
                <Input
                  id={`edit-repeat-limit-${profile.id}`}
                  type="number"
                  min="1"
                  max="10"
                  className="h-8 text-xs"
                  value={editData.repeat_limit}
                  onChange={(e) => setEditData({ ...editData, repeat_limit: parseInt(e.target.value) || 2 })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-nudge-fraction-${profile.id}`} className="text-xs" tooltip="RLM convergence: fraction of max_steps at which a soft nudge is sent to encourage finalization. E.g., 0.5 = nudge at step 16 of 32.">Nudge at Fraction</FieldLabel>
                <Input
                  id={`edit-nudge-fraction-${profile.id}`}
                  type="number"
                  step="0.05"
                  min="0.1"
                  max="1.0"
                  className="h-8 text-xs"
                  value={editData.nudge_at_fraction}
                  onChange={(e) => setEditData({ ...editData, nudge_at_fraction: parseFloat(e.target.value) || 0.6 })}
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label className="text-xs font-medium">System Prompts</Label>
              <p className="text-xs text-muted-foreground">
                Use the global defaults from the Prompts tab, a built-in preset, or write custom prompts.
              </p>
              <Select
                value={deriveTemplateName(editData.system_prompts)}
                onValueChange={(value) => {
                  if (value === "__global__") {
                    setEditData({ ...editData, system_prompts: {} });
                    setShowCustomEditors(false);
                  } else if (value === "__live__") {
                    if (livePrompts) {
                      setEditData({
                        ...editData,
                        system_prompts: {
                          direct: livePrompts.direct ?? "",
                          rlm: livePrompts.rlm ?? "",
                          rag: livePrompts.rag ?? "",
                        },
                      });
                    }
                    setShowCustomEditors(false);
                  } else if (value === "__custom__") {
                    // Keep current prompts, open editors
                    setShowCustomEditors(true);
                  } else {
                    const tpl = templates.find((t) => t.name === value);
                    if (tpl) {
                      setEditData({
                        ...editData,
                        system_prompts: {
                          direct: tpl.prompts.direct ?? "",
                          rlm: tpl.prompts.rlm ?? "",
                          rag: tpl.prompts.rag ?? "",
                        },
                      });
                    }
                    setShowCustomEditors(false);
                  }
                }}
              >
                <SelectTrigger className="h-8 text-xs" id={`edit-prompt-template-${profile.id}`}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__global__">
                    <span>Use global defaults</span>
                    <span className="ml-2 text-muted-foreground">— inherits from Prompts tab</span>
                  </SelectItem>
                  {livePrompts && (
                    <SelectItem value="__live__" className="text-xs">
                      <span>Prompts tab (current)</span>
                      <span className="ml-2 text-muted-foreground">— snapshot current Prompts tab values</span>
                    </SelectItem>
                  )}
                  {templates.map((t) => (
                    <SelectItem key={t.name} value={t.name} className="text-xs">
                      <span>{t.name}</span>
                      {t.description && (
                        <span className="ml-2 text-muted-foreground">— {t.description}</span>
                      )}
                    </SelectItem>
                  ))}
                  <SelectItem value="__custom__" className="text-xs">
                    <span>Custom</span>
                    <span className="ml-2 text-muted-foreground">— edit per-mode prompts manually</span>
                  </SelectItem>
                </SelectContent>
              </Select>
              {(showCustomEditors || deriveTemplateName(editData.system_prompts) === "__custom__") && (
                <div className="space-y-2 pt-1">
                  <button
                    type="button"
                    className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                    onClick={() => setShowCustomEditors((v) => !v)}
                  >
                    {showCustomEditors ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                    {showCustomEditors ? "Hide prompt editors" : "Show prompt editors"}
                  </button>
                  {showCustomEditors && (
                    <div className="space-y-2">
                      {(["direct", "rlm", "rag"] as const).map((mode) => (
                        <div key={mode} className="space-y-1">
                          <Label htmlFor={`edit-prompt-${mode}-${profile.id}`} className="text-xs capitalize">
                            {mode} mode
                          </Label>
                          <Textarea
                            id={`edit-prompt-${mode}-${profile.id}`}
                            className="min-h-[100px] text-xs font-mono resize-y"
                            placeholder={`System prompt for ${mode} mode`}
                            value={editData.system_prompts[mode] ?? ""}
                            onChange={(e) =>
                              setEditData({
                                ...editData,
                                system_prompts: {
                                  ...editData.system_prompts,
                                  [mode]: e.target.value,
                                },
                              })
                            }
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
            <div className="flex gap-2">
              <Button size="sm" onClick={handleSaveEdit} disabled={saving}>
                {saving ? "Saving..." : "Save"}
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => setEditing(false)}
                disabled={saving}
              >
                Cancel
              </Button>
            </div>
          </div>
        )}

        {message && (
          <p className="text-xs text-destructive mt-2" role="status">
            {message}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
