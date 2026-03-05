"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  activateProfile,
  createProfile,
  deleteProfile,
  updateProfile,
  type RunProfile,
  type ChatProviderConfig,
} from "@/lib/api";
import { Play, Trash2, Lock, Edit2, Copy } from "lucide-react";

interface ProfileCardProps {
  profile: RunProfile;
  chatProviders?: ChatProviderConfig[];
  onActivated?: () => void;
  onDeleted?: () => void;
  onUpdated?: () => void;
  onCloned?: () => void;
}

export function ProfileCard({
  profile,
  chatProviders = [],
  onActivated,
  onDeleted,
  onUpdated,
  onCloned,
}: ProfileCardProps) {
  const [activating, setActivating] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [cloning, setCloning] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [editData, setEditData] = useState({
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
  });

  const usedBy = chatProviders.filter((cp) => cp.profile_id === profile.id);

  const handleActivate = async () => {
    setActivating(true);
    setMessage(null);
    try {
      await activateProfile(profile.id);
      setMessage("Profile activated");
      onActivated?.();
    } catch {
      setMessage("Failed to activate profile");
    } finally {
      setActivating(false);
    }
  };

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
      setMessage("Profile cloned");
      onCloned?.();
    } catch {
      setMessage("Failed to clone profile");
    } finally {
      setCloning(false);
    }
  };

  const handleStartEdit = () => {
    setEditData({
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
    });
    setEditing(true);
    setMessage(null);
  };

  const handleSaveEdit = async () => {
    setSaving(true);
    setMessage(null);
    try {
      await updateProfile(profile.id, {
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
        },
      });
      setEditing(false);
      setMessage("Profile updated");
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
              onClick={handleClone}
              disabled={cloning}
              aria-label={`Clone ${profile.name} profile`}
            >
              <Copy className="h-4 w-4" aria-hidden="true" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleActivate}
              disabled={activating}
              aria-label={`Activate ${profile.name} profile`}
            >
              <Play className="h-4 w-4" aria-hidden="true" />
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
                <Label htmlFor={`edit-temp-${profile.id}`} className="text-xs">Temperature</Label>
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
                <Label htmlFor={`edit-topp-${profile.id}`} className="text-xs">Top P</Label>
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
                <Label htmlFor={`edit-maxtokens-${profile.id}`} className="text-xs">Max Tokens</Label>
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
                <Label htmlFor={`edit-timeout-${profile.id}`} className="text-xs">Timeout (s)</Label>
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
                <Label htmlFor={`edit-steps-${profile.id}`} className="text-xs">Max Steps</Label>
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
                <Label htmlFor={`edit-cost-${profile.id}`} className="text-xs">Max Cost ($)</Label>
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
          <p className="text-xs text-green-600 dark:text-green-400 mt-2" role="status">
            {message}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
