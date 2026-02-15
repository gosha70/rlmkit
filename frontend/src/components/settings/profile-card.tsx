"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { activateProfile, deleteProfile, type RunProfile } from "@/lib/api";
import { Play, Trash2, Lock } from "lucide-react";

interface ProfileCardProps {
  profile: RunProfile;
  onActivated?: () => void;
  onDeleted?: () => void;
}

export function ProfileCard({ profile, onActivated, onDeleted }: ProfileCardProps) {
  const [activating, setActivating] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

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
    setDeleting(true);
    try {
      await deleteProfile(profile.id);
      onDeleted?.();
    } catch {
      setMessage("Failed to delete profile");
      setDeleting(false);
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
        {message && (
          <p className="text-xs text-green-600 dark:text-green-400 mt-2" role="status">
            {message}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
