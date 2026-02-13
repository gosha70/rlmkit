"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { StatusIndicator } from "@/components/shared/status-indicator";
import { ModelSelector } from "./model-selector";
import { ConnectionTester } from "./connection-tester";
import { saveProvider, type ProviderInfo } from "@/lib/api";
import { Save } from "lucide-react";

type ProviderStatus = "connected" | "configured" | "offline" | "not_configured";

function mapStatus(status: string): ProviderStatus {
  if (status === "connected") return "connected";
  if (status === "configured") return "configured";
  if (status === "not_configured") return "not_configured";
  return "offline";
}

interface ProviderCardProps {
  provider: ProviderInfo;
  onSave?: (provider: string, model: string) => void;
  onProviderSaved?: () => void;
}

export function ProviderCard({ provider, onSave, onProviderSaved }: ProviderCardProps) {
  const [selectedModel, setSelectedModel] = useState(provider.default_model);
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  const handleSaveApiKey = async () => {
    if (!apiKey.trim()) return;
    setSaving(true);
    setSaveMessage(null);
    try {
      const res = await saveProvider(provider.name, {
        api_key: apiKey,
        model: selectedModel,
      });
      setSaveMessage(res.message);
      setApiKey("");
      onProviderSaved?.();
    } catch (err) {
      console.error("Failed to save provider:", err);
      setSaveMessage("Failed to save API key");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{provider.display_name}</CardTitle>
          <StatusIndicator status={mapStatus(provider.status)} />
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label>Model</Label>
          <ModelSelector
            models={provider.models.length > 0 ? provider.models : getDefaultModels(provider.name)}
            value={selectedModel}
            onChange={(model) => {
              setSelectedModel(model);
              onSave?.(provider.name, model);
            }}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor={`api-key-${provider.name}`}>API Key</Label>
          <div className="flex gap-2">
            <Input
              id={`api-key-${provider.name}`}
              type="password"
              placeholder={provider.configured ? "Configured — enter new key to override" : "Enter API key"}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              aria-label={`API key for ${provider.display_name}`}
              className="flex-1"
            />
            <Button
              variant="outline"
              size="sm"
              onClick={handleSaveApiKey}
              disabled={saving || !apiKey.trim()}
              aria-label={`Save API key for ${provider.display_name}`}
              className="shrink-0"
            >
              <Save className="mr-1 h-4 w-4" aria-hidden="true" />
              {saving ? "Saving..." : "Save"}
            </Button>
          </div>
          {saveMessage && (
            <p className="text-xs text-green-600 dark:text-green-400" role="status">
              {saveMessage}
            </p>
          )}
          <p className="text-xs text-muted-foreground">
            {provider.configured
              ? "Environment variable detected. Enter a new key to override and save to .env."
              : "Enter your API key. It will be saved to the .env file."}
          </p>
        </div>

        <ConnectionTester
          provider={provider.name}
          apiKey={apiKey || null}
          model={selectedModel}
          onResult={() => onProviderSaved?.()}
        />
      </CardContent>
    </Card>
  );
}

function getDefaultModels(providerName: string): string[] {
  const defaults: Record<string, string[]> = {
    openai: ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    anthropic: ["claude-opus-4-6", "claude-sonnet-4-5-20250929"],
    ollama: ["llama3", "codellama", "mistral"],
  };
  return defaults[providerName] ?? [];
}
