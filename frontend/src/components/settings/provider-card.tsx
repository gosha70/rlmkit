"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { StatusIndicator } from "@/components/shared/status-indicator";
import { ModelSelector } from "./model-selector";
import { ConnectionTester } from "./connection-tester";
import {
  saveProvider,
  type ProviderInfo,
  type RuntimeSettings,
} from "@/lib/api";
import { Save, ChevronDown, ChevronRight, Settings2 } from "lucide-react";

type ProviderStatus = "connected" | "configured" | "offline" | "not_configured";

function mapStatus(status: string): ProviderStatus {
  if (status === "connected") return "connected";
  if (status === "configured") return "configured";
  if (status === "not_configured") return "not_configured";
  return "offline";
}

const DEFAULT_RUNTIME: RuntimeSettings = {
  temperature: 0.7,
  top_p: 1.0,
  max_output_tokens: 4096,
  timeout_seconds: 30,
};

interface ProviderCardProps {
  provider: ProviderInfo;
  savedConfig?: { provider: string; model: string; runtime_settings: RuntimeSettings; enabled: boolean } | null;
  onProviderSaved?: () => void;
}

export function ProviderCard({ provider, savedConfig, onProviderSaved }: ProviderCardProps) {
  const [selectedModel, setSelectedModel] = useState(savedConfig?.model || provider.default_model);
  const [apiKey, setApiKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [enabled, setEnabled] = useState(savedConfig?.enabled ?? false);
  const [endpoint, setEndpoint] = useState(provider.default_endpoint || "");
  const [runtimeOpen, setRuntimeOpen] = useState(false);
  const [runtime, setRuntime] = useState<RuntimeSettings>(savedConfig?.runtime_settings ?? { ...DEFAULT_RUNTIME });

  // Sync state when savedConfig loads asynchronously from SWR
  useEffect(() => {
    if (savedConfig) {
      setSelectedModel(savedConfig.model);
      setEnabled(savedConfig.enabled);
      setRuntime(savedConfig.runtime_settings);
    }
  }, [savedConfig]);

  const isLocal = !provider.requires_api_key;

  const handleSave = async () => {
    setSaving(true);
    setSaveMessage(null);
    try {
      const res = await saveProvider(provider.name, {
        api_key: apiKey || null,
        model: selectedModel,
        runtime_settings: runtime,
        enabled,
      });
      setSaveMessage(res.message);
      setApiKey("");
      onProviderSaved?.();
    } catch (err) {
      console.error("Failed to save provider:", err);
      setSaveMessage("Failed to save configuration");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <CardTitle className="text-base">{provider.display_name}</CardTitle>
            <StatusIndicator status={mapStatus(provider.status)} />
          </div>
          <div className="flex items-center gap-2">
            <Label htmlFor={`enable-${provider.name}`} className="text-sm text-muted-foreground">
              Enable for Chat
            </Label>
            <Switch
              id={`enable-${provider.name}`}
              checked={enabled}
              onCheckedChange={setEnabled}
              aria-label={`Enable ${provider.display_name} for chat`}
            />
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Model Selection */}
        <div className="space-y-2">
          <Label>Model</Label>
          <ModelSelector
            models={provider.models}
            value={selectedModel}
            onChange={setSelectedModel}
            freeText={isLocal}
            placeholder={provider.model_input_hint || "Select model"}
          />
        </div>

        {/* API Key (only for cloud providers) */}
        {provider.requires_api_key && (
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
                onClick={handleSave}
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
        )}

        {/* Endpoint (for local providers) */}
        {isLocal && provider.default_endpoint && (
          <div className="space-y-2">
            <Label htmlFor={`endpoint-${provider.name}`}>Endpoint</Label>
            <Input
              id={`endpoint-${provider.name}`}
              value={endpoint}
              onChange={(e) => setEndpoint(e.target.value)}
              placeholder={provider.default_endpoint}
              aria-label={`Endpoint for ${provider.display_name}`}
            />
            <p className="text-xs text-muted-foreground">
              Default: {provider.default_endpoint}
            </p>
          </div>
        )}

        {/* Runtime Settings (collapsible) */}
        <Collapsible open={runtimeOpen} onOpenChange={setRuntimeOpen}>
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="sm" className="w-full justify-start gap-2 px-0">
              <Settings2 className="h-4 w-4" aria-hidden="true" />
              Runtime Settings
              {runtimeOpen ? (
                <ChevronDown className="ml-auto h-4 w-4" aria-hidden="true" />
              ) : (
                <ChevronRight className="ml-auto h-4 w-4" aria-hidden="true" />
              )}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-4 pt-2">
            {/* Temperature */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Temperature</Label>
                <span className="text-sm text-muted-foreground">{runtime.temperature.toFixed(1)}</span>
              </div>
              <Slider
                value={[runtime.temperature]}
                onValueChange={([v]) => setRuntime({ ...runtime, temperature: v })}
                min={0}
                max={2}
                step={0.1}
                aria-label="Temperature"
              />
              <p className="text-xs text-muted-foreground">
                Lower = more deterministic, higher = more creative
              </p>
            </div>

            {/* Top-P */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Top-P</Label>
                <span className="text-sm text-muted-foreground">{runtime.top_p.toFixed(2)}</span>
              </div>
              <Slider
                value={[runtime.top_p]}
                onValueChange={([v]) => setRuntime({ ...runtime, top_p: v })}
                min={0}
                max={1}
                step={0.05}
                aria-label="Top P"
              />
            </div>

            {/* Max Output Tokens */}
            <div className="space-y-2">
              <Label htmlFor={`max-tokens-${provider.name}`}>Max Output Tokens</Label>
              <Input
                id={`max-tokens-${provider.name}`}
                type="number"
                value={runtime.max_output_tokens}
                onChange={(e) =>
                  setRuntime({ ...runtime, max_output_tokens: parseInt(e.target.value) || 4096 })
                }
                min={100}
                max={128000}
                aria-label="Max output tokens"
              />
            </div>

            {/* Timeout */}
            <div className="space-y-2">
              <Label htmlFor={`timeout-${provider.name}`}>Timeout (seconds)</Label>
              <Input
                id={`timeout-${provider.name}`}
                type="number"
                value={runtime.timeout_seconds}
                onChange={(e) =>
                  setRuntime({ ...runtime, timeout_seconds: parseInt(e.target.value) || 30 })
                }
                min={1}
                max={300}
                aria-label="Timeout in seconds"
              />
            </div>
          </CollapsibleContent>
        </Collapsible>

        {/* Save configuration (model, runtime, enabled) */}
        <Button
          variant="outline"
          size="sm"
          onClick={handleSave}
          disabled={saving}
          className="w-full"
        >
          <Save className="mr-1 h-4 w-4" aria-hidden="true" />
          {saving ? "Saving..." : "Save Configuration"}
        </Button>
        {saveMessage && (
          <p className="text-xs text-green-600 dark:text-green-400" role="status">
            {saveMessage}
          </p>
        )}

        <ConnectionTester
          provider={provider.name}
          apiKey={apiKey || null}
          endpoint={endpoint || null}
          model={selectedModel}
          onResult={() => onProviderSaved?.()}
        />
      </CardContent>
    </Card>
  );
}
