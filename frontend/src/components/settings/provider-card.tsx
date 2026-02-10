"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { StatusIndicator } from "@/components/shared/status-indicator";
import { ModelSelector } from "./model-selector";
import { ConnectionTester } from "./connection-tester";
import type { ProviderInfo } from "@/lib/api";

type ProviderStatus = "connected" | "offline" | "not_configured";

function mapStatus(status: string): ProviderStatus {
  if (status === "connected") return "connected";
  if (status === "not_configured") return "not_configured";
  return "offline";
}

interface ProviderCardProps {
  provider: ProviderInfo;
  onSave?: (provider: string, model: string) => void;
}

export function ProviderCard({ provider, onSave }: ProviderCardProps) {
  const [selectedModel, setSelectedModel] = useState(provider.default_model);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{provider.display_name}</CardTitle>
          <StatusIndicator status={mapStatus(provider.status)} />
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {provider.configured && provider.models.length > 0 && (
          <div className="space-y-2">
            <Label>Model</Label>
            <ModelSelector
              models={provider.models}
              value={selectedModel}
              onChange={(model) => {
                setSelectedModel(model);
                onSave?.(provider.name, model);
              }}
            />
          </div>
        )}

        <div className="space-y-2">
          <Label>API Key</Label>
          <Input type="password" placeholder="Set via environment variable" disabled readOnly />
          <p className="text-xs text-muted-foreground">
            API keys are read from environment variables for security.
          </p>
        </div>

        <ConnectionTester
          provider={provider.name}
          model={selectedModel}
        />
      </CardContent>
    </Card>
  );
}
