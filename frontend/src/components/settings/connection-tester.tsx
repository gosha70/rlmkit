"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { StatusIndicator } from "@/components/shared/status-indicator";
import { testProvider, type ProviderTestResponse } from "@/lib/api";

interface ConnectionTesterProps {
  provider: string;
  apiKey?: string | null;
  endpoint?: string | null;
  model?: string | null;
  onResult?: (result: ProviderTestResponse) => void;
}

export function ConnectionTester({ provider, apiKey, endpoint, model, onResult }: ConnectionTesterProps) {
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<ProviderTestResponse | null>(null);

  const handleTest = async () => {
    setTesting(true);
    setResult(null);
    try {
      const res = await testProvider({ provider, api_key: apiKey, endpoint, model });
      setResult(res);
      onResult?.(res);
    } catch {
      setResult({ connected: false, error: "Request failed" });
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="flex items-center gap-3">
      <Button variant="outline" size="sm" onClick={handleTest} disabled={testing}>
        {testing ? "Testing..." : "Test Connection"}
      </Button>
      {testing && <StatusIndicator status="loading" label="Testing" />}
      {!testing && result && (
        <StatusIndicator
          status={result.connected ? "connected" : "offline"}
          label={
            result.connected
              ? `Connected${result.latency_ms ? `, ${result.latency_ms}ms` : ""}`
              : result.error ?? "Failed"
          }
        />
      )}
    </div>
  );
}
