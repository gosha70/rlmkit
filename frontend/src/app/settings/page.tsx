"use client";

import { useState } from "react";
import useSWR from "swr";
import { useTheme } from "next-themes";
import { AppShell } from "@/components/shared/app-shell";
import { ProviderCard } from "@/components/settings/provider-card";
import { BudgetConfig } from "@/components/settings/budget-config";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import {
  getConfig,
  getProviders,
  updateConfig,
  type AppConfig,
  type ProviderInfo,
} from "@/lib/api";

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const [saving, setSaving] = useState(false);

  const { data: config, mutate: mutateConfig } = useSWR<AppConfig>("config", getConfig);
  const { data: providers = [] } = useSWR<ProviderInfo[]>("providers", getProviders);

  const handleSaveBudget = async (budget: AppConfig["budget"]) => {
    if (!config) return;
    setSaving(true);
    try {
      const updated = await updateConfig({ budget });
      mutateConfig(updated, false);
    } catch (err) {
      console.error("Failed to save budget:", err);
    } finally {
      setSaving(false);
    }
  };

  const handleProviderModelChange = async (provider: string, model: string) => {
    if (!config) return;
    try {
      const updated = await updateConfig({
        active_provider: provider,
        active_model: model,
      });
      mutateConfig(updated, false);
    } catch (err) {
      console.error("Failed to update provider/model:", err);
    }
  };

  const handleThemeChange = async (newTheme: string) => {
    setTheme(newTheme);
    try {
      await updateConfig({ appearance: { theme: newTheme, sidebar_collapsed: false } });
    } catch (err) {
      console.error("Failed to save theme preference:", err);
    }
  };

  return (
    <AppShell>
      <div className="mx-auto max-w-[1200px] space-y-6 p-6">
        <h2 className="text-2xl font-semibold">Settings</h2>

        <Tabs defaultValue="providers">
          <TabsList>
            <TabsTrigger value="providers">Providers</TabsTrigger>
            <TabsTrigger value="budget">Budget</TabsTrigger>
            <TabsTrigger value="appearance">Appearance</TabsTrigger>
          </TabsList>

          <TabsContent value="providers" className="space-y-4">
            {providers.map((p) => (
              <ProviderCard
                key={p.name}
                provider={p}
                onSave={handleProviderModelChange}
              />
            ))}
            {providers.length === 0 && (
              <Card>
                <CardContent className="py-10 text-center text-muted-foreground">
                  No providers available. Start the backend server first.
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="budget">
            {config ? (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Budget Limits</CardTitle>
                </CardHeader>
                <CardContent>
                  <BudgetConfig config={config.budget} onChange={handleSaveBudget} />
                  {saving && (
                    <p className="mt-2 text-sm text-muted-foreground">Saving...</p>
                  )}
                </CardContent>
              </Card>
            ) : (
              <Card>
                <CardContent className="py-10 text-center text-muted-foreground">
                  Loading configuration...
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="appearance">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Theme</CardTitle>
              </CardHeader>
              <CardContent>
                <RadioGroup
                  value={theme ?? "system"}
                  onValueChange={handleThemeChange}
                  className="flex gap-4"
                >
                  {(["system", "light", "dark"] as const).map((t) => (
                    <div key={t} className="flex items-center gap-2">
                      <RadioGroupItem value={t} id={`theme-${t}`} />
                      <Label htmlFor={`theme-${t}`} className="capitalize">
                        {t}
                      </Label>
                    </div>
                  ))}
                </RadioGroup>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </AppShell>
  );
}
