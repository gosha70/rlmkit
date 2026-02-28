"use client";

import { useState } from "react";
import useSWR from "swr";
import { useTheme } from "next-themes";
import { AppShell } from "@/components/shared/app-shell";
import { ProviderCard } from "@/components/settings/provider-card";
import { BudgetConfig } from "@/components/settings/budget-config";
import { ProfileCard } from "@/components/settings/profile-card";
import { SystemPromptEditor } from "@/components/settings/system-prompt-editor";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Button } from "@/components/ui/button";
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
  getConfig,
  getProviders,
  getProfiles,
  createProfile,
  updateConfig,
  getChatProviders,
  createChatProvider,
  updateChatProvider,
  deleteChatProvider,
  type AppConfig,
  type ProviderInfo,
  type RunProfile,
  type ChatProviderConfig,
  type ChatProviderCreateRequest,
  type RuntimeSettings,
  type RAGConfig,
} from "@/lib/api";
import { Plus, Edit2, Trash2 } from "lucide-react";

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const [saving, setSaving] = useState(false);
  const [newProfileName, setNewProfileName] = useState("");
  const [creatingProfile, setCreatingProfile] = useState(false);

  // Chat Providers state
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [formData, setFormData] = useState<{
    name: string;
    llm_provider: string;
    llm_model: string;
    execution_mode: "direct" | "rlm" | "rag";
    runtime_settings: RuntimeSettings;
    rag_config: RAGConfig;
    rlm_max_steps: number;
    rlm_timeout_seconds: number;
  }>({
    name: "",
    llm_provider: "",
    llm_model: "",
    execution_mode: "direct",
    runtime_settings: {
      temperature: 0.7,
      top_p: 0.9,
      max_output_tokens: 2048,
      timeout_seconds: 30,
    },
    rag_config: {
      chunk_size: 512,
      chunk_overlap: 64,
      top_k: 5,
      embedding_model: "text-embedding-3-small",
    },
    rlm_max_steps: 10,
    rlm_timeout_seconds: 60,
  });
  const [savingProvider, setSavingProvider] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const { data: config, mutate: mutateConfig } = useSWR<AppConfig>("config", getConfig);
  const { data: providers = [], mutate: mutateProviders } = useSWR<ProviderInfo[]>("providers", getProviders);
  const { data: profiles = [], mutate: mutateProfiles } = useSWR<RunProfile[]>("profiles", getProfiles);
  const { data: chatProviders = [], mutate: mutateChatProviders } = useSWR<ChatProviderConfig[]>(
    "chat-providers",
    getChatProviders
  );

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

  const handleProviderSaved = () => {
    mutateProviders();
    mutateConfig();
  };

  const handleThemeChange = async (newTheme: string) => {
    setTheme(newTheme);
    try {
      await updateConfig({ appearance: { theme: newTheme, sidebar_collapsed: false } });
    } catch (err) {
      console.error("Failed to save theme preference:", err);
    }
  };

  const handleCreateProfile = async () => {
    if (!newProfileName.trim()) return;
    setCreatingProfile(true);
    try {
      await createProfile({ name: newProfileName.trim() });
      setNewProfileName("");
      mutateProfiles();
    } catch (err) {
      console.error("Failed to create profile:", err);
    } finally {
      setCreatingProfile(false);
    }
  };

  const handleEditChatProvider = (provider: ChatProviderConfig) => {
    setEditingId(provider.id);
    setFormData({
      name: provider.name,
      llm_provider: provider.llm_provider,
      llm_model: provider.llm_model,
      execution_mode: provider.execution_mode,
      runtime_settings: provider.runtime_settings,
      rag_config: provider.rag_config || {
        chunk_size: 512,
        chunk_overlap: 64,
        top_k: 5,
        embedding_model: "text-embedding-3-small",
      },
      rlm_max_steps: provider.rlm_max_steps,
      rlm_timeout_seconds: provider.rlm_timeout_seconds,
    });
    setShowCreateForm(true);
  };

  const handleSaveChatProvider = async () => {
    if (!formData.name.trim() || !formData.llm_provider || !formData.llm_model) {
      console.error("Missing required fields");
      return;
    }

    setSavingProvider(true);
    try {
      const payload: ChatProviderCreateRequest = {
        name: formData.name,
        llm_provider: formData.llm_provider,
        llm_model: formData.llm_model,
        execution_mode: formData.execution_mode,
        runtime_settings: formData.runtime_settings,
        rlm_max_steps: formData.rlm_max_steps,
        rlm_timeout_seconds: formData.rlm_timeout_seconds,
      };

      if (formData.execution_mode === "rag") {
        payload.rag_config = formData.rag_config;
      }

      if (editingId) {
        await updateChatProvider(editingId, payload);
      } else {
        await createChatProvider(payload);
      }

      setShowCreateForm(false);
      setEditingId(null);
      resetForm();
      mutateChatProviders();
    } catch (err) {
      console.error("Failed to save chat provider:", err);
    } finally {
      setSavingProvider(false);
    }
  };

  const handleDeleteChatProvider = async (id: string) => {
    if (!confirm("Are you sure you want to delete this Chat Provider?")) return;

    setDeletingId(id);
    try {
      await deleteChatProvider(id);
      mutateChatProviders();
    } catch (err) {
      console.error("Failed to delete chat provider:", err);
    } finally {
      setDeletingId(null);
    }
  };

  const resetForm = () => {
    setFormData({
      name: "",
      llm_provider: "",
      llm_model: "",
      execution_mode: "direct",
      runtime_settings: {
        temperature: 0.7,
        top_p: 0.9,
        max_output_tokens: 2048,
        timeout_seconds: 30,
      },
      rag_config: {
        chunk_size: 512,
        chunk_overlap: 64,
        top_k: 5,
        embedding_model: "text-embedding-3-small",
      },
      rlm_max_steps: 10,
      rlm_timeout_seconds: 60,
    });
  };

  return (
    <AppShell>
      <div className="mx-auto max-w-[1200px] space-y-6 p-6">
        <h2 className="text-2xl font-semibold">Settings</h2>

        <Tabs defaultValue="providers">
          <TabsList>
            <TabsTrigger value="providers">Providers</TabsTrigger>
            <TabsTrigger value="chat-providers">Chat Providers</TabsTrigger>
            <TabsTrigger value="budget">Budget</TabsTrigger>
            <TabsTrigger value="profiles">Profiles</TabsTrigger>
            <TabsTrigger value="prompts">Prompts</TabsTrigger>
            <TabsTrigger value="appearance">Appearance</TabsTrigger>
          </TabsList>

          <TabsContent value="providers" className="space-y-4">
            {providers.map((p) => {
              const savedConfig = config?.provider_configs?.find(
                (pc) => pc.provider === p.name,
              );
              return (
                <ProviderCard
                  key={p.name}
                  provider={p}
                  savedConfig={savedConfig}
                  onProviderSaved={handleProviderSaved}
                />
              );
            })}
            {providers.length === 0 && (
              <Card>
                <CardContent className="py-10 text-center text-muted-foreground">
                  No providers available. Start the backend server first.
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="chat-providers" className="space-y-4">
            {!showCreateForm && (
              <Button
                onClick={() => {
                  setEditingId(null);
                  resetForm();
                  setShowCreateForm(true);
                }}
                size="sm"
                className="mb-4"
              >
                <Plus className="mr-2 h-4 w-4" aria-hidden="true" />
                Create Chat Provider
              </Button>
            )}

            {showCreateForm && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">
                    {editingId ? "Edit Chat Provider" : "Create Chat Provider"}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="cp-name">Name</Label>
                    <Input
                      id="cp-name"
                      placeholder="e.g., My GPT-4 Config"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="cp-provider">LLM Provider</Label>
                    <Select value={formData.llm_provider} onValueChange={(value) => {
                      setFormData({ ...formData, llm_provider: value, llm_model: "" });
                    }}>
                      <SelectTrigger id="cp-provider">
                        <SelectValue placeholder="Select a provider" />
                      </SelectTrigger>
                      <SelectContent>
                        {providers.map((p) => (
                          <SelectItem key={p.name} value={p.name}>
                            {p.display_name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="cp-model">Model</Label>
                    <Select
                      value={formData.llm_model}
                      onValueChange={(value) => setFormData({ ...formData, llm_model: value })}
                      disabled={!formData.llm_provider}
                    >
                      <SelectTrigger id="cp-model">
                        <SelectValue placeholder="Select a model" />
                      </SelectTrigger>
                      <SelectContent>
                        {formData.llm_provider &&
                          providers
                            .find((p) => p.name === formData.llm_provider)
                            ?.models.map((m) => (
                              <SelectItem key={m.name} value={m.name}>
                                {m.name}
                              </SelectItem>
                            ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-3">
                    <Label>Execution Mode</Label>
                    <RadioGroup
                      value={formData.execution_mode}
                      onValueChange={(value) =>
                        setFormData({
                          ...formData,
                          execution_mode: value as "direct" | "rlm" | "rag",
                        })
                      }
                    >
                      {(["direct", "rlm", "rag"] as const).map((mode) => (
                        <div key={mode} className="flex items-center gap-2">
                          <RadioGroupItem value={mode} id={`mode-${mode}`} />
                          <Label htmlFor={`mode-${mode}`} className="capitalize cursor-pointer">
                            {mode}
                          </Label>
                        </div>
                      ))}
                    </RadioGroup>
                  </div>

                  {formData.execution_mode === "rlm" && (
                    <div className="space-y-4 rounded-lg border border-muted p-4">
                      <h4 className="font-medium text-sm">RLM Settings</h4>
                      <div className="space-y-2">
                        <Label htmlFor="cp-rlm-steps">Max Steps</Label>
                        <Input
                          id="cp-rlm-steps"
                          type="number"
                          min="1"
                          value={formData.rlm_max_steps}
                          onChange={(e) =>
                            setFormData({
                              ...formData,
                              rlm_max_steps: parseInt(e.target.value) || 10,
                            })
                          }
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="cp-rlm-timeout">Timeout (seconds)</Label>
                        <Input
                          id="cp-rlm-timeout"
                          type="number"
                          min="1"
                          value={formData.rlm_timeout_seconds}
                          onChange={(e) =>
                            setFormData({
                              ...formData,
                              rlm_timeout_seconds: parseInt(e.target.value) || 60,
                            })
                          }
                        />
                      </div>
                    </div>
                  )}

                  {formData.execution_mode === "rag" && (
                    <div className="space-y-4 rounded-lg border border-muted p-4">
                      <h4 className="font-medium text-sm">RAG Settings</h4>
                      <div className="space-y-2">
                        <Label htmlFor="cp-chunk-size">Chunk Size</Label>
                        <Input
                          id="cp-chunk-size"
                          type="number"
                          min="1"
                          value={formData.rag_config.chunk_size}
                          onChange={(e) =>
                            setFormData({
                              ...formData,
                              rag_config: {
                                ...formData.rag_config,
                                chunk_size: parseInt(e.target.value) || 512,
                              },
                            })
                          }
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="cp-chunk-overlap">Chunk Overlap</Label>
                        <Input
                          id="cp-chunk-overlap"
                          type="number"
                          min="0"
                          value={formData.rag_config.chunk_overlap}
                          onChange={(e) =>
                            setFormData({
                              ...formData,
                              rag_config: {
                                ...formData.rag_config,
                                chunk_overlap: parseInt(e.target.value) || 64,
                              },
                            })
                          }
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="cp-top-k">Top K</Label>
                        <Input
                          id="cp-top-k"
                          type="number"
                          min="1"
                          value={formData.rag_config.top_k}
                          onChange={(e) =>
                            setFormData({
                              ...formData,
                              rag_config: {
                                ...formData.rag_config,
                                top_k: parseInt(e.target.value) || 5,
                              },
                            })
                          }
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="cp-embedding-model">Embedding Model</Label>
                        <Input
                          id="cp-embedding-model"
                          placeholder="e.g., text-embedding-3-small"
                          value={formData.rag_config.embedding_model}
                          onChange={(e) =>
                            setFormData({
                              ...formData,
                              rag_config: {
                                ...formData.rag_config,
                                embedding_model: e.target.value,
                              },
                            })
                          }
                        />
                      </div>
                    </div>
                  )}

                  <div className="space-y-4 rounded-lg border border-muted p-4">
                    <h4 className="font-medium text-sm">Runtime Settings</h4>
                    <div className="space-y-2">
                      <Label htmlFor="cp-temperature">Temperature</Label>
                      <Input
                        id="cp-temperature"
                        type="number"
                        min="0"
                        max="2"
                        step="0.1"
                        value={formData.runtime_settings.temperature}
                        onChange={(e) =>
                          setFormData({
                            ...formData,
                            runtime_settings: {
                              ...formData.runtime_settings,
                              temperature: parseFloat(e.target.value) || 0.7,
                            },
                          })
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="cp-top-p">Top P</Label>
                      <Input
                        id="cp-top-p"
                        type="number"
                        min="0"
                        max="1"
                        step="0.1"
                        value={formData.runtime_settings.top_p}
                        onChange={(e) =>
                          setFormData({
                            ...formData,
                            runtime_settings: {
                              ...formData.runtime_settings,
                              top_p: parseFloat(e.target.value) || 0.9,
                            },
                          })
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="cp-max-tokens">Max Output Tokens</Label>
                      <Input
                        id="cp-max-tokens"
                        type="number"
                        min="1"
                        value={formData.runtime_settings.max_output_tokens}
                        onChange={(e) =>
                          setFormData({
                            ...formData,
                            runtime_settings: {
                              ...formData.runtime_settings,
                              max_output_tokens: parseInt(e.target.value) || 2048,
                            },
                          })
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="cp-timeout">Timeout (seconds)</Label>
                      <Input
                        id="cp-timeout"
                        type="number"
                        min="1"
                        value={formData.runtime_settings.timeout_seconds}
                        onChange={(e) =>
                          setFormData({
                            ...formData,
                            runtime_settings: {
                              ...formData.runtime_settings,
                              timeout_seconds: parseInt(e.target.value) || 30,
                            },
                          })
                        }
                      />
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <Button
                      onClick={handleSaveChatProvider}
                      disabled={
                        savingProvider ||
                        !formData.name.trim() ||
                        !formData.llm_provider ||
                        !formData.llm_model
                      }
                    >
                      {editingId ? "Update" : "Create"}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => {
                        setShowCreateForm(false);
                        setEditingId(null);
                        resetForm();
                      }}
                      disabled={savingProvider}
                    >
                      Cancel
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {chatProviders.map((provider) => (
              <Card key={provider.id}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="space-y-1">
                      <CardTitle className="text-base">{provider.name}</CardTitle>
                      <p className="text-sm text-muted-foreground">
                        {provider.llm_provider} · {provider.llm_model}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="inline-flex items-center rounded-full bg-secondary px-2.5 py-0.5 text-xs font-medium text-secondary-foreground capitalize">
                        {provider.execution_mode}
                      </span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleEditChatProvider(provider)}
                        disabled={showCreateForm}
                      >
                        <Edit2 className="h-4 w-4" aria-hidden="true" />
                        <span className="sr-only">Edit</span>
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleDeleteChatProvider(provider.id)}
                        disabled={deletingId === provider.id}
                      >
                        <Trash2 className="h-4 w-4" aria-hidden="true" />
                        <span className="sr-only">Delete</span>
                      </Button>
                    </div>
                  </div>
                </CardHeader>
              </Card>
            ))}

            {chatProviders.length === 0 && !showCreateForm && (
              <Card>
                <CardContent className="py-10 text-center text-muted-foreground">
                  No Chat Providers configured yet. Create one to get started.
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

          <TabsContent value="profiles" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Create Profile</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex gap-2">
                  <Input
                    placeholder="Profile name"
                    value={newProfileName}
                    onChange={(e) => setNewProfileName(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleCreateProfile()}
                    aria-label="New profile name"
                    className="flex-1"
                  />
                  <Button
                    onClick={handleCreateProfile}
                    disabled={creatingProfile || !newProfileName.trim()}
                    size="sm"
                  >
                    <Plus className="mr-1 h-4 w-4" aria-hidden="true" />
                    Create
                  </Button>
                </div>
              </CardContent>
            </Card>

            {profiles.map((profile) => (
              <ProfileCard
                key={profile.id}
                profile={profile}
                onActivated={() => {
                  mutateConfig();
                  mutateProfiles();
                }}
                onDeleted={() => mutateProfiles()}
              />
            ))}

            {profiles.length === 0 && (
              <Card>
                <CardContent className="py-10 text-center text-muted-foreground">
                  No profiles yet. Create one above or use built-in presets.
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="prompts">
            <SystemPromptEditor />
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
                  aria-label="Theme selection"
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
