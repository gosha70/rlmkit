"use client";

import { useEffect, useRef, useState } from "react";
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
  getProviderModels,
  type AppConfig,
  type ProviderInfo,
  type ModelInfo,
  type RunProfile,
  type ChatProviderConfig,
  type ChatProviderCreateRequest,
  type RAGConfig,
} from "@/lib/api";
import { Plus, Edit2, Trash2, Upload, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const [saving, setSaving] = useState(false);
  const [newProfileName, setNewProfileName] = useState("");
  const [newProfileStrategy, setNewProfileStrategy] = useState("direct");
  const [newProfileDescription, setNewProfileDescription] = useState("");
  const [creatingProfile, setCreatingProfile] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);
  const importFileRef = useRef<HTMLInputElement>(null);

  // Chat Providers state
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [formData, setFormData] = useState<{
    name: string;
    llm_provider: string;
    llm_model: string;
    profile_id: string;
    rag_config: RAGConfig;
  }>({
    name: "",
    llm_provider: "",
    llm_model: "",
    profile_id: "",
    rag_config: {
      chunk_size: 512,
      chunk_overlap: 64,
      top_k: 5,
      embedding_model: "text-embedding-3-small",
    },
  });
  const [savingProvider, setSavingProvider] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [dynamicModels, setDynamicModels] = useState<ModelInfo[]>([]);
  const [refreshingModels, setRefreshingModels] = useState(false);

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
    const duplicate = profiles.some(
      (p) => p.name.toLowerCase() === newProfileName.trim().toLowerCase(),
    );
    if (duplicate) {
      setProfileError(`Profile "${newProfileName.trim()}" already exists`);
      return;
    }
    setProfileError(null);
    setCreatingProfile(true);
    try {
      await createProfile({
        name: newProfileName.trim(),
        strategy: newProfileStrategy,
        description: newProfileDescription.trim() || undefined,
      });
      setNewProfileName("");
      setNewProfileStrategy("direct");
      setNewProfileDescription("");
      mutateProfiles();
    } catch (err) {
      console.error("Failed to create profile:", err);
      toast.error("Failed to create profile");
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
      profile_id: provider.profile_id || "",
      rag_config: provider.rag_config || {
        chunk_size: 512,
        chunk_overlap: 64,
        top_k: 5,
        embedding_model: "text-embedding-3-small",
      },
    });
  };

  const handleSaveChatProvider = async () => {
    if (!formData.name.trim() || !formData.llm_provider || !formData.llm_model || !formData.profile_id) {
      console.error("Missing required fields");
      return;
    }

    setSavingProvider(true);
    try {
      const selectedProfile = profiles.find((p) => p.id === formData.profile_id);
      const payload: ChatProviderCreateRequest = {
        name: formData.name,
        llm_provider: formData.llm_provider,
        llm_model: formData.llm_model,
        profile_id: formData.profile_id,
      };

      if (selectedProfile?.strategy === "rag") {
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
      toast.error("Failed to save Chat Provider");
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
      // Backend clears judge_chat_provider_id when the deleted provider was
      // the active judge — revalidate config so the UI reflects this immediately.
      if (config?.judge_chat_provider_id === id) {
        mutateConfig();
      }
    } catch (err) {
      console.error("Failed to delete chat provider:", err);
      toast.error("Failed to delete Chat Provider");
    } finally {
      setDeletingId(null);
    }
  };

  const handleImportProfile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setProfileError(null);
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      // Auto-rename if name already exists
      let name = data.name || "Imported Profile";
      const nameExists = (n: string) =>
        profiles.some((p) => p.name.toLowerCase() === n.toLowerCase());
      if (nameExists(name)) {
        let i = 2;
        while (nameExists(`${name} (${i})`)) i++;
        name = `${name} (${i})`;
      }
      await createProfile({
        name,
        description: data.description || "",
        strategy: data.strategy || "direct",
        runtime_settings: data.runtime_settings,
        budget: data.budget,
        system_prompts: data.system_prompts || {},
      });
      mutateProfiles();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to import profile";
      setProfileError(msg);
    }
    // Reset file input so same file can be re-imported
    e.target.value = "";
  };

  const fetchModelsForProvider = async (providerName: string) => {
    if (!providerName) return;
    setRefreshingModels(true);
    try {
      // Pass the provider's endpoint so local providers on non-default ports work
      const providerInfo = providers.find((p) => p.name === providerName);
      const models = await getProviderModels(
        providerName,
        providerInfo?.default_endpoint ?? undefined,
      );
      setDynamicModels(models);
    } catch (err) {
      console.error("Failed to fetch models:", err);
    } finally {
      setRefreshingModels(false);
    }
  };

  const handleRefreshModels = () => fetchModelsForProvider(formData.llm_provider);

  // Auto-fetch models when a provider is selected that has no static model list
  // (local providers like Ollama/LMStudio whose models vary per installation).
  useEffect(() => {
    if (!formData.llm_provider || providers.length === 0) return;
    const providerInfo = providers.find((p) => p.name === formData.llm_provider);
    if (!providerInfo) return;
    // Auto-fetch for local providers (no API key) or providers with no static models
    if (!providerInfo.requires_api_key || providerInfo.models.length === 0) {
      fetchModelsForProvider(formData.llm_provider);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [formData.llm_provider, providers.length]);

  // Resolve the selected profile for the form
  const selectedProfile = profiles.find((p) => p.id === formData.profile_id);

  const resetForm = () => {
    setFormData({
      name: "",
      llm_provider: "",
      llm_model: "",
      profile_id: "",
      rag_config: {
        chunk_size: 512,
        chunk_overlap: 64,
        top_k: 5,
        embedding_model: "text-embedding-3-small",
      },
    });
    setDynamicModels([]);
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
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Judge Provider</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="mb-2 text-sm text-muted-foreground">
                  Select a Chat Provider to use as the LLM-as-Judge for auto-scoring responses.
                </p>
                <Select
                  value={config?.judge_chat_provider_id || "__none__"}
                  onValueChange={async (value) => {
                    try {
                      const updated = await updateConfig({
                        judge_chat_provider_id: value === "__none__" ? "" : value,
                      });
                      mutateConfig(updated, false);
                    } catch (err) {
                      console.error("Failed to save judge provider:", err);
                    }
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="None (judging disabled)" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none__">None (judging disabled)</SelectItem>
                    {chatProviders.map((cp) => (
                      <SelectItem key={cp.id} value={cp.id}>
                        {cp.name} ({cp.llm_provider}/{cp.llm_model})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </CardContent>
            </Card>

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

            {showCreateForm && !editingId && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Create Chat Provider</CardTitle>
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
                      setFormData({ ...formData, llm_provider: value, llm_model: "" }); setDynamicModels([]);
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
                    <div className="flex gap-2">
                      <Select
                        value={formData.llm_model}
                        onValueChange={(value) => setFormData({ ...formData, llm_model: value })}
                        disabled={!formData.llm_provider}
                      >
                        <SelectTrigger id="cp-model" className="flex-1">
                          <SelectValue placeholder="Select a model" />
                        </SelectTrigger>
                        <SelectContent>
                          {(dynamicModels.length > 0
                            ? dynamicModels
                            : providers.find((p) => p.name === formData.llm_provider)?.models ?? []
                          ).map((m) => (
                            <SelectItem key={m.name} value={m.name}>
                              {m.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Button
                        type="button"
                        variant="outline"
                        size="icon"
                        disabled={!formData.llm_provider || refreshingModels}
                        onClick={handleRefreshModels}
                        aria-label="Refresh models from provider API"
                      >
                        <RefreshCw className={cn("h-4 w-4", refreshingModels && "animate-spin")} aria-hidden="true" />
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="cp-profile">Profile</Label>
                    <Select
                      value={formData.profile_id}
                      onValueChange={(value) => setFormData({ ...formData, profile_id: value })}
                    >
                      <SelectTrigger id="cp-profile">
                        <SelectValue placeholder="Select a profile" />
                      </SelectTrigger>
                      <SelectContent>
                        {profiles.map((p) => (
                          <SelectItem key={p.id} value={p.id}>
                            {p.name}{p.is_builtin ? " (built-in)" : ""}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {selectedProfile && (
                    <div className="rounded-lg border border-muted bg-muted/30 p-4 space-y-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="inline-flex items-center rounded-full bg-secondary px-2.5 py-0.5 text-xs font-medium text-secondary-foreground capitalize">
                          {selectedProfile.strategy}
                        </span>
                        {selectedProfile.description && (
                          <span className="text-xs text-muted-foreground">{selectedProfile.description}</span>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Temp {selectedProfile.runtime_settings.temperature} · Top P {selectedProfile.runtime_settings.top_p} · Max tokens {selectedProfile.runtime_settings.max_output_tokens} · Timeout {selectedProfile.runtime_settings.timeout_seconds}s
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Steps {selectedProfile.budget.max_steps} · Max cost ${selectedProfile.budget.max_cost_usd} · Time {selectedProfile.budget.max_time_seconds}s
                      </p>
                    </div>
                  )}

                  {selectedProfile?.strategy === "rag" && (
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

                  <div className="flex gap-2">
                    <Button
                      onClick={handleSaveChatProvider}
                      disabled={
                        savingProvider ||
                        !formData.name.trim() ||
                        !formData.llm_provider ||
                        !formData.llm_model ||
                        !formData.profile_id
                      }
                    >
                      Create
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

            {chatProviders.map((provider) => {
              const providerProfile = profiles.find((p) => p.id === provider.profile_id);
              const isEditing = editingId === provider.id && !showCreateForm;
              const editProfile = isEditing ? profiles.find((p) => p.id === formData.profile_id) : null;
              return (
              <Card key={provider.id}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="space-y-1">
                      <CardTitle className="text-base">{provider.name}</CardTitle>
                      <p className="text-sm text-muted-foreground">
                        {provider.llm_provider} · {provider.llm_model}
                        {providerProfile && ` · ${providerProfile.name}`}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="inline-flex items-center rounded-full bg-secondary px-2.5 py-0.5 text-xs font-medium text-secondary-foreground capitalize">
                        {provider.execution_mode}
                      </span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => isEditing ? setEditingId(null) : handleEditChatProvider(provider)}
                        disabled={showCreateForm}
                      >
                        <Edit2 className="h-4 w-4" aria-hidden="true" />
                        <span className="sr-only">{isEditing ? "Cancel edit" : "Edit"}</span>
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
                {!isEditing && (
                  <CardContent className="pt-0 pb-3">
                    <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                      <span>Temp: {provider.runtime_settings.temperature}</span>
                      <span>Max tokens: {provider.runtime_settings.max_output_tokens}</span>
                      {provider.execution_mode === "rlm" && <span>Steps: {provider.rlm_max_steps}</span>}
                      {provider.execution_mode === "rag" && provider.rag_config && (
                        <span>Top K: {provider.rag_config.top_k}</span>
                      )}
                    </div>
                  </CardContent>
                )}
                {isEditing && (
                  <CardContent className="space-y-4 border-t pt-4">
                    <div className="space-y-2">
                      <Label htmlFor={`edit-cp-name-${provider.id}`}>Name</Label>
                      <Input
                        id={`edit-cp-name-${provider.id}`}
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor={`edit-cp-provider-${provider.id}`}>LLM Provider</Label>
                        <Select value={formData.llm_provider} onValueChange={(value) => {
                          setFormData({ ...formData, llm_provider: value, llm_model: "" }); setDynamicModels([]);
                        }}>
                          <SelectTrigger id={`edit-cp-provider-${provider.id}`}>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {providers.map((p) => (
                              <SelectItem key={p.name} value={p.name}>{p.display_name}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor={`edit-cp-model-${provider.id}`}>Model</Label>
                        <div className="flex gap-2">
                          <Select
                            value={formData.llm_model}
                            onValueChange={(value) => setFormData({ ...formData, llm_model: value })}
                            disabled={!formData.llm_provider}
                          >
                            <SelectTrigger id={`edit-cp-model-${provider.id}`} className="flex-1">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {(dynamicModels.length > 0
                                ? dynamicModels
                                : providers.find((p) => p.name === formData.llm_provider)?.models ?? []
                              ).map((m) => (
                                <SelectItem key={m.name} value={m.name}>{m.name}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <Button
                            type="button"
                            variant="outline"
                            size="icon"
                            disabled={!formData.llm_provider || refreshingModels}
                            onClick={handleRefreshModels}
                            aria-label="Refresh models from provider API"
                          >
                            <RefreshCw className={cn("h-4 w-4", refreshingModels && "animate-spin")} aria-hidden="true" />
                          </Button>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor={`edit-cp-profile-${provider.id}`}>Profile</Label>
                      <Select
                        value={formData.profile_id}
                        onValueChange={(value) => setFormData({ ...formData, profile_id: value })}
                      >
                        <SelectTrigger id={`edit-cp-profile-${provider.id}`}>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {profiles.map((p) => (
                            <SelectItem key={p.id} value={p.id}>
                              {p.name}{p.is_builtin ? " (built-in)" : ""}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    {editProfile && (
                      <div className="rounded-lg border border-muted bg-muted/30 p-3 text-xs text-muted-foreground space-y-1">
                        <span className="inline-flex items-center rounded-full bg-secondary px-2 py-0.5 text-xs font-medium capitalize">
                          {editProfile.strategy}
                        </span>
                        <p>Temp {editProfile.runtime_settings.temperature} · Max tokens {editProfile.runtime_settings.max_output_tokens} · Steps {editProfile.budget.max_steps}</p>
                      </div>
                    )}
                    {editProfile?.strategy === "rag" && (
                      <div className="space-y-3 rounded-lg border border-muted p-3">
                        <h4 className="font-medium text-sm">RAG Settings</h4>
                        <div className="grid grid-cols-2 gap-3">
                          <div className="space-y-1">
                            <Label htmlFor={`edit-cp-chunk-${provider.id}`} className="text-xs">Chunk Size</Label>
                            <Input id={`edit-cp-chunk-${provider.id}`} type="number" min="1" className="h-8 text-xs"
                              value={formData.rag_config.chunk_size}
                              onChange={(e) => setFormData({ ...formData, rag_config: { ...formData.rag_config, chunk_size: parseInt(e.target.value) || 512 }})}
                            />
                          </div>
                          <div className="space-y-1">
                            <Label htmlFor={`edit-cp-overlap-${provider.id}`} className="text-xs">Overlap</Label>
                            <Input id={`edit-cp-overlap-${provider.id}`} type="number" min="0" className="h-8 text-xs"
                              value={formData.rag_config.chunk_overlap}
                              onChange={(e) => setFormData({ ...formData, rag_config: { ...formData.rag_config, chunk_overlap: parseInt(e.target.value) || 64 }})}
                            />
                          </div>
                          <div className="space-y-1">
                            <Label htmlFor={`edit-cp-topk-${provider.id}`} className="text-xs">Top K</Label>
                            <Input id={`edit-cp-topk-${provider.id}`} type="number" min="1" className="h-8 text-xs"
                              value={formData.rag_config.top_k}
                              onChange={(e) => setFormData({ ...formData, rag_config: { ...formData.rag_config, top_k: parseInt(e.target.value) || 5 }})}
                            />
                          </div>
                          <div className="space-y-1">
                            <Label htmlFor={`edit-cp-embed-${provider.id}`} className="text-xs">Embedding Model</Label>
                            <Input id={`edit-cp-embed-${provider.id}`} className="h-8 text-xs"
                              value={formData.rag_config.embedding_model}
                              onChange={(e) => setFormData({ ...formData, rag_config: { ...formData.rag_config, embedding_model: e.target.value }})}
                            />
                          </div>
                        </div>
                      </div>
                    )}
                    <div className="flex gap-2">
                      <Button size="sm" onClick={handleSaveChatProvider}
                        disabled={savingProvider || !formData.name.trim() || !formData.llm_provider || !formData.llm_model || !formData.profile_id}
                      >
                        {savingProvider ? "Saving..." : "Save"}
                      </Button>
                      <Button size="sm" variant="outline" onClick={() => setEditingId(null)} disabled={savingProvider}>
                        Cancel
                      </Button>
                    </div>
                  </CardContent>
                )}
              </Card>
              );
            })}

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
              <CardContent className="space-y-3">
                <div className="flex gap-2">
                  <Input
                    placeholder="Profile name"
                    value={newProfileName}
                    onChange={(e) => {
                      setNewProfileName(e.target.value);
                      setProfileError(null);
                    }}
                    onKeyDown={(e) => e.key === "Enter" && handleCreateProfile()}
                    aria-label="New profile name"
                    className="flex-1"
                  />
                  <Select value={newProfileStrategy} onValueChange={setNewProfileStrategy}>
                    <SelectTrigger className="w-[120px]" aria-label="Strategy">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="direct">Direct</SelectItem>
                      <SelectItem value="rlm">RLM</SelectItem>
                      <SelectItem value="rag">RAG</SelectItem>
                    </SelectContent>
                  </Select>
                  <Button
                    onClick={handleCreateProfile}
                    disabled={creatingProfile || !newProfileName.trim()}
                    size="sm"
                  >
                    <Plus className="mr-1 h-4 w-4" aria-hidden="true" />
                    Create
                  </Button>
                </div>
                <div className="flex gap-2 items-center">
                  <Input
                    placeholder="Description (optional)"
                    value={newProfileDescription}
                    onChange={(e) => setNewProfileDescription(e.target.value)}
                    aria-label="Profile description"
                    className="flex-1"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => importFileRef.current?.click()}
                  >
                    <Upload className="mr-1 h-4 w-4" aria-hidden="true" />
                    Import
                  </Button>
                  <input
                    ref={importFileRef}
                    type="file"
                    accept=".json"
                    className="hidden"
                    onChange={handleImportProfile}
                  />
                </div>
                {profileError && (
                  <p className="text-xs text-destructive">{profileError}</p>
                )}
              </CardContent>
            </Card>

            {profiles.map((profile) => (
              <ProfileCard
                key={profile.id}
                profile={profile}
                chatProviders={chatProviders}
                isActive={config?.active_profile_id === profile.id}
                onActivated={() => {
                  mutateConfig();
                  mutateProfiles();
                }}
                onDeleted={() => mutateProfiles()}
                onUpdated={() => {
                  mutateProfiles();
                  mutateChatProviders();
                }}
                onCloned={() => mutateProfiles()}
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
