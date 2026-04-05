"use client";

import { useEffect, useRef, useState } from "react";
import useSWR from "swr";
import { useTheme } from "next-themes";
import { AppShell } from "@/components/shared/app-shell";
import { Badge } from "@/components/ui/badge";
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
  getProfiles,
  createProfile,
  updateConfig,
  getChatProviders,
  createChatProvider,
  updateChatProvider,
  deleteChatProvider,
  getLLMProviders,
  createLLMProvider,
  updateLLMProvider,
  deleteLLMProvider,
  testLLMProvider,
  type AppConfig,
  type RunProfile,
  type ChatProviderConfig,
  type ChatProviderCreateRequest,
  type RAGConfig,
  type LLMProviderConfig,
  type LLMProviderCreateRequest,
} from "@/lib/api";
import { Plus, Edit2, Trash2, Upload, Check, X, Wifi } from "lucide-react";
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
    llm_provider_id: string;
    profile_id: string;
    rag_config: RAGConfig;
  }>({
    name: "",
    llm_provider_id: "",
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

  // LLM Providers state
  const [showLLMProviderForm, setShowLLMProviderForm] = useState(false);
  const [editingLLMProviderId, setEditingLLMProviderId] = useState<string | null>(null);
  const [llmProviderForm, setLLMProviderForm] = useState<{
    name: string;
    backend: string;
    model: string;
    api_key: string;
    endpoint: string;
  }>({
    name: "",
    backend: "openai",
    model: "",
    api_key: "",
    endpoint: "",
  });
  const [savingLLMProvider, setSavingLLMProvider] = useState(false);
  const [deletingLLMProviderId, setDeletingLLMProviderId] = useState<string | null>(null);
  const [testingLLMProviderId, setTestingLLMProviderId] = useState<string | null>(null);
  const [testResults, setTestResults] = useState<Record<string, { connected: boolean; error?: string | null; latency_ms?: number | null }>>({});

  const { data: config, mutate: mutateConfig } = useSWR<AppConfig>("config", getConfig);
  const { data: profiles = [], mutate: mutateProfiles } = useSWR<RunProfile[]>("profiles", getProfiles);
  const { data: chatProviders = [], mutate: mutateChatProviders } = useSWR<ChatProviderConfig[]>(
    "chat-providers",
    getChatProviders
  );
  const { data: llmProviders = [], mutate: mutateLLMProviders } = useSWR<LLMProviderConfig[]>(
    "llm-providers",
    getLLMProviders
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
      llm_provider_id: provider.llm_provider_id,
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
    if (!formData.name.trim() || !formData.llm_provider_id || !formData.profile_id) {
      console.error("Missing required fields");
      return;
    }

    setSavingProvider(true);
    try {
      const selectedProfile = profiles.find((p) => p.id === formData.profile_id);
      const payload: ChatProviderCreateRequest = {
        name: formData.name,
        llm_provider_id: formData.llm_provider_id,
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

  // Resolve the selected profile for the form
  const selectedProfile = profiles.find((p) => p.id === formData.profile_id);

  const resetForm = () => {
    setFormData({
      name: "",
      llm_provider_id: "",
      profile_id: "",
      rag_config: {
        chunk_size: 512,
        chunk_overlap: 64,
        top_k: 5,
        embedding_model: "text-embedding-3-small",
      },
    });
  };

  const resetLLMProviderForm = () => {
    setLLMProviderForm({ name: "", backend: "openai", model: "", api_key: "", endpoint: "" });
    setEditingLLMProviderId(null);
  };

  const handleEditLLMProvider = (lp: LLMProviderConfig) => {
    setEditingLLMProviderId(lp.id);
    setLLMProviderForm({
      name: lp.name,
      backend: lp.backend,
      model: lp.model,
      api_key: "",
      endpoint: lp.endpoint || "",
    });
    setShowLLMProviderForm(true);
  };

  const handleSaveLLMProvider = async () => {
    if (!llmProviderForm.name.trim() || !llmProviderForm.model.trim()) return;
    setSavingLLMProvider(true);
    try {
      const req: LLMProviderCreateRequest = {
        name: llmProviderForm.name.trim(),
        backend: llmProviderForm.backend,
        model: llmProviderForm.model.trim(),
        api_key: llmProviderForm.api_key.trim() || null,
        endpoint: llmProviderForm.endpoint.trim() || null,
      };
      if (editingLLMProviderId) {
        await updateLLMProvider(editingLLMProviderId, req);
      } else {
        await createLLMProvider(req);
      }
      setShowLLMProviderForm(false);
      resetLLMProviderForm();
      mutateLLMProviders();
      toast.success(editingLLMProviderId ? "LLM Provider updated" : "LLM Provider created");
    } catch (err) {
      console.error("Failed to save LLM provider:", err);
      toast.error("Failed to save LLM Provider");
    } finally {
      setSavingLLMProvider(false);
    }
  };

  const handleDeleteLLMProvider = async (id: string) => {
    if (!confirm("Delete this LLM Provider? Any Chat Providers referencing it will stop working.")) return;
    setDeletingLLMProviderId(id);
    try {
      await deleteLLMProvider(id);
      mutateLLMProviders();
    } catch (err) {
      console.error("Failed to delete LLM provider:", err);
      toast.error("Failed to delete LLM Provider");
    } finally {
      setDeletingLLMProviderId(null);
    }
  };

  const handleTestLLMProvider = async (id: string) => {
    setTestingLLMProviderId(id);
    try {
      const result = await testLLMProvider(id);
      setTestResults((prev) => ({
        ...prev,
        [id]: { connected: result.connected, error: result.error, latency_ms: result.latency_ms },
      }));
      mutateLLMProviders();
    } catch (err) {
      console.error("Test failed:", err);
      setTestResults((prev) => ({
        ...prev,
        [id]: { connected: false, error: String(err) },
      }));
    } finally {
      setTestingLLMProviderId(null);
    }
  };

  const LOCAL_BACKENDS = ["ollama", "lmstudio"];
  const CLOUD_BACKENDS = ["openai", "anthropic"];
  const BACKEND_LABELS: Record<string, string> = {
    openai: "OpenAI",
    anthropic: "Anthropic",
    ollama: "Ollama",
    lmstudio: "LM Studio",
  };

  const STATUS_COLORS: Record<string, string> = {
    connected: "bg-emerald-500",
    configured: "bg-amber-500",
    offline: "bg-gray-400",
    not_configured: "bg-gray-400",
  };

  return (
    <AppShell>
      <div className="mx-auto max-w-[1200px] space-y-6 p-6">
        <h2 className="text-2xl font-semibold">Settings</h2>

        <Tabs defaultValue="llm-providers">
          <TabsList>
            <TabsTrigger value="llm-providers">LLM Providers</TabsTrigger>
            <TabsTrigger value="chat-providers">Chat Providers</TabsTrigger>
            <TabsTrigger value="budget">Budget</TabsTrigger>
            <TabsTrigger value="profiles">Profiles</TabsTrigger>
            <TabsTrigger value="prompts">Prompts</TabsTrigger>
            <TabsTrigger value="appearance">Appearance</TabsTrigger>
          </TabsList>

          <TabsContent value="llm-providers" className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-muted-foreground">Named LLM Provider instances</h3>
              {!showLLMProviderForm && (
                <Button
                  size="sm"
                  onClick={() => { resetLLMProviderForm(); setShowLLMProviderForm(true); }}
                >
                  <Plus className="mr-2 h-4 w-4" aria-hidden="true" />
                  Add LLM Provider
                </Button>
              )}
            </div>

            {showLLMProviderForm && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">
                    {editingLLMProviderId ? "Edit LLM Provider" : "Add LLM Provider"}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="llmp-name">Name</Label>
                    <Input
                      id="llmp-name"
                      placeholder="e.g., My GPT-4o"
                      value={llmProviderForm.name}
                      onChange={(e) => setLLMProviderForm({ ...llmProviderForm, name: e.target.value })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="llmp-backend">Backend</Label>
                    <Select
                      value={llmProviderForm.backend}
                      onValueChange={(v) => setLLMProviderForm({ ...llmProviderForm, backend: v, model: "", endpoint: "" })}
                      disabled={!!editingLLMProviderId}
                    >
                      <SelectTrigger id="llmp-backend">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="openai">OpenAI</SelectItem>
                        <SelectItem value="anthropic">Anthropic</SelectItem>
                        <SelectItem value="ollama">Ollama</SelectItem>
                        <SelectItem value="lmstudio">LM Studio</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="llmp-model">Model</Label>
                    <Input
                      id="llmp-model"
                      placeholder={
                        llmProviderForm.backend === "openai" ? "e.g., gpt-4o" :
                        llmProviderForm.backend === "anthropic" ? "e.g., claude-3-5-sonnet-20241022" :
                        "e.g., llama3.2"
                      }
                      value={llmProviderForm.model}
                      onChange={(e) => setLLMProviderForm({ ...llmProviderForm, model: e.target.value })}
                    />
                  </div>
                  {LOCAL_BACKENDS.includes(llmProviderForm.backend) && (
                    <div className="space-y-2">
                      <Label htmlFor="llmp-endpoint">Endpoint</Label>
                      <Input
                        id="llmp-endpoint"
                        placeholder={llmProviderForm.backend === "ollama" ? "http://localhost:11434" : "http://localhost:1234"}
                        value={llmProviderForm.endpoint}
                        onChange={(e) => setLLMProviderForm({ ...llmProviderForm, endpoint: e.target.value })}
                      />
                    </div>
                  )}
                  {CLOUD_BACKENDS.includes(llmProviderForm.backend) && (
                    <div className="space-y-2">
                      <Label htmlFor="llmp-apikey">API Key</Label>
                      <Input
                        id="llmp-apikey"
                        type="password"
                        placeholder="sk-..."
                        value={llmProviderForm.api_key}
                        onChange={(e) => setLLMProviderForm({ ...llmProviderForm, api_key: e.target.value })}
                      />
                    </div>
                  )}
                  <div className="flex gap-2">
                    <Button
                      onClick={handleSaveLLMProvider}
                      disabled={savingLLMProvider || !llmProviderForm.name.trim() || !llmProviderForm.model.trim()}
                    >
                      {editingLLMProviderId ? "Save" : "Create"}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => { setShowLLMProviderForm(false); resetLLMProviderForm(); }}
                      disabled={savingLLMProvider}
                    >
                      Cancel
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {llmProviders.map((lp) => {
              const testResult = testResults[lp.id];
              const statusColor = STATUS_COLORS[lp.status] || "bg-gray-400";
              return (
                <Card key={lp.id}>
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <CardTitle className="text-base">{lp.name}</CardTitle>
                          <span
                            className={cn("inline-block h-2 w-2 rounded-full", statusColor)}
                            title={lp.status}
                          />
                          <Badge variant="secondary" className="text-xs capitalize">
                            {lp.status.replace("_", " ")}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {BACKEND_LABELS[lp.backend] || lp.backend} &middot; {lp.model}
                          {lp.endpoint && <> &middot; {lp.endpoint}</>}
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleTestLLMProvider(lp.id)}
                          disabled={testingLLMProviderId === lp.id}
                        >
                          <Wifi className="mr-1 h-3 w-3" aria-hidden="true" />
                          {testingLLMProviderId === lp.id ? "Testing..." : "Test"}
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleEditLLMProvider(lp)}
                          disabled={showLLMProviderForm && editingLLMProviderId === lp.id}
                        >
                          <Edit2 className="h-4 w-4" aria-hidden="true" />
                          <span className="sr-only">Edit</span>
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDeleteLLMProvider(lp.id)}
                          disabled={deletingLLMProviderId === lp.id}
                        >
                          <Trash2 className="h-4 w-4" aria-hidden="true" />
                          <span className="sr-only">Delete</span>
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  {testResult && (
                    <CardContent className="pt-0 pb-3">
                      <div className={cn("flex items-center gap-2 text-xs", testResult.connected ? "text-emerald-600" : "text-destructive")}>
                        {testResult.connected
                          ? <><Check className="h-3 w-3" /> Connected{testResult.latency_ms != null && ` (${testResult.latency_ms}ms)`}</>
                          : <><X className="h-3 w-3" /> {testResult.error || "Connection failed"}</>
                        }
                      </div>
                    </CardContent>
                  )}
                </Card>
              );
            })}

            {llmProviders.length === 0 && !showLLMProviderForm && (
              <Card>
                <CardContent className="py-10 text-center text-muted-foreground">
                  No LLM Providers configured yet. Add one to get started.
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
                        {cp.name} ({cp.llm_provider_name || cp.llm_provider_id})
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
                    <Label htmlFor="cp-llm-provider">LLM Provider</Label>
                    <Select
                      value={formData.llm_provider_id}
                      onValueChange={(value) => setFormData({ ...formData, llm_provider_id: value })}
                    >
                      <SelectTrigger id="cp-llm-provider">
                        <SelectValue placeholder="Select an LLM Provider" />
                      </SelectTrigger>
                      <SelectContent>
                        {llmProviders.map((lp) => (
                          <SelectItem key={lp.id} value={lp.id}>
                            <span>{lp.name}</span>
                            <span className="ml-2 text-xs text-muted-foreground">
                              {BACKEND_LABELS[lp.backend] || lp.backend} &middot; {lp.model}
                            </span>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
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
                        !formData.llm_provider_id ||
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
                        {provider.llm_provider_name || provider.llm_provider_id}
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
                      <span>Timeout: {provider.runtime_settings.timeout_seconds}s</span>
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
                    <div className="space-y-2">
                      <Label htmlFor={`edit-cp-llm-provider-${provider.id}`}>LLM Provider</Label>
                      <Select
                        value={formData.llm_provider_id}
                        onValueChange={(value) => setFormData({ ...formData, llm_provider_id: value })}
                      >
                        <SelectTrigger id={`edit-cp-llm-provider-${provider.id}`}>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {llmProviders.map((lp) => (
                            <SelectItem key={lp.id} value={lp.id}>
                              {lp.name} &middot; {BACKEND_LABELS[lp.backend] || lp.backend} &middot; {lp.model}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
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
                        disabled={savingProvider || !formData.name.trim() || !formData.llm_provider_id || !formData.profile_id}
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
