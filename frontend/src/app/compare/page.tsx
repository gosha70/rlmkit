"use client";

/**
 * Compare page — Provider × Mode Matrix Comparison UI (SDD Bet 3 item 3.4).
 *
 * Wraps the synchronous POST /api/chat/compare-matrix endpoint in a
 * self-contained single-purpose view:
 *
 *   - Query input + content/file input
 *   - LLM Provider chip multi-select (filtered to connected/configured)
 *   - Mode multi-select (direct / rlm / rag)
 *   - Profile dropdown (pre-fills runtime_settings + budget)
 *   - Advanced settings (collapsible): runtime, budget, rag config
 *   - Ranking metric dropdown
 *   - "Run Compare" button
 *   - Result grid: one card per (provider × mode) slot with answer
 *     preview, token/cost/latency metrics, and an expand button that
 *     reveals the full answer.
 *   - Summary strip: best slot, total elapsed, comparison_group_id
 *
 * The endpoint is synchronous (waits for every slot to finish before
 * returning), so this page shows a spinner for the duration of the run
 * and renders results all at once.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import useSWR from "swr";
import { Loader2, Trophy, Hash, DollarSign, Clock, Upload, X, ChevronDown, ChevronUp } from "lucide-react";

import { AppShell } from "@/components/shared/app-shell";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import {
  getLLMProviders,
  getProfiles,
  submitCompareMatrix,
  uploadFile,
  type LLMProviderConfig,
  type RunProfile,
  type CompareMatrixResponse,
  type CompareMatrixSlotResponse,
  type FileUploadResponse,
  type MatrixRankingMetric,
  type MatrixSlotMode,
  type RuntimeSettings,
  type BudgetConfig,
  type RAGConfig,
  type CompareMatrixRequestV2,
} from "@/lib/api";
import { ALL_EXECUTION_MODES, MODE_DIRECT, MODE_RAG } from "@/lib/constants";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ALL_MODES: MatrixSlotMode[] = [...ALL_EXECUTION_MODES];
const MODE_DESCRIPTIONS: Record<MatrixSlotMode, string> = {
  direct: "Single LLM call with full content",
  rlm: "Recursive exploration with sandbox",
  rag: "Retrieval over embedded chunks",
};

const RANKING_METRICS: {
  value: MatrixRankingMetric;
  label: string;
  help: string;
}[] = [
  { value: "cost", label: "Cost (lowest wins)", help: "Cheapest successful slot" },
  { value: "tokens", label: "Tokens (fewest wins)", help: "Smallest token footprint" },
  { value: "latency", label: "Latency (fastest wins)", help: "Wall-clock elapsed time" },
  {
    value: "answer_per_cost",
    label: "Answer/Cost (most wins)",
    help: "Answer length divided by cost",
  },
];

// Matches the backend MAX_SLOTS constant in RunMatrixComparisonUseCase.
const MAX_SLOTS = 10;

interface InlineConfig {
  // Runtime
  temperature: number;
  top_p: number;
  max_output_tokens: number;
  timeout_seconds: number;
  // Budget
  max_steps: number;
  max_time_seconds: number;
  max_cost_usd: number;
  repeat_limit: number;
  nudge_at_fraction: number;
  // RAG
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  embedding_model: string;
}

// NOTE: These defaults mirror the backend's sandbox_vars.py / models.py
// defaults.  If the backend changes, these must be updated to match.
const DEFAULT_CONFIG: InlineConfig = {
  temperature: 0.7,
  top_p: 1.0,
  max_output_tokens: 4096,
  timeout_seconds: 120,
  max_steps: 16,
  max_time_seconds: 600,
  max_cost_usd: 5.0,
  repeat_limit: 2,
  nudge_at_fraction: 0.4,
  chunk_size: 1000,
  chunk_overlap: 150,
  top_k: 5,
  embedding_model: "text-embedding-3-small",
};

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function ComparePage() {
  // --- Data ----------------------------------------------------------------
  const { data: llmProviders = [] } = useSWR<LLMProviderConfig[]>(
    "llm-providers",
    () => getLLMProviders(),
  );
  const { data: profiles = [] } = useSWR<RunProfile[]>(
    "profiles",
    () => getProfiles(),
  );

  // --- Inputs --------------------------------------------------------------
  const [query, setQuery] = useState("");
  const [content, setContent] = useState("");
  const [uploadedFile, setUploadedFile] = useState<FileUploadResponse | null>(null);
  const [uploadingFile, setUploadingFile] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [selectedLLMProviderIds, setSelectedLLMProviderIds] = useState<string[]>([]);
  const [selectedModes, setSelectedModes] = useState<Set<MatrixSlotMode>>(
    new Set([MODE_DIRECT]),
  );
  const [rankingMetric, setRankingMetric] = useState<MatrixRankingMetric>("cost");
  const [selectedProfileId, setSelectedProfileId] = useState<string>("");
  const [config, setConfig] = useState<InlineConfig>({ ...DEFAULT_CONFIG });
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // --- Execution state -----------------------------------------------------
  const [result, setResult] = useState<CompareMatrixResponse | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [expandedSlotId, setExpandedSlotId] = useState<string | null>(null);

  // --- Auto-select first available LLM provider once (not on every SWR revalidation) ---
  const hasAutoSelected = useRef(false);
  useEffect(() => {
    if (hasAutoSelected.current) return;
    if (selectedLLMProviderIds.length === 0 && llmProviders.length > 0) {
      const firstUsable = llmProviders.find(
        (lp) => lp.status === "connected" || lp.status === "configured",
      );
      if (firstUsable) {
        setSelectedLLMProviderIds([firstUsable.id]);
        hasAutoSelected.current = true;
      }
    }
  }, [llmProviders, selectedLLMProviderIds.length]);

  // --- Profile pre-fill ----------------------------------------------------
  const handleProfileChange = useCallback(
    (profileId: string) => {
      setSelectedProfileId(profileId);
      if (!profileId) {
        setConfig({ ...DEFAULT_CONFIG });
        return;
      }
      const profile = profiles.find((p) => p.id === profileId);
      if (!profile) return;
      setConfig((prev) => ({
        ...prev,
        temperature: profile.runtime_settings?.temperature ?? prev.temperature,
        top_p: profile.runtime_settings?.top_p ?? prev.top_p,
        max_output_tokens: profile.runtime_settings?.max_output_tokens ?? prev.max_output_tokens,
        timeout_seconds: profile.runtime_settings?.timeout_seconds ?? prev.timeout_seconds,
        max_steps: profile.budget?.max_steps ?? prev.max_steps,
        max_time_seconds: profile.budget?.max_time_seconds ?? prev.max_time_seconds,
        repeat_limit: profile.budget?.repeat_limit ?? prev.repeat_limit,
        nudge_at_fraction: profile.budget?.nudge_at_fraction ?? prev.nudge_at_fraction,
      }));
    },
    [profiles],
  );

  // --- Derived -------------------------------------------------------------
  const usableProviders = useMemo(
    () => llmProviders.filter((lp) => lp.status === "connected" || lp.status === "configured"),
    [llmProviders],
  );

  const totalSlots = selectedLLMProviderIds.length * selectedModes.size;
  const tooManySlots = totalSlots > MAX_SLOTS;
  const modesSorted = useMemo(
    () => ALL_MODES.filter((m) => selectedModes.has(m)),
    [selectedModes],
  );

  const canRun =
    query.trim().length > 0 &&
    selectedLLMProviderIds.length > 0 &&
    selectedModes.size > 0 &&
    !tooManySlots &&
    !isRunning &&
    (content.trim().length > 0 || uploadedFile !== null);

  // --- Handlers ------------------------------------------------------------
  const toggleMode = useCallback((mode: MatrixSlotMode) => {
    setSelectedModes((prev) => {
      const next = new Set(prev);
      if (next.has(mode)) {
        if (next.size > 1) next.delete(mode);
      } else {
        next.add(mode);
      }
      return next;
    });
  }, []);

  const toggleLLMProvider = useCallback((id: string) => {
    setSelectedLLMProviderIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );
  }, []);

  const handleFileUpload = useCallback(async (file: File) => {
    setUploadingFile(true);
    setUploadError(null);
    try {
      const rec = await uploadFile(file);
      setUploadedFile(rec);
      setContent("");
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploadingFile(false);
    }
  }, []);

  const patchConfig = useCallback(
    <K extends keyof InlineConfig>(key: K, value: InlineConfig[K]) => {
      setConfig((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  const handleRun = useCallback(async () => {
    if (!canRun) return;
    setIsRunning(true);
    setRunError(null);
    setResult(null);
    setExpandedSlotId(null);
    try {
      const req: CompareMatrixRequestV2 = {
        query: query.trim(),
        content: uploadedFile ? null : content.trim() || null,
        file_ids: uploadedFile ? [uploadedFile.id] : null,
        llm_provider_ids: selectedLLMProviderIds,
        modes: modesSorted,
        ranking_metric: rankingMetric,
        runtime_settings: {
          temperature: config.temperature,
          top_p: config.top_p,
          max_output_tokens: config.max_output_tokens,
          timeout_seconds: config.timeout_seconds,
        } satisfies RuntimeSettings,
        budget: {
          max_steps: config.max_steps,
          max_tokens: 50000,
          max_cost_usd: config.max_cost_usd,
          max_time_seconds: config.max_time_seconds,
          max_recursion_depth: 5,
          repeat_limit: config.repeat_limit,
          nudge_at_fraction: config.nudge_at_fraction,
        } satisfies BudgetConfig,
        rag_config: selectedModes.has(MODE_RAG)
          ? ({
              chunk_size: config.chunk_size,
              chunk_overlap: config.chunk_overlap,
              top_k: config.top_k,
              embedding_model: config.embedding_model,
            } satisfies RAGConfig)
          : null,
      };
      const resp = await submitCompareMatrix(req);
      setResult(resp);
    } catch (err) {
      setRunError(err instanceof Error ? err.message : "Matrix run failed");
    } finally {
      setIsRunning(false);
    }
  }, [
    canRun,
    query,
    content,
    uploadedFile,
    selectedLLMProviderIds,
    modesSorted,
    rankingMetric,
    selectedModes,
    config,
  ]);

  // --- Render --------------------------------------------------------------
  return (
    <AppShell>
      <div className="mx-auto max-w-6xl space-y-6 p-6">
        <header>
          <h1 className="text-2xl font-bold">Matrix Compare</h1>
          <p className="text-sm text-muted-foreground">
            Run the same query across{" "}
            <span className="font-mono">N providers × M modes</span> in
            parallel and see a ranked comparison.
          </p>
        </header>

        {/* ---------- Inputs panel ---------- */}
        <Card>
          <CardHeader>
            <CardTitle>Configure run</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Query */}
            <div>
              <label className="mb-1 block text-sm font-medium">Query</label>
              <Textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What would you like to ask?"
                rows={2}
                disabled={isRunning}
              />
            </div>

            {/* Content or file */}
            <div>
              <label className="mb-1 block text-sm font-medium">
                Content or file
              </label>
              {uploadedFile ? (
                <div className="flex items-center justify-between rounded-md border bg-muted/30 p-2 text-sm">
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">{uploadedFile.name}</Badge>
                    <span className="text-muted-foreground">
                      {uploadedFile.token_count.toLocaleString()} tok
                    </span>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setUploadedFile(null)}
                    disabled={isRunning}
                    aria-label="Remove uploaded file"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ) : (
                <div className="space-y-2">
                  <Textarea
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    placeholder="Paste text content here, or upload a file below."
                    rows={4}
                    disabled={isRunning}
                  />
                  <label
                    className={`inline-flex cursor-pointer items-center gap-2 rounded-md border border-dashed px-3 py-2 text-sm text-muted-foreground hover:bg-accent ${
                      uploadingFile ? "opacity-50" : ""
                    }`}
                  >
                    {uploadingFile ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Upload className="h-4 w-4" />
                    )}
                    <span>{uploadingFile ? "Uploading…" : "Upload file"}</span>
                    <input
                      type="file"
                      hidden
                      disabled={uploadingFile || isRunning}
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) handleFileUpload(file);
                        e.target.value = "";
                      }}
                    />
                  </label>
                  {uploadError && (
                    <div className="text-sm text-destructive">{uploadError}</div>
                  )}
                </div>
              )}
            </div>

            {/* LLM Provider chip selector */}
            <div>
              <label className="mb-2 block text-sm font-medium">
                LLM providers ({selectedLLMProviderIds.length} selected)
              </label>
              {usableProviders.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No connected or configured LLM providers. Add one in Settings.
                </p>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {usableProviders.map((lp) => {
                    const active = selectedLLMProviderIds.includes(lp.id);
                    return (
                      <button
                        key={lp.id}
                        type="button"
                        onClick={() => toggleLLMProvider(lp.id)}
                        disabled={isRunning}
                        className={`flex flex-col items-start rounded-md border px-3 py-1.5 text-left transition-colors ${
                          active
                            ? "border-primary bg-primary text-primary-foreground"
                            : "border-input bg-background hover:bg-accent"
                        } ${isRunning ? "cursor-not-allowed opacity-50" : ""}`}
                        aria-pressed={active}
                      >
                        <span className="text-sm font-medium">{lp.name}</span>
                        <span
                          className={`text-xs ${
                            active ? "text-primary-foreground/70" : "text-muted-foreground"
                          }`}
                        >
                          {lp.model}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Mode selector */}
            <div>
              <label className="mb-2 block text-sm font-medium">
                Modes ({selectedModes.size} selected)
              </label>
              <div className="flex flex-wrap gap-2">
                {ALL_MODES.map((mode) => {
                  const active = selectedModes.has(mode);
                  return (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => toggleMode(mode)}
                      disabled={isRunning}
                      className={`rounded-md border px-3 py-1.5 text-sm transition-colors ${
                        active
                          ? "border-primary bg-primary text-primary-foreground"
                          : "border-input bg-background hover:bg-accent"
                      } ${isRunning ? "cursor-not-allowed opacity-50" : ""}`}
                      aria-pressed={active}
                      title={MODE_DESCRIPTIONS[mode]}
                    >
                      {mode}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Ranking metric */}
            <div>
              <label className="mb-1 block text-sm font-medium">
                Ranking metric
              </label>
              <Select
                value={rankingMetric}
                onValueChange={(v) => setRankingMetric(v as MatrixRankingMetric)}
                disabled={isRunning}
              >
                <SelectTrigger className="w-80">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {RANKING_METRICS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      <div className="flex flex-col">
                        <span>{opt.label}</span>
                        <span className="text-xs text-muted-foreground">
                          {opt.help}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Profile dropdown */}
            <div>
              <label className="mb-1 block text-sm font-medium">Profile</label>
              <Select
                value={selectedProfileId}
                onValueChange={handleProfileChange}
                disabled={isRunning}
              >
                <SelectTrigger className="w-80">
                  <SelectValue placeholder="No profile (use defaults)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">No profile (use defaults)</SelectItem>
                  {profiles.map((p) => (
                    <SelectItem key={p.id} value={p.id}>
                      <div className="flex flex-col">
                        <span>{p.name}</span>
                        {p.description && (
                          <span className="text-xs text-muted-foreground">
                            {p.description}
                          </span>
                        )}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Advanced settings (collapsible) */}
            <div className="rounded-md border">
              <button
                type="button"
                className="flex w-full items-center justify-between px-4 py-2 text-sm font-medium hover:bg-accent"
                onClick={() => setAdvancedOpen((v) => !v)}
                disabled={isRunning}
              >
                <span>Advanced settings</span>
                {advancedOpen ? (
                  <ChevronUp className="h-4 w-4" />
                ) : (
                  <ChevronDown className="h-4 w-4" />
                )}
              </button>
              {advancedOpen && (
                <div className="space-y-5 border-t px-4 py-4">
                  {/* Runtime settings */}
                  <div>
                    <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      Runtime
                    </p>
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                      <div>
                        <label className="mb-1 block text-xs text-muted-foreground">
                          Temperature (0–2)
                        </label>
                        <Input
                          type="number"
                          min={0}
                          max={2}
                          step={0.1}
                          value={config.temperature}
                          onChange={(e) =>
                            patchConfig("temperature", parseFloat(e.target.value) || 0)
                          }
                          disabled={isRunning}
                          className="h-8 text-sm"
                        />
                      </div>
                      <div>
                        <label className="mb-1 block text-xs text-muted-foreground">
                          Top-p (0–1)
                        </label>
                        <Input
                          type="number"
                          min={0}
                          max={1}
                          step={0.05}
                          value={config.top_p}
                          onChange={(e) =>
                            patchConfig("top_p", parseFloat(e.target.value) || 0)
                          }
                          disabled={isRunning}
                          className="h-8 text-sm"
                        />
                      </div>
                      <div>
                        <label className="mb-1 block text-xs text-muted-foreground">
                          Max output tokens
                        </label>
                        <Input
                          type="number"
                          min={1}
                          value={config.max_output_tokens}
                          onChange={(e) =>
                            patchConfig(
                              "max_output_tokens",
                              parseInt(e.target.value, 10) || 1,
                            )
                          }
                          disabled={isRunning}
                          className="h-8 text-sm"
                        />
                      </div>
                      <div>
                        <label className="mb-1 block text-xs text-muted-foreground">
                          Timeout (s)
                        </label>
                        <Input
                          type="number"
                          min={1}
                          value={config.timeout_seconds}
                          onChange={(e) =>
                            patchConfig(
                              "timeout_seconds",
                              parseInt(e.target.value, 10) || 1,
                            )
                          }
                          disabled={isRunning}
                          className="h-8 text-sm"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Budget settings */}
                  <div>
                    <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      Budget
                    </p>
                    <div className="space-y-3">
                      <div>
                        <div className="mb-1 flex items-center justify-between">
                          <label className="text-xs text-muted-foreground">
                            Max steps (1–50)
                          </label>
                          <span className="text-xs font-mono">{config.max_steps}</span>
                        </div>
                        <Slider
                          min={1}
                          max={50}
                          step={1}
                          value={[config.max_steps]}
                          onValueChange={([v]) => patchConfig("max_steps", v)}
                          disabled={isRunning}
                        />
                      </div>
                      <div>
                        <div className="mb-1 flex items-center justify-between">
                          <label className="text-xs text-muted-foreground">
                            Max time (5–600 s)
                          </label>
                          <span className="text-xs font-mono">
                            {config.max_time_seconds}s
                          </span>
                        </div>
                        <Slider
                          min={5}
                          max={600}
                          step={5}
                          value={[config.max_time_seconds]}
                          onValueChange={([v]) => patchConfig("max_time_seconds", v)}
                          disabled={isRunning}
                        />
                      </div>
                      <div>
                        <label className="mb-1 block text-xs text-muted-foreground">
                          Max cost per slot (USD)
                        </label>
                        <Input
                          type="number"
                          min={0.01}
                          max={100}
                          step={0.5}
                          value={config.max_cost_usd}
                          onChange={(e) =>
                            patchConfig(
                              "max_cost_usd",
                              parseFloat(e.target.value) || 0.01,
                            )
                          }
                          disabled={isRunning}
                          className="h-8 text-sm"
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="mb-1 block text-xs text-muted-foreground">
                            Repeat limit (1–10)
                          </label>
                          <Input
                            type="number"
                            min={1}
                            max={10}
                            value={config.repeat_limit}
                            onChange={(e) =>
                              patchConfig(
                                "repeat_limit",
                                parseInt(e.target.value, 10) || 1,
                              )
                            }
                            disabled={isRunning}
                            className="h-8 text-sm"
                          />
                        </div>
                        <div>
                          <label className="mb-1 block text-xs text-muted-foreground">
                            Nudge at fraction (0.1–1.0)
                          </label>
                          <Input
                            type="number"
                            min={0.1}
                            max={1.0}
                            step={0.1}
                            value={config.nudge_at_fraction}
                            onChange={(e) =>
                              patchConfig(
                                "nudge_at_fraction",
                                parseFloat(e.target.value) || 0.1,
                              )
                            }
                            disabled={isRunning}
                            className="h-8 text-sm"
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* RAG config — only visible when rag mode is selected */}
                  {selectedModes.has(MODE_RAG) && (
                    <div>
                      <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        RAG
                      </p>
                      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                        <div>
                          <label className="mb-1 block text-xs text-muted-foreground">
                            Chunk size
                          </label>
                          <Input
                            type="number"
                            min={100}
                            value={config.chunk_size}
                            onChange={(e) =>
                              patchConfig(
                                "chunk_size",
                                parseInt(e.target.value, 10) || 100,
                              )
                            }
                            disabled={isRunning}
                            className="h-8 text-sm"
                          />
                        </div>
                        <div>
                          <label className="mb-1 block text-xs text-muted-foreground">
                            Chunk overlap
                          </label>
                          <Input
                            type="number"
                            min={0}
                            value={config.chunk_overlap}
                            onChange={(e) =>
                              patchConfig(
                                "chunk_overlap",
                                parseInt(e.target.value, 10) || 0,
                              )
                            }
                            disabled={isRunning}
                            className="h-8 text-sm"
                          />
                        </div>
                        <div>
                          <label className="mb-1 block text-xs text-muted-foreground">
                            Top-k
                          </label>
                          <Input
                            type="number"
                            min={1}
                            value={config.top_k}
                            onChange={(e) =>
                              patchConfig("top_k", parseInt(e.target.value, 10) || 1)
                            }
                            disabled={isRunning}
                            className="h-8 text-sm"
                          />
                        </div>
                        <div>
                          <label className="mb-1 block text-xs text-muted-foreground">
                            Embedding model
                          </label>
                          <Input
                            type="text"
                            value={config.embedding_model}
                            onChange={(e) =>
                              patchConfig("embedding_model", e.target.value)
                            }
                            disabled={isRunning}
                            className="h-8 text-sm"
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Slot count + validation */}
            <div className="flex items-center gap-3 text-sm">
              <Badge
                variant={tooManySlots ? "destructive" : "secondary"}
                className="font-mono"
              >
                {totalSlots} slot{totalSlots === 1 ? "" : "s"} ({selectedLLMProviderIds.length} × {selectedModes.size})
              </Badge>
              {tooManySlots && (
                <span className="text-destructive">
                  Exceeds MAX_SLOTS={MAX_SLOTS}. Reduce providers or modes.
                </span>
              )}
            </div>

            {/* Run button */}
            <div className="flex items-center gap-3">
              <Button onClick={handleRun} disabled={!canRun} size="lg">
                {isRunning ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Running {totalSlots} slots…
                  </>
                ) : (
                  "Run Compare"
                )}
              </Button>
              {runError && (
                <span className="text-sm text-destructive">{runError}</span>
              )}
            </div>
          </CardContent>
        </Card>

        {/* ---------- Results panel ---------- */}
        {result && (
          <ResultsPanel
            result={result}
            expandedSlotId={expandedSlotId}
            setExpandedSlotId={setExpandedSlotId}
          />
        )}
      </div>
    </AppShell>
  );
}

// ---------------------------------------------------------------------------
// Results panel
// ---------------------------------------------------------------------------

interface ResultsPanelProps {
  result: CompareMatrixResponse;
  expandedSlotId: string | null;
  setExpandedSlotId: (id: string | null) => void;
}

function ResultsPanel({ result, expandedSlotId, setExpandedSlotId }: ResultsPanelProps) {
  const rankByIndex = useMemo(() => {
    const map = new Map<number, number>();
    result.ranking.forEach((slotIdx, rankPos) => {
      map.set(slotIdx, rankPos + 1);
    });
    return map;
  }, [result.ranking]);

  const bestSlot: CompareMatrixSlotResponse | null =
    result.ranking.length > 0 && result.slots[result.ranking[0]]?.success
      ? result.slots[result.ranking[0]]
      : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Results</span>
          <span className="text-sm font-normal text-muted-foreground">
            {result.total_elapsed.toFixed(1)}s wall-clock
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Summary strip */}
        <div className="flex flex-wrap gap-4 rounded-md border bg-muted/30 p-3 text-sm">
          {bestSlot ? (
            <div className="flex items-center gap-2">
              <Trophy className="h-4 w-4 text-amber-500" />
              <span className="font-medium">Best:</span>
              <span>{bestSlot.label}</span>
            </div>
          ) : (
            <div className="text-destructive">No successful slots</div>
          )}
          <div className="flex items-center gap-2 text-muted-foreground">
            <span className="font-medium">Ranked by:</span>
            <span>{result.ranking_metric}</span>
          </div>
          <div
            className="flex items-center gap-2 font-mono text-xs text-muted-foreground"
            title="Shared comparison_group_id — every slot's telemetry row references this"
          >
            <span>group:</span>
            <span>{result.comparison_group_id.slice(0, 12)}…</span>
          </div>
        </div>

        {/* Slot grid */}
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          {result.slots.map((slot, idx) => {
            const rank = rankByIndex.get(idx) ?? null;
            const isExpanded = expandedSlotId === slot.slot_id;
            return (
              <SlotCard
                key={slot.slot_id}
                slot={slot}
                rank={rank}
                expanded={isExpanded}
                onToggle={() => setExpandedSlotId(isExpanded ? null : slot.slot_id)}
              />
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Slot card
// ---------------------------------------------------------------------------

interface SlotCardProps {
  slot: CompareMatrixSlotResponse;
  rank: number | null;
  expanded: boolean;
  onToggle: () => void;
}

function SlotCard({ slot, rank, expanded, onToggle }: SlotCardProps) {
  const statusColor = slot.success
    ? rank === 1
      ? "border-amber-500"
      : "border-emerald-500"
    : "border-destructive";

  return (
    <div
      className={`space-y-2 rounded-md border-2 bg-card p-3 ${statusColor}`}
      data-testid={`matrix-slot-${slot.mode}-${slot.provider}`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {rank === 1 && <Trophy className="h-4 w-4 text-amber-500" />}
          <span className="text-sm font-semibold">{slot.label}</span>
        </div>
        {rank !== null && (
          <Badge variant="outline" className="text-xs">
            #{rank}
          </Badge>
        )}
      </div>

      <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
        <span className="flex items-center gap-1" title="Total tokens">
          <Hash className="h-3 w-3" />
          {slot.total_tokens.toLocaleString()}
        </span>
        <span className="flex items-center gap-1" title="Cost (USD)">
          <DollarSign className="h-3 w-3" />
          {slot.total_cost.toFixed(4)}
        </span>
        <span className="flex items-center gap-1" title="Elapsed seconds">
          <Clock className="h-3 w-3" />
          {slot.elapsed_seconds.toFixed(1)}s
        </span>
      </div>

      {slot.success ? (
        <>
          <div className="rounded bg-muted/40 p-2 text-sm">
            {expanded ? (
              <div className="whitespace-pre-wrap">{slot.answer}</div>
            ) : (
              <div className="line-clamp-3 whitespace-pre-wrap">{slot.answer}</div>
            )}
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggle}
            className="w-full text-xs"
          >
            {expanded ? "Collapse" : "Expand full answer"}
          </Button>
        </>
      ) : (
        <div className="rounded bg-destructive/10 p-2 text-sm text-destructive">
          {slot.error || "Slot failed"}
        </div>
      )}
    </div>
  );
}
