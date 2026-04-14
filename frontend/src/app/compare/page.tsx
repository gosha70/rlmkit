"use client";

/**
 * Compare page — Provider × Mode Matrix Comparison UI (SDD Bet 3 item 3.4).
 *
 * Wraps the synchronous POST /api/chat/compare-matrix endpoint in a
 * self-contained single-purpose view:
 *
 *   - Query input + content/file input
 *   - Chat Provider multi-select (reuses the existing selector)
 *   - Mode multi-select (direct / rlm / rag)
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

import { useCallback, useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { Loader2, Trophy, Hash, DollarSign, Clock, Upload, X } from "lucide-react";

import { AppShell } from "@/components/shared/app-shell";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { ChatProviderSelector } from "@/components/chat/chat-provider-selector";
import {
  getChatProviders,
  getLLMProviders,
  submitCompareMatrix,
  uploadFile,
  type ChatProviderConfig,
  type LLMProviderConfig,
  type CompareMatrixResponse,
  type CompareMatrixSlotResponse,
  type FileUploadResponse,
  type MatrixRankingMetric,
  type MatrixSlotMode,
} from "@/lib/api";
import { ALL_EXECUTION_MODES, MODE_DIRECT } from "@/lib/constants";

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

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function ComparePage() {
  // --- Data ----------------------------------------------------------------
  const { data: chatProviders = [] } = useSWR<ChatProviderConfig[]>(
    "chat-providers",
    () => getChatProviders(),
  );
  const { data: llmProviders = [] } = useSWR<LLMProviderConfig[]>(
    "llm-providers",
    () => getLLMProviders(),
  );

  // --- Inputs --------------------------------------------------------------
  const [query, setQuery] = useState("");
  const [content, setContent] = useState("");
  const [uploadedFile, setUploadedFile] = useState<FileUploadResponse | null>(null);
  const [uploadingFile, setUploadingFile] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [selectedChatProviderIds, setSelectedChatProviderIds] = useState<string[]>([]);
  const [selectedModes, setSelectedModes] = useState<Set<MatrixSlotMode>>(
    new Set([MODE_DIRECT]),
  );
  const [rankingMetric, setRankingMetric] = useState<MatrixRankingMetric>("cost");

  // --- Execution state -----------------------------------------------------
  const [result, setResult] = useState<CompareMatrixResponse | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [expandedSlotId, setExpandedSlotId] = useState<string | null>(null);

  // --- Auto-select first available chat provider once data loads ----------
  useEffect(() => {
    if (selectedChatProviderIds.length === 0 && chatProviders.length > 0) {
      // Prefer the first provider whose underlying LLM is connected/configured.
      const lpMap = new Map(llmProviders.map((p) => [p.id, p]));
      const firstUsable = chatProviders.find((cp) => {
        const lp = lpMap.get(cp.llm_provider_id);
        return lp && (lp.status === "connected" || lp.status === "configured");
      });
      if (firstUsable) {
        setSelectedChatProviderIds([firstUsable.id]);
      }
    }
  }, [chatProviders, llmProviders, selectedChatProviderIds.length]);

  // --- Derived -------------------------------------------------------------
  const totalSlots = selectedChatProviderIds.length * selectedModes.size;
  const tooManySlots = totalSlots > MAX_SLOTS;
  const modesSorted = useMemo(
    () => ALL_MODES.filter((m) => selectedModes.has(m)),
    [selectedModes],
  );

  const canRun =
    query.trim().length > 0 &&
    selectedChatProviderIds.length > 0 &&
    selectedModes.size > 0 &&
    !tooManySlots &&
    !isRunning &&
    (content.trim().length > 0 || uploadedFile !== null);

  // --- Handlers ------------------------------------------------------------
  const toggleMode = useCallback((mode: MatrixSlotMode) => {
    setSelectedModes((prev) => {
      const next = new Set(prev);
      if (next.has(mode)) {
        if (next.size > 1) next.delete(mode); // keep at least one
      } else {
        next.add(mode);
      }
      return next;
    });
  }, []);

  const handleFileUpload = useCallback(
    async (file: File) => {
      setUploadingFile(true);
      setUploadError(null);
      try {
        const rec = await uploadFile(file);
        setUploadedFile(rec);
        // Clear the text content field — file takes precedence.
        setContent("");
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : "Upload failed");
      } finally {
        setUploadingFile(false);
      }
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
      const resp = await submitCompareMatrix({
        query: query.trim(),
        content: uploadedFile ? null : content.trim() || null,
        file_ids: uploadedFile ? [uploadedFile.id] : null,
        chat_provider_ids: selectedChatProviderIds,
        modes: modesSorted,
        ranking_metric: rankingMetric,
      });
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
    selectedChatProviderIds,
    modesSorted,
    rankingMetric,
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
                        e.target.value = ""; // allow re-upload of the same name
                      }}
                    />
                  </label>
                  {uploadError && (
                    <div className="text-sm text-destructive">{uploadError}</div>
                  )}
                </div>
              )}
            </div>

            {/* Chat provider selector */}
            <div>
              <label className="mb-2 block text-sm font-medium">
                Chat providers ({selectedChatProviderIds.length} selected)
              </label>
              <ChatProviderSelector
                chatProviders={chatProviders}
                llmProviders={llmProviders}
                selectedIds={selectedChatProviderIds}
                onSelectionChange={setSelectedChatProviderIds}
                disabled={isRunning}
              />
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

            {/* Slot count + validation */}
            <div className="flex items-center gap-3 text-sm">
              <Badge
                variant={tooManySlots ? "destructive" : "secondary"}
                className="font-mono"
              >
                {totalSlots} slot{totalSlots === 1 ? "" : "s"} ({selectedChatProviderIds.length} × {selectedModes.size})
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
        {result && <ResultsPanel
          result={result}
          expandedSlotId={expandedSlotId}
          setExpandedSlotId={setExpandedSlotId}
        />}
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
  // Map ranking position to slot index for crown rendering.
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
                onToggle={() =>
                  setExpandedSlotId(isExpanded ? null : slot.slot_id)
                }
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
              <div className="line-clamp-3 whitespace-pre-wrap">
                {slot.answer}
              </div>
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
