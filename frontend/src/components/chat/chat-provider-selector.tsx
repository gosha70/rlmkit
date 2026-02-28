"use client";

import { type ChatProviderConfig, type ProviderInfo } from "@/lib/api";
import { cn } from "@/lib/utils";

interface ChatProviderSelectorProps {
  chatProviders: ChatProviderConfig[];
  providers: ProviderInfo[];
  selectedIds: string[];
  onSelectionChange: (ids: string[]) => void;
  disabled?: boolean;
}

const MODE_BADGES: Record<string, string> = {
  direct: "D",
  rlm: "R",
  rag: "G",
};

export function ChatProviderSelector({
  chatProviders,
  providers,
  selectedIds,
  onSelectionChange,
  disabled = false,
}: ChatProviderSelectorProps) {
  // Create a map of provider names to their info for quick lookup
  const providerMap = new Map(providers.map((p) => [p.name, p]));

  // Filter chat providers to only show those with connected or configured underlying LLM providers
  const availableChatProviders = chatProviders.filter((cp) => {
    const providerInfo = providerMap.get(cp.llm_provider);
    return (
      providerInfo &&
      (providerInfo.status === "connected" || providerInfo.status === "configured")
    );
  });

  // Get status and badge for a provider
  const getProviderStatus = (llmProviderName: string) => {
    const providerInfo = providerMap.get(llmProviderName);
    if (!providerInfo) return { status: "unavailable", color: "bg-gray-400" };
    if (providerInfo.status === "connected") return { status: "connected", color: "bg-emerald-500" };
    if (providerInfo.status === "configured") return { status: "configured", color: "bg-amber-500" };
    return { status: "unavailable", color: "bg-gray-400" };
  };

  const toggle = (id: string) => {
    if (disabled) return;

    if (selectedIds.includes(id)) {
      // Don't allow deselecting the last provider
      if (selectedIds.length === 1) return;
      onSelectionChange(selectedIds.filter((sid) => sid !== id));
    } else {
      onSelectionChange([...selectedIds, id]);
    }
  };

  if (availableChatProviders.length === 0) {
    return (
      <div
        className={cn(
          "flex items-center rounded-lg border bg-muted p-3 text-xs text-muted-foreground",
          disabled && "opacity-50",
        )}
      >
        No Chat Providers available. Configure one in Settings.
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex flex-wrap gap-2 rounded-lg border bg-muted p-2",
        disabled && "opacity-50",
      )}
      role="group"
      aria-label="Chat providers"
    >
      {availableChatProviders.map((cp) => {
        const isSelected = selectedIds.includes(cp.id);
        const { status, color } = getProviderStatus(cp.llm_provider);
        const modeBadge = MODE_BADGES[cp.execution_mode] || "?";

        return (
          <button
            key={cp.id}
            role="checkbox"
            aria-checked={isSelected}
            aria-label={`${cp.name} (${status})`}
            disabled={disabled || (isSelected && selectedIds.length === 1)}
            onClick={() => toggle(cp.id)}
            className={cn(
              "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
              "disabled:cursor-not-allowed",
              isSelected
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground hover:bg-background/50",
            )}
          >
            {/* Provider name */}
            <span>{cp.name}</span>

            {/* Mode badge */}
            <span className="rounded bg-muted px-1 py-0.5 text-xs font-semibold text-muted-foreground">
              {modeBadge}
            </span>

            {/* Status dot */}
            <span
              className={cn("inline-block h-2 w-2 rounded-full", color)}
              title={status}
              aria-label={`Provider status: ${status}`}
            />
          </button>
        );
      })}
    </div>
  );
}
