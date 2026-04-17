"use client";

/**
 * Search + quick-filter chips for the Troubleshoot page.
 *
 * Controlled component: parent owns `query` and `categories`, this
 * component just renders controls and emits change events. Chips
 * toggle category membership; clicking an active chip removes it.
 */

import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import type { TroubleshootCategory } from "@/lib/api";
import { TROUBLESHOOT_CATEGORIES } from "./troubleshoot-filter";

interface TroubleshootSearchProps {
  query: string;
  onQueryChange: (value: string) => void;
  categories: ReadonlySet<TroubleshootCategory>;
  onToggleCategory: (category: TroubleshootCategory) => void;
  className?: string;
}

export function TroubleshootSearch({
  query,
  onQueryChange,
  categories,
  onToggleCategory,
  className,
}: TroubleshootSearchProps) {
  return (
    <div className={cn("flex flex-col gap-3", className)}>
      <label className="relative block" aria-label="Search troubleshoot entries">
        <Search
          aria-hidden="true"
          className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
        />
        <Input
          type="search"
          value={query}
          placeholder="Search symptoms, errors, or keywords"
          onChange={(e) => onQueryChange(e.target.value)}
          className="pl-9"
          aria-label="Search"
        />
      </label>

      <div
        role="group"
        aria-label="Filter by category"
        className="flex flex-wrap gap-2"
      >
        {TROUBLESHOOT_CATEGORIES.map((cat) => {
          const active = categories.has(cat);
          return (
            <button
              key={cat}
              type="button"
              aria-pressed={active}
              onClick={() => onToggleCategory(cat)}
              className={cn(
                "rounded-full border px-3 py-1 text-xs font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                active
                  ? "border-primary bg-primary text-primary-foreground"
                  : "border-input bg-background text-muted-foreground hover:bg-muted",
              )}
            >
              {cat}
            </button>
          );
        })}
      </div>
    </div>
  );
}
