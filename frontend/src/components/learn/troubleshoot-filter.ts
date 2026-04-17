/**
 * Pure client-side filter for Troubleshoot entries.
 *
 * Exported as a standalone helper so the matching logic is
 * unit-testable without mounting the page. Matches the spec §5
 * Filtering requirement: case-insensitive substring match across
 * title + symptom + cause, plus optional category narrowing.
 */

import type {
  TroubleshootCategory,
  TroubleshootEntry,
} from "@/lib/api";

export const TROUBLESHOOT_CATEGORIES: ReadonlyArray<TroubleshootCategory> = [
  "Setup",
  "Provider",
  "Compare",
  "Judge",
  "Budget",
  "Runtime",
];

export interface TroubleshootFilterOptions {
  query?: string;
  categories?: ReadonlySet<TroubleshootCategory>;
}

export function filterTroubleshootEntries(
  entries: ReadonlyArray<TroubleshootEntry>,
  options: TroubleshootFilterOptions = {},
): TroubleshootEntry[] {
  const raw = (options.query ?? "").trim().toLowerCase();
  const categories = options.categories;
  const hasQuery = raw.length > 0;
  const hasCategories = categories !== undefined && categories.size > 0;

  if (!hasQuery && !hasCategories) {
    // Spec §5 Filtering: empty query returns all entries.
    return [...entries];
  }

  return entries.filter((entry) => {
    if (hasCategories && !categories!.has(entry.category)) return false;
    if (!hasQuery) return true;
    const haystack = `${entry.title}\n${entry.symptom}\n${entry.cause}`.toLowerCase();
    return haystack.includes(raw);
  });
}
