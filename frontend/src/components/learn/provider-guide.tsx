"use client";

/**
 * Two-pane provider guide: left rail of H2 anchors, right pane of
 * rendered markdown. Both panes draw from the same SWR cache entry
 * for the doc, so the TOC and the rendered body stay in sync without
 * a second fetch.
 */

import Link from "next/link";
import useSWR from "swr";
import { ExternalLink } from "lucide-react";
import { getDoc, type DocResponse } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { MarkdownDoc } from "./markdown-doc";
import { topLevelHeadings } from "./markdown-toc";
import {
  docSlugForProvider,
  type CookbookProvider,
  type ProviderDifficulty,
} from "./provider-catalog";

const DIFFICULTY_VARIANT: Record<
  ProviderDifficulty,
  "success" | "warning" | "outline"
> = {
  Easy: "success",
  Moderate: "warning",
  Advanced: "outline",
};

interface ProviderGuideProps {
  provider: CookbookProvider;
}

export function ProviderGuide({ provider }: ProviderGuideProps) {
  const slug = docSlugForProvider(provider.id);
  const { data } = useSWR<DocResponse>(
    ["learn-doc", slug],
    () => getDoc(slug),
    { revalidateOnFocus: false },
  );

  const headings = data ? topLevelHeadings(data.content) : [];
  const settingsHref = `/settings?provider=${encodeURIComponent(provider.id)}`;

  return (
    <div className="flex flex-col gap-4">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-2xl font-semibold tracking-tight">
              {provider.name}
            </h2>
            <Badge variant={DIFFICULTY_VARIANT[provider.difficulty]}>
              {provider.difficulty}
            </Badge>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            {provider.bestFor}
          </p>
        </div>
        <Link
          href={settingsHref}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-md border bg-card px-3 py-2 text-sm font-medium shadow-sm transition hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label={`Open ${provider.name} in Settings`}
        >
          Open in Settings
          <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
        </Link>
      </header>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-[12rem_1fr]">
        <aside
          aria-label={`${provider.name} guide sections`}
          className="hidden md:block"
        >
          {headings.length > 0 ? (
            <nav>
              <ol className="sticky top-4 flex flex-col gap-1 text-sm">
                {headings.map((h) => (
                  <li key={h.id}>
                    <a
                      href={`#${h.id}`}
                      className="block rounded px-2 py-1 text-muted-foreground transition hover:bg-muted hover:text-foreground"
                    >
                      {h.text}
                    </a>
                  </li>
                ))}
              </ol>
            </nav>
          ) : null}
        </aside>

        <MarkdownDoc slug={slug} />
      </div>
    </div>
  );
}
