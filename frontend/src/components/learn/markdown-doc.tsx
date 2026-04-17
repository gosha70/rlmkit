"use client";

/**
 * Markdown loader for the Learn tab.
 *
 * Fetches an allowlisted doc from GET /api/docs/{slug} and renders it
 * through the existing ReactMarkdown + remarkGfm pipeline. The render
 * wrapper matches the chat/compare prose styling so long-form content
 * reads the same everywhere.
 */

import useSWR from "swr";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getDoc, type DocResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

interface MarkdownDocProps {
  slug: string;
  className?: string;
}

export function MarkdownDoc({ slug, className }: MarkdownDocProps) {
  const { data, error, isLoading } = useSWR<DocResponse>(
    ["learn-doc", slug],
    () => getDoc(slug),
    { revalidateOnFocus: false },
  );

  if (isLoading || (!data && !error)) {
    return (
      <div
        role="status"
        aria-label="Loading document"
        className={cn("text-sm text-muted-foreground", className)}
      >
        Loading…
      </div>
    );
  }

  if (error || !data) {
    return (
      <div
        role="alert"
        className={cn(
          "rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 text-sm text-destructive",
          className,
        )}
      >
        Couldn’t load this guide.
      </div>
    );
  }

  return (
    <div
      data-slug={data.slug}
      className={cn(
        "prose prose-sm dark:prose-invert max-w-none overflow-x-auto",
        className,
      )}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{data.content}</ReactMarkdown>
    </div>
  );
}
