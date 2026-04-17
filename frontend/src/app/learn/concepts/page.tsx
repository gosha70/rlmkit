"use client";

import useSWR from "swr";
import { AppShell } from "@/components/shared/app-shell";
import { BackToLearn } from "@/components/learn/back-to-learn";
import { DiagnosticsStrip } from "@/components/learn/diagnostics-strip";
import { MarkdownDoc } from "@/components/learn/markdown-doc";
import { getDiagnostics } from "@/lib/api";

/**
 * Concepts page — V1 scope only.
 *
 * Renders two static sections:
 * - "What is RLM?" sourced from docs/rlm-concepts.md via the
 *   allowlisted markdown loader.
 * - "Which mode should I use?" — static decision strip + "when
 *   not to use RLM" callout. Anchor id="mode-guide" so the Learn
 *   landing page can deep-link card 2 to this section.
 *
 * The interactive replay walkthrough (spec §3 Section C) is V2
 * and intentionally absent here.
 */
export default function ConceptsPage() {
  const { data } = useSWR("learn-diagnostics", getDiagnostics, {
    refreshInterval: 30_000,
    dedupingInterval: 30_000,
    errorRetryCount: 2,
  });

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl px-6 py-8">
        <BackToLearn className="mb-4" />
        <header className="mb-4">
          <h2 className="text-2xl font-semibold tracking-tight">Concepts</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Understand Direct, RLM, and Compare — and when each makes sense.
          </p>
        </header>

        <DiagnosticsStrip data={data ?? null} className="mb-6" />

        <section
          id="what-is-rlm"
          aria-labelledby="concepts-what-is-rlm"
          className="mb-10"
        >
          <h3
            id="concepts-what-is-rlm"
            className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground"
          >
            What is RLM?
          </h3>
          <MarkdownDoc slug="rlm-concepts" />
        </section>

        <section
          id="mode-guide"
          aria-labelledby="concepts-mode-guide"
          className="mb-10"
        >
          <h3
            id="concepts-mode-guide"
            className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground"
          >
            Which mode should I use?
          </h3>
          <div className="rounded-lg border bg-card p-4 shadow-sm">
            <p className="text-sm font-medium">How big or complex is the task?</p>
            <ul className="mt-3 flex flex-col gap-2 text-sm">
              <li className="flex items-start gap-3">
                <span className="mt-1 inline-block h-2 w-2 shrink-0 rounded-full bg-emerald-500" />
                <span>
                  <span className="font-semibold">Direct</span> — small,
                  self-contained task. One-shot answer.
                </span>
              </li>
              <li className="flex items-start gap-3">
                <span className="mt-1 inline-block h-2 w-2 shrink-0 rounded-full bg-blue-500" />
                <span>
                  <span className="font-semibold">RLM</span> — large input,
                  needs iteration. Model inspects and reasons step-by-step.
                </span>
              </li>
              <li className="flex items-start gap-3">
                <span className="mt-1 inline-block h-2 w-2 shrink-0 rounded-full bg-purple-500" />
                <span>
                  <span className="font-semibold">Compare</span> — side-by-side
                  evaluation across providers, profiles, or modes.
                </span>
              </li>
            </ul>
          </div>

          <aside className="mt-4 rounded-md border border-amber-300/60 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-400/30 dark:bg-amber-500/10 dark:text-amber-200">
            <p className="font-semibold">When not to use RLM</p>
            <p className="mt-1">
              Tiny prompts, simple chat, and tasks where the answer is obvious
              from a short input. Direct is cheaper and faster in those cases.
            </p>
          </aside>
        </section>
      </div>
    </AppShell>
  );
}
