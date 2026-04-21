"use client";

/**
 * Trace-backed replay page — Learn tab V2b (NEXT.md §3f step 6).
 *
 * Loads a {@link LearnReplay} for ``executionId`` via SWR and renders
 * it through the shared {@link ReplayWalkthrough} widget — the same
 * one the Concepts page uses for the bundled replay. Loading / error /
 * ready states mirror Concepts so the two surfaces feel identical.
 *
 * Unknown execution ids surface the standard 404 envelope from the
 * backend (``/api/replays/{id}`` raises a ``NOT_FOUND`` HTTPException
 * when the id is absent from both the live ``AppState.executions``
 * dict and the persisted telemetry store). The alert + back link give
 * the user a one-click path out regardless of how they landed here.
 */

import Link from "next/link";
import { useParams } from "next/navigation";
import useSWR from "swr";

import { BackToLearn } from "@/components/learn/back-to-learn";
import { ReplayWalkthrough } from "@/components/learn/replay-walkthrough";
import { AppShell } from "@/components/shared/app-shell";
import { getReplay, type LearnReplay } from "@/lib/api";

export default function ReplayPage() {
  const params = useParams<{ executionId: string }>();
  const executionId = params?.executionId ?? "";

  const { data: replay, error } = useSWR<LearnReplay>(
    executionId ? ["learn-replay", executionId] : null,
    () => getReplay(executionId),
    { revalidateOnFocus: false },
  );

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl px-6 py-8">
        <BackToLearn className="mb-4" />

        <header className="mb-6">
          <h2 className="text-2xl font-semibold tracking-tight">
            Execution replay
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Reconstructed from the saved trace for this execution.
          </p>
        </header>

        {error ? (
          <div
            role="alert"
            className="rounded-md border border-destructive/40 bg-destructive/5 px-4 py-3 text-sm text-destructive"
          >
            <p className="font-medium">Couldn’t load this replay.</p>
            <p className="mt-1 text-destructive/80">
              The execution may have been deleted, or the id is
              unknown.{" "}
              <Link
                href="/traces"
                className="underline underline-offset-2 hover:text-destructive"
              >
                Back to Traces
              </Link>
              .
            </p>
          </div>
        ) : !replay ? (
          <p className="text-sm text-muted-foreground">Loading replay…</p>
        ) : (
          <>
            {replay.metadata.truncated ? (
              <div
                role="note"
                className="mb-4 rounded-md border border-amber-300/60 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-400/30 dark:bg-amber-500/10 dark:text-amber-200"
              >
                This replay was truncated to stay within the display
                cap. The full replay would have had{" "}
                {replay.metadata.originalStepCount} steps.
              </div>
            ) : null}
            <ReplayWalkthrough replay={replay} />
          </>
        )}
      </div>
    </AppShell>
  );
}
