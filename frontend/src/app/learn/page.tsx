"use client";

import useSWR from "swr";
import { BookOpen, Compass, Plug, LifeBuoy } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { AppShell } from "@/components/shared/app-shell";
import { DiagnosticsStrip } from "@/components/learn/diagnostics-strip";
import { LandingCard } from "@/components/learn/landing-card";
import { getDiagnostics } from "@/lib/api";

interface LearnCard {
  title: string;
  description: string;
  cta: string;
  href: string;
  icon: LucideIcon;
}

// Spec §2 Landing Page — four intent-driven cards. Keep these in
// the spec's order; the test `CookbookPage` / `LearnPage` suites
// index into the array.
const LEARN_CARDS: ReadonlyArray<LearnCard> = [
  {
    title: "What is RLM?",
    description:
      "RLM lets the model inspect and reason iteratively instead of forcing everything into one prompt.",
    cta: "Open Concepts",
    href: "/learn/concepts",
    icon: BookOpen,
  },
  {
    title: "Which mode should I use?",
    description: "A quick decision helper for Direct, RLM, and Compare.",
    cta: "Choose a mode",
    href: "/learn/concepts#mode-guide",
    icon: Compass,
  },
  {
    title: "Set up a model host",
    description:
      "Connect Ollama, LM Studio, vLLM, DGX Spark, or a cloud provider.",
    cta: "Open Cookbook",
    href: "/learn/cookbook",
    icon: Plug,
  },
  {
    title: "Something not working?",
    description: "Check known issues and run diagnostics.",
    cta: "Open Troubleshoot",
    href: "/learn/troubleshoot",
    icon: LifeBuoy,
  },
];

export default function LearnPage() {
  const { data } = useSWR("learn-diagnostics", getDiagnostics, {
    refreshInterval: 30_000,
    dedupingInterval: 30_000,
    errorRetryCount: 2,
  });

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl px-6 py-8">
        <header className="mb-4">
          <h2 className="text-2xl font-semibold tracking-tight">Learn</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Understand RLM Studio, set up providers, and troubleshoot common issues.
          </p>
        </header>

        <DiagnosticsStrip data={data ?? null} className="mb-6" />

        <section
          className="grid grid-cols-1 gap-4 sm:grid-cols-2"
          aria-label="Learn landing cards"
        >
          {LEARN_CARDS.map((c) => (
            <LandingCard
              key={c.href}
              title={c.title}
              description={c.description}
              cta={c.cta}
              href={c.href}
              icon={c.icon}
            />
          ))}
        </section>
      </div>
    </AppShell>
  );
}
