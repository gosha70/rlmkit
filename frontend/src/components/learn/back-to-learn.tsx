"use client";

/**
 * Small back-navigation link used on every Learn sub-page.
 *
 * Defaults to /learn (the Learn landing page).  When the `from` query
 * param is present (e.g. `?from=traces`), the link points back to the
 * referring page instead — so "Replay in Learn" from Traces shows
 * "Back to Traces" rather than "Back to Learn".
 */

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import { cn } from "@/lib/utils";

const FROM_TARGETS: Record<string, { href: string; label: string }> = {
  traces: { href: "/traces", label: "Back to Traces" },
};

interface BackToLearnProps {
  className?: string;
}

export function BackToLearn({ className }: BackToLearnProps) {
  const searchParams = useSearchParams();
  const from = searchParams.get("from");
  const target = (from && FROM_TARGETS[from]) || { href: "/learn", label: "Back to Learn" };

  return (
    <Link
      href={target.href}
      aria-label={target.label}
      className={cn(
        "inline-flex items-center gap-1 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded",
        className,
      )}
    >
      <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
      {target.label}
    </Link>
  );
}
