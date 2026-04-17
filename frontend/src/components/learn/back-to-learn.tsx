"use client";

/**
 * Small back-navigation link used on every Learn sub-page.
 *
 * Always points at /learn — the Learn landing page — so the user
 * has a one-click path out of any sub-surface regardless of how
 * deep they navigated.
 */

import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { cn } from "@/lib/utils";

interface BackToLearnProps {
  className?: string;
}

export function BackToLearn({ className }: BackToLearnProps) {
  return (
    <Link
      href="/learn"
      aria-label="Back to Learn"
      className={cn(
        "inline-flex items-center gap-1 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded",
        className,
      )}
    >
      <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
      Back to Learn
    </Link>
  );
}
