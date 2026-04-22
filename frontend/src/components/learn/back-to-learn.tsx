"use client";

/**
 * Small back-navigation link used on every Learn sub-page.
 *
 * Defaults to /learn (the Learn landing page).  The replay page can
 * override the target via props when the user arrived from Traces,
 * so they see "Back to Traces" instead of "Back to Learn".
 *
 * Does NOT use useSearchParams — that would require every page that
 * renders this component to be wrapped in a Suspense boundary (Next.js
 * static-generation constraint).
 */

import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { cn } from "@/lib/utils";

interface BackToLearnProps {
  className?: string;
  /** Override the back-link target (default: "/learn"). */
  href?: string;
  /** Override the label (default: "Back to Learn"). */
  label?: string;
}

export function BackToLearn({
  className,
  href = "/learn",
  label = "Back to Learn",
}: BackToLearnProps) {
  return (
    <Link
      href={href}
      aria-label={label}
      className={cn(
        "inline-flex items-center gap-1 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded",
        className,
      )}
    >
      <ArrowLeft className="h-3.5 w-3.5" aria-hidden="true" />
      {label}
    </Link>
  );
}
