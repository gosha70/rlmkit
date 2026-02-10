"use client";

import { cn } from "@/lib/utils";
import type { ChatMode } from "@/lib/api";

interface TypingIndicatorProps {
  mode?: ChatMode;
  step?: string;
  className?: string;
}

export function TypingIndicator({ mode, step, className }: TypingIndicatorProps) {
  return (
    <div className={cn("flex items-center gap-2 text-sm text-muted-foreground", className)} role="status" aria-label="Generating response">
      <div className="flex gap-1">
        <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:0ms]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:150ms]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:300ms]" />
      </div>
      {step ? (
        <span>{step}</span>
      ) : mode ? (
        <span>Generating ({mode})...</span>
      ) : (
        <span>Generating...</span>
      )}
    </div>
  );
}
