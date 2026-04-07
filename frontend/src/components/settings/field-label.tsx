"use client";

import { HelpCircle } from "lucide-react";
import { Label } from "@/components/ui/label";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface FieldLabelProps {
  htmlFor?: string;
  tooltip: string;
  children: React.ReactNode;
  className?: string;
}

/**
 * A label with an inline (?) tooltip icon that shows a help message on hover.
 */
export function FieldLabel({ htmlFor, tooltip, children, className }: FieldLabelProps) {
  return (
    <div className={`flex items-center gap-1 ${className ?? ""}`}>
      <Label htmlFor={htmlFor}>{children}</Label>
      <TooltipProvider delayDuration={200}>
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              className="inline-flex items-center justify-center rounded-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring cursor-help shrink-0"
              aria-label="Help"
            >
              <HelpCircle
                className="h-3.5 w-3.5 text-muted-foreground/60 hover:text-muted-foreground"
                aria-hidden="true"
              />
            </button>
          </TooltipTrigger>
          <TooltipContent side="top" className="max-w-[280px] text-xs">
            {tooltip}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  );
}
