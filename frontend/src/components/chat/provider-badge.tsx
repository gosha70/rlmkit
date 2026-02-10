"use client";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

interface ProviderBadgeProps {
  provider: string;
  model?: string | null;
  status?: "connected" | "offline" | "not_configured";
  className?: string;
}

const statusDotColor: Record<string, string> = {
  connected: "bg-emerald-500",
  offline: "bg-red-500",
  not_configured: "bg-gray-400",
};

export function ProviderBadge({ provider, model, status = "connected", className }: ProviderBadgeProps) {
  return (
    <Badge variant="outline" className={cn("gap-1.5 font-normal", className)}>
      <span
        className={cn("inline-block h-2 w-2 rounded-full", statusDotColor[status])}
        aria-hidden="true"
      />
      <span>{model ? `${model} via ${provider}` : provider}</span>
    </Badge>
  );
}
