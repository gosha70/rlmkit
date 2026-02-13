"use client";

import { TrendingDown } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { MessageMetrics } from "@/lib/api";

interface ComparisonBannerProps {
  rlmMetrics: MessageMetrics;
  directMetrics: MessageMetrics;
}

export function ComparisonBanner({ rlmMetrics, directMetrics }: ComparisonBannerProps) {
  const tokenSavings = directMetrics.total_tokens > 0
    ? ((1 - rlmMetrics.total_tokens / directMetrics.total_tokens) * 100).toFixed(0)
    : "0";
  const costSavings = directMetrics.cost_usd > 0
    ? ((1 - rlmMetrics.cost_usd / directMetrics.cost_usd) * 100).toFixed(0)
    : "0";

  return (
    <Alert variant="success">
      <TrendingDown className="h-4 w-4" aria-hidden="true" />
      <AlertTitle>Comparison Result</AlertTitle>
      <AlertDescription>
        RLM saves {tokenSavings}% tokens, {costSavings}% cost compared to Direct mode.
        ({rlmMetrics.total_tokens.toLocaleString()} vs {directMetrics.total_tokens.toLocaleString()} tokens,
        ${rlmMetrics.cost_usd.toFixed(4)} vs ${directMetrics.cost_usd.toFixed(4)})
      </AlertDescription>
    </Alert>
  );
}
