"use client";

/**
 * Single provider card on the Cookbook landing page.
 *
 * Rendered as a whole-card Link — the entire card is the CTA to
 * /learn/cookbook/[id]. Keeps the accessibility tree simple (one
 * activatable name per card) and gives a large click target.
 */

import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type {
  CookbookProvider,
  ProviderDifficulty,
} from "./provider-catalog";

interface ProviderCardProps {
  provider: CookbookProvider;
}

const DIFFICULTY_VARIANT: Record<
  ProviderDifficulty,
  "success" | "warning" | "outline"
> = {
  Easy: "success",
  Moderate: "warning",
  Advanced: "outline",
};

export function ProviderCard({ provider }: ProviderCardProps) {
  return (
    <Link
      href={`/learn/cookbook?provider=${encodeURIComponent(provider.id)}`}
      aria-label={`Open ${provider.name} guide (${provider.difficulty})`}
      className={cn(
        "group block rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
      )}
    >
      <Card className="h-full transition hover:border-primary/40 hover:bg-accent/50">
        <CardHeader className="p-4 pb-2">
          <div className="flex items-start justify-between gap-2">
            <h3 className="text-base font-semibold leading-tight">
              {provider.name}
            </h3>
            <Badge variant={DIFFICULTY_VARIANT[provider.difficulty]}>
              {provider.difficulty}
            </Badge>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            {provider.bestFor}
          </p>
        </CardHeader>
        <CardContent className="p-4 pt-0">
          <span className="inline-flex items-center text-xs font-medium text-primary group-hover:underline">
            Open guide →
          </span>
        </CardContent>
      </Card>
    </Link>
  );
}
