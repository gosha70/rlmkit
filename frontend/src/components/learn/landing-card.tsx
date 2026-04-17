"use client";

/**
 * Intent-driven Learn landing card.
 *
 * Whole-card Link (same pattern as ProviderCard) for a large target
 * and a simple a11y tree — one activatable name per card.
 */

import Link from "next/link";
import type { LucideIcon } from "lucide-react";
import { ArrowRight } from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface LandingCardProps {
  title: string;
  description: string;
  cta: string;
  href: string;
  icon?: LucideIcon;
}

export function LandingCard({
  title,
  description,
  cta,
  href,
  icon: Icon,
}: LandingCardProps) {
  return (
    <Link
      href={href}
      aria-label={`${title}: ${cta}`}
      className={cn(
        "group block rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
      )}
    >
      <Card className="h-full transition hover:border-primary/40 hover:bg-accent/50">
        <CardHeader className="p-4 pb-2">
          <div className="flex items-start gap-3">
            {Icon && (
              <Icon
                aria-hidden="true"
                className="mt-0.5 h-5 w-5 shrink-0 text-primary"
              />
            )}
            <div className="min-w-0">
              <h3 className="text-base font-semibold leading-tight">{title}</h3>
              <p className="mt-1 text-sm text-muted-foreground">{description}</p>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-4 pt-0">
          <span className="inline-flex items-center gap-1 text-xs font-medium text-primary group-hover:underline">
            {cta}
            <ArrowRight className="h-3 w-3" aria-hidden="true" />
          </span>
        </CardContent>
      </Card>
    </Link>
  );
}
