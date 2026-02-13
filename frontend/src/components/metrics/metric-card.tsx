import { type LucideIcon } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  label: string;
  value: string;
  unit?: string;
  icon: LucideIcon;
  className?: string;
}

export function MetricCard({ label, value, unit, icon: Icon, className }: MetricCardProps) {
  return (
    <Card className={cn("", className)} aria-label={`${label}: ${value}${unit ? ` ${unit}` : ""}`}>
      <CardContent className="flex items-center gap-4 p-6">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-primary/10" aria-hidden="true">
          <Icon className="h-6 w-6 text-primary" />
        </div>
        <div>
          <p className="text-sm text-muted-foreground">{label}</p>
          <p className="text-2xl font-bold">
            {value}
            {unit && <span className="text-sm font-normal text-muted-foreground ml-1">{unit}</span>}
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
