import { CheckCircle2, XCircle, AlertTriangle, Loader2, Circle } from "lucide-react";
import { cn } from "@/lib/utils";

type Status = "connected" | "offline" | "warning" | "loading" | "not_configured";

interface StatusIndicatorProps {
  status: Status;
  label?: string;
  className?: string;
}

const CONFIG: Record<Status, { icon: typeof CheckCircle2; color: string; dotColor: string; text: string }> = {
  connected: { icon: CheckCircle2, color: "text-emerald-600 dark:text-emerald-400", dotColor: "bg-emerald-500", text: "Connected" },
  offline: { icon: XCircle, color: "text-red-600 dark:text-red-400", dotColor: "bg-red-500", text: "Offline" },
  warning: { icon: AlertTriangle, color: "text-amber-600 dark:text-amber-400", dotColor: "bg-amber-500", text: "Slow" },
  loading: { icon: Loader2, color: "text-muted-foreground", dotColor: "bg-muted-foreground", text: "Loading..." },
  not_configured: { icon: Circle, color: "text-muted-foreground", dotColor: "bg-muted-foreground", text: "Not configured" },
};

export function StatusIndicator({ status, label, className }: StatusIndicatorProps) {
  const { icon: Icon, color, dotColor, text } = CONFIG[status];
  const isLoading = status === "loading";

  return (
    <div
      className={cn("flex items-center gap-1.5 text-xs", className)}
      role="status"
      aria-label={`${label ?? ""} status: ${text}`}
    >
      <span className={cn("inline-block h-2 w-2 rounded-full", dotColor, isLoading && "animate-pulse")} />
      <Icon className={cn("h-3.5 w-3.5", color, isLoading && "animate-spin")} />
      <span className={color}>{label ? `${label}: ${text}` : text}</span>
    </div>
  );
}
