"use client";

import { useTheme } from "next-themes";

/**
 * Returns a Recharts-compatible `contentStyle` object that adapts to the
 * current colour scheme: inverted background (foreground colour) with matching
 * text so the tooltip is always high-contrast over the chart surface.
 */
export function useChartTooltipStyle(): React.CSSProperties {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";
  return {
    backgroundColor: "#f8fafc",
    color: "#0f172a",
    border: `1px solid ${isDark ? "#475569" : "#cbd5e1"}`,
    borderRadius: "6px",
    fontSize: "12px",
    boxShadow: isDark
      ? "0 4px 16px rgba(0,0,0,0.6)"
      : "0 4px 12px rgba(0,0,0,0.12)",
  };
}
