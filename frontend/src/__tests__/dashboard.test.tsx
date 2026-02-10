/**
 * Dashboard and metrics component test stubs.
 *
 * Covers: MetricCard, ComparisonChart, PerformanceTrend,
 * CostBreakdown, and the Dashboard page.
 */

import { describe, test } from "vitest";

// ---------------------------------------------------------------------------
// MetricCard
// ---------------------------------------------------------------------------

describe("MetricCard", () => {
  test.todo("renders label and value");
  test.todo("formats large numbers with comma separators");
  test.todo("shows trend indicator (up/down arrow) when delta provided");
  test.todo("applies correct color for positive vs negative trends");
});

// ---------------------------------------------------------------------------
// ComparisonChart
// ---------------------------------------------------------------------------

describe("ComparisonChart", () => {
  test.todo("renders bar chart with direct and RLM data");
  test.todo("shows legend for each mode");
  test.todo("handles empty data gracefully");
});

// ---------------------------------------------------------------------------
// PerformanceTrend
// ---------------------------------------------------------------------------

describe("PerformanceTrend", () => {
  test.todo("renders line chart with timeline data");
  test.todo("shows tooltip on hover with exact values");
  test.todo("handles single data point without error");
});

// ---------------------------------------------------------------------------
// CostBreakdown
// ---------------------------------------------------------------------------

describe("CostBreakdown", () => {
  test.todo("renders cost by provider");
  test.todo("renders cost by mode");
  test.todo("formats USD values to 4 decimal places");
});

// ---------------------------------------------------------------------------
// Dashboard page
// ---------------------------------------------------------------------------

describe("DashboardPage", () => {
  test.todo("renders summary metrics cards");
  test.todo("renders charts section");
  test.todo("shows empty state when no sessions exist");
  test.todo("fetches metrics data on mount");
});
