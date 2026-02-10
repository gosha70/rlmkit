/**
 * Settings component test stubs.
 *
 * Covers: ModelSelector, ConnectionTester, ProviderCard,
 * BudgetConfig, and the Settings page.
 */

import { describe, test } from "vitest";

// ---------------------------------------------------------------------------
// ModelSelector
// ---------------------------------------------------------------------------

describe("ModelSelector", () => {
  test.todo("renders dropdown with available models");
  test.todo("shows current model as selected");
  test.todo("calls onChange when a model is selected");
  test.todo("disables models from unconfigured providers");
});

// ---------------------------------------------------------------------------
// ConnectionTester
// ---------------------------------------------------------------------------

describe("ConnectionTester", () => {
  test.todo("renders test button");
  test.todo("shows loading spinner during test");
  test.todo("shows success state with latency on successful test");
  test.todo("shows error message on failed test");
});

// ---------------------------------------------------------------------------
// ProviderCard
// ---------------------------------------------------------------------------

describe("ProviderCard", () => {
  test.todo("renders provider name and status");
  test.todo("shows API key input field");
  test.todo("masks API key value");
  test.todo("shows list of available models when configured");
  test.todo("shows not_configured badge when unconfigured");
});

// ---------------------------------------------------------------------------
// BudgetConfig
// ---------------------------------------------------------------------------

describe("BudgetConfig", () => {
  test.todo("renders sliders for max_steps, max_tokens, max_cost, max_time");
  test.todo("shows current values next to sliders");
  test.todo("calls onChange when slider values change");
  test.todo("validates min/max bounds on inputs");
});

// ---------------------------------------------------------------------------
// Settings page
// ---------------------------------------------------------------------------

describe("SettingsPage", () => {
  test.todo("renders provider cards section");
  test.todo("renders budget configuration section");
  test.todo("renders appearance/theme section");
  test.todo("fetches config on mount");
  test.todo("saves config changes via PUT /api/config");
});
