/**
 * Settings component tests.
 *
 * Covers: BudgetConfig.
 * ModelSelector, ConnectionTester, ProviderCard, and SettingsPage require
 * SWR or routing mocks — left as stubs.
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { vi, describe, test, expect } from "vitest";
import { BudgetConfig } from "@/components/settings/budget-config";
import type { BudgetConfig as BudgetConfigType } from "@/lib/api";

// ---------------------------------------------------------------------------
// BudgetConfig
// ---------------------------------------------------------------------------

const DEFAULT_CONFIG: BudgetConfigType = {
  max_steps: 16,
  max_tokens: 50000,
  max_cost_usd: 2.0,
  max_time_seconds: 30,
  max_recursion_depth: 5,
};

describe("BudgetConfig", () => {
  test("renders Max Steps label with current value", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    expect(screen.getByText(/Max Steps: 16/)).toBeInTheDocument();
  });

  test("renders Max Tokens input with current value", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    const input = screen.getByLabelText("Max Tokens") as HTMLInputElement;
    expect(input).toBeInTheDocument();
    expect(input.value).toBe("50000");
  });

  test("renders Max Cost input with current value", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    const input = screen.getByLabelText("Max Cost (USD)") as HTMLInputElement;
    expect(input).toBeInTheDocument();
    expect(input.value).toBe("2");
  });

  test("renders Max Time label with current value", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    expect(screen.getByText(/Max Time.*: 30/)).toBeInTheDocument();
  });

  test("renders Max Recursion Depth label with current value", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    expect(screen.getByText(/Max Recursion Depth: 5/)).toBeInTheDocument();
  });

  test("renders Save and Reset to Defaults buttons", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Save" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reset to Defaults" })).toBeInTheDocument();
  });

  test("calls onChange with current values when Save clicked", () => {
    const onChange = vi.fn();
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onChange).toHaveBeenCalledWith(DEFAULT_CONFIG);
  });

  test("updates max_tokens input and calls onChange on Save", () => {
    const onChange = vi.fn();
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={onChange} />);

    const input = screen.getByLabelText("Max Tokens");
    fireEvent.change(input, { target: { value: "100000" } });

    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onChange).toHaveBeenCalledWith({
      ...DEFAULT_CONFIG,
      max_tokens: 100000,
    });
  });

  test("updates max_cost_usd input and calls onChange on Save", () => {
    const onChange = vi.fn();
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={onChange} />);

    const input = screen.getByLabelText("Max Cost (USD)");
    fireEvent.change(input, { target: { value: "5" } });

    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(onChange).toHaveBeenCalledWith({
      ...DEFAULT_CONFIG,
      max_cost_usd: 5,
    });
  });

  test("Reset to Defaults calls onChange with default values", () => {
    const onChange = vi.fn();
    const customConfig: BudgetConfigType = {
      max_steps: 5,
      max_tokens: 10000,
      max_cost_usd: 0.5,
      max_time_seconds: 10,
      max_recursion_depth: 2,
    };
    render(<BudgetConfig config={customConfig} onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: "Reset to Defaults" }));
    expect(onChange).toHaveBeenCalledWith({
      max_steps: 16,
      max_tokens: 50000,
      max_cost_usd: 2.0,
      max_time_seconds: 30,
      max_recursion_depth: 5,
    });
  });

  test("Reset to Defaults updates displayed values", () => {
    const customConfig: BudgetConfigType = {
      max_steps: 5,
      max_tokens: 10000,
      max_cost_usd: 0.5,
      max_time_seconds: 10,
      max_recursion_depth: 2,
    };
    render(<BudgetConfig config={customConfig} onChange={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Reset to Defaults" }));

    const tokensInput = screen.getByLabelText("Max Tokens") as HTMLInputElement;
    expect(tokensInput.value).toBe("50000");
  });

  test("max-steps slider exists with aria-valuetext", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    // Radix Slider renders a <span role="slider"> with aria-valuenow; the
    // aria-label is on the parent container span, not the thumb.
    const sliders = screen.getAllByRole("slider");
    const stepsSlider = sliders.find(
      (s) => s.getAttribute("aria-valuenow") === "16"
    );
    expect(stepsSlider).toBeInTheDocument();
  });

  test("max-time slider exists with aria-valuenow", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    const sliders = screen.getAllByRole("slider");
    const timeSlider = sliders.find(
      (s) => s.getAttribute("aria-valuenow") === "30"
    );
    expect(timeSlider).toBeInTheDocument();
  });

  test("max-depth slider exists with aria-valuenow", () => {
    render(<BudgetConfig config={DEFAULT_CONFIG} onChange={vi.fn()} />);
    const sliders = screen.getAllByRole("slider");
    const depthSlider = sliders.find(
      (s) => s.getAttribute("aria-valuenow") === "5"
    );
    expect(depthSlider).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// ModelSelector (stubs — needs SWR mocking)
// ---------------------------------------------------------------------------

describe("ModelSelector", () => {
  test.todo("renders dropdown with available models");
  test.todo("shows current model as selected");
  test.todo("calls onChange when a model is selected");
  test.todo("disables models from unconfigured providers");
});

// ---------------------------------------------------------------------------
// ConnectionTester (stubs — needs fetch mocking)
// ---------------------------------------------------------------------------

describe("ConnectionTester", () => {
  test.todo("renders test button");
  test.todo("shows loading spinner during test");
  test.todo("shows success state with latency on successful test");
  test.todo("shows error message on failed test");
});

// ---------------------------------------------------------------------------
// ProviderCard (stubs — needs SWR mocking)
// ---------------------------------------------------------------------------

describe("ProviderCard", () => {
  test.todo("renders provider name and status");
  test.todo("shows API key input field");
  test.todo("masks API key value");
  test.todo("shows list of available models when configured");
  test.todo("shows not_configured badge when unconfigured");
});

// ---------------------------------------------------------------------------
// Settings page (stubs — needs SWR + routing)
// ---------------------------------------------------------------------------

describe("SettingsPage", () => {
  test.todo("renders provider cards section");
  test.todo("renders budget configuration section");
  test.todo("renders appearance/theme section");
  test.todo("fetches config on mount");
  test.todo("saves config changes via PUT /api/config");
});
