/**
 * Chat component tests.
 *
 * Covers: ChatInput, TypingIndicator, FileAttachment, ProviderBadge,
 * ChatProviderSelector.
 *
 * MessageBubble, ModeSelector, TraceInline, ComparisonBanner left as stubs
 * (require heavier mocking or are covered elsewhere).
 */

import { render, screen, fireEvent } from "@testing-library/react";
import { vi, describe, test, expect } from "vitest";
import { ChatInput } from "@/components/chat/chat-input";
import { TypingIndicator } from "@/components/chat/typing-indicator";
import { FileAttachment } from "@/components/chat/file-attachment";
import { ProviderBadge } from "@/components/chat/provider-badge";
import { ChatProviderSelector } from "@/components/chat/chat-provider-selector";
import type { ChatProviderConfig, ProviderInfo } from "@/lib/api";

// ---------------------------------------------------------------------------
// ChatInput
// ---------------------------------------------------------------------------

describe("ChatInput", () => {
  test("renders textarea and send button", () => {
    render(<ChatInput onSend={vi.fn()} />);
    expect(screen.getByLabelText("Message input")).toBeInTheDocument();
    expect(screen.getByLabelText("Send message")).toBeInTheDocument();
  });

  test("disables send button when input is empty", () => {
    render(<ChatInput onSend={vi.fn()} />);
    expect(screen.getByLabelText("Send message")).toBeDisabled();
  });

  test("enables send button when input has text", () => {
    render(<ChatInput onSend={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Message input"), {
      target: { value: "hello" },
    });
    expect(screen.getByLabelText("Send message")).not.toBeDisabled();
  });

  test("calls onSend with trimmed text on button click", () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);
    fireEvent.change(screen.getByLabelText("Message input"), {
      target: { value: "  hello world  " },
    });
    fireEvent.click(screen.getByLabelText("Send message"));
    expect(onSend).toHaveBeenCalledWith("hello world");
  });

  test("clears input after submit", () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByLabelText("Message input");
    fireEvent.change(textarea, { target: { value: "hello" } });
    fireEvent.click(screen.getByLabelText("Send message"));
    expect(textarea).toHaveValue("");
  });

  test("Enter key submits and clears input", () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByLabelText("Message input");
    fireEvent.change(textarea, { target: { value: "hello" } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
    expect(onSend).toHaveBeenCalledWith("hello");
    expect(textarea).toHaveValue("");
  });

  test("Shift+Enter does not submit", () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByLabelText("Message input");
    fireEvent.change(textarea, { target: { value: "hello" } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });
    expect(onSend).not.toHaveBeenCalled();
  });

  test("does not call onSend for whitespace-only input", () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);
    fireEvent.change(screen.getByLabelText("Message input"), {
      target: { value: "   " },
    });
    fireEvent.click(screen.getByLabelText("Send message"));
    expect(onSend).not.toHaveBeenCalled();
  });

  test("disables textarea and send button when disabled prop is true", () => {
    render(<ChatInput onSend={vi.fn()} disabled />);
    expect(screen.getByLabelText("Message input")).toBeDisabled();
    expect(screen.getByLabelText("Send message")).toBeDisabled();
  });

  test("shows upload button when onFileUpload provided", () => {
    render(<ChatInput onSend={vi.fn()} onFileUpload={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Upload file" })).toBeInTheDocument();
  });

  test("does not show upload button when onFileUpload not provided", () => {
    render(<ChatInput onSend={vi.fn()} />);
    expect(screen.queryByRole("button", { name: "Upload file" })).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// TypingIndicator
// ---------------------------------------------------------------------------

describe("TypingIndicator", () => {
  test("shows 'Generating...' with no props", () => {
    render(<TypingIndicator />);
    expect(screen.getByText("Generating...")).toBeInTheDocument();
  });

  test("shows mode in text when mode provided", () => {
    render(<TypingIndicator mode="rlm" />);
    expect(screen.getByText("Generating (rlm)...")).toBeInTheDocument();
  });

  test("shows step text when step provided", () => {
    render(<TypingIndicator step="Running code..." />);
    expect(screen.getByText("Running code...")).toBeInTheDocument();
  });

  test("step takes precedence over mode", () => {
    render(<TypingIndicator mode="rlm" step="Running code..." />);
    expect(screen.getByText("Running code...")).toBeInTheDocument();
    expect(screen.queryByText("Generating (rlm)...")).not.toBeInTheDocument();
  });

  test("has role='status' and aria-live='polite'", () => {
    render(<TypingIndicator />);
    const el = screen.getByRole("status");
    expect(el).toBeInTheDocument();
    expect(el).toHaveAttribute("aria-live", "polite");
  });
});

// ---------------------------------------------------------------------------
// FileAttachment
// ---------------------------------------------------------------------------

describe("FileAttachment", () => {
  test("renders file name", () => {
    render(
      <FileAttachment
        name="report.pdf"
        sizeBytes={2048}
        tokenCount={100}
        onRemove={vi.fn()}
      />
    );
    expect(screen.getByText("report.pdf")).toBeInTheDocument();
  });

  test("renders formatted file size in KB", () => {
    render(
      <FileAttachment
        name="report.pdf"
        sizeBytes={2048}
        tokenCount={0}
        onRemove={vi.fn()}
      />
    );
    expect(screen.getByText(/2\.0 KB/)).toBeInTheDocument();
  });

  test("renders file size in bytes for small files", () => {
    render(
      <FileAttachment
        name="tiny.txt"
        sizeBytes={500}
        tokenCount={0}
        onRemove={vi.fn()}
      />
    );
    expect(screen.getByText(/500 B/)).toBeInTheDocument();
  });

  test("renders file size in MB for large files", () => {
    render(
      <FileAttachment
        name="big.pdf"
        sizeBytes={2 * 1024 * 1024}
        tokenCount={0}
        onRemove={vi.fn()}
      />
    );
    expect(screen.getByText(/2\.0 MB/)).toBeInTheDocument();
  });

  test("renders token count when > 0", () => {
    render(
      <FileAttachment
        name="doc.txt"
        sizeBytes={1024}
        tokenCount={1500}
        onRemove={vi.fn()}
      />
    );
    expect(screen.getByText(/1,500 tokens/)).toBeInTheDocument();
  });

  test("does not render token count when 0", () => {
    render(
      <FileAttachment
        name="doc.txt"
        sizeBytes={1024}
        tokenCount={0}
        onRemove={vi.fn()}
      />
    );
    expect(screen.queryByText(/tokens/)).not.toBeInTheDocument();
  });

  test("has remove button with correct aria-label", () => {
    render(
      <FileAttachment
        name="report.pdf"
        sizeBytes={1024}
        tokenCount={0}
        onRemove={vi.fn()}
      />
    );
    expect(screen.getByLabelText("Remove file report.pdf")).toBeInTheDocument();
  });

  test("calls onRemove when remove button clicked", () => {
    const onRemove = vi.fn();
    render(
      <FileAttachment
        name="report.pdf"
        sizeBytes={1024}
        tokenCount={0}
        onRemove={onRemove}
      />
    );
    fireEvent.click(screen.getByLabelText("Remove file report.pdf"));
    expect(onRemove).toHaveBeenCalledOnce();
  });
});

// ---------------------------------------------------------------------------
// ProviderBadge
// ---------------------------------------------------------------------------

describe("ProviderBadge", () => {
  test("renders provider name alone when no model", () => {
    render(<ProviderBadge provider="openai" />);
    expect(screen.getByText("openai")).toBeInTheDocument();
  });

  test("renders 'model via provider' when model provided", () => {
    render(<ProviderBadge provider="openai" model="gpt-4o" />);
    expect(screen.getByText("gpt-4o via openai")).toBeInTheDocument();
  });

  test("has aria-label describing provider and status", () => {
    const { container } = render(
      <ProviderBadge provider="openai" status="connected" />
    );
    const badge = container.firstChild as HTMLElement;
    expect(badge).toHaveAttribute("aria-label", "openai - connected");
  });

  test("aria-label includes model when provided", () => {
    const { container } = render(
      <ProviderBadge provider="openai" model="gpt-4o" status="offline" />
    );
    const badge = container.firstChild as HTMLElement;
    expect(badge).toHaveAttribute("aria-label", "gpt-4o via openai - offline");
  });

  test("defaults to connected status", () => {
    const { container } = render(<ProviderBadge provider="openai" />);
    const badge = container.firstChild as HTMLElement;
    expect(badge).toHaveAttribute("aria-label", "openai - connected");
  });
});

// ---------------------------------------------------------------------------
// ChatProviderSelector
// ---------------------------------------------------------------------------

const makeCP = (overrides: Partial<ChatProviderConfig> = {}): ChatProviderConfig => ({
  id: "cp-1",
  name: "GPT-4o Direct",
  llm_provider: "openai",
  llm_model: "gpt-4o",
  execution_mode: "direct",
  runtime_settings: {
    temperature: 0.7,
    top_p: 1,
    max_output_tokens: 2048,
    timeout_seconds: 30,
  },
  rlm_max_steps: 16,
  rlm_timeout_seconds: 30,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
  ...overrides,
});

const makeProvider = (overrides: Partial<ProviderInfo> = {}): ProviderInfo => ({
  name: "openai",
  display_name: "OpenAI",
  status: "connected",
  models: [],
  default_model: "gpt-4o",
  configured: true,
  requires_api_key: true,
  default_endpoint: null,
  model_input_hint: "",
  masked_api_key: "sk-...xxxx",
  ...overrides,
});

describe("ChatProviderSelector", () => {
  test("shows empty state when no available chat providers", () => {
    // Provider is not connected → chatProvider filtered out
    render(
      <ChatProviderSelector
        chatProviders={[makeCP()]}
        providers={[makeProvider({ status: "not_configured" })]}
        selectedIds={[]}
        onSelectionChange={vi.fn()}
      />
    );
    expect(
      screen.getByText(/No Chat Providers available/)
    ).toBeInTheDocument();
  });

  test("renders chip for each available provider", () => {
    const cp1 = makeCP({ id: "cp-1", name: "GPT-4o Direct" });
    const cp2 = makeCP({ id: "cp-2", name: "Claude Direct", llm_provider: "anthropic" });
    const p1 = makeProvider({ name: "openai" });
    const p2 = makeProvider({ name: "anthropic" });

    render(
      <ChatProviderSelector
        chatProviders={[cp1, cp2]}
        providers={[p1, p2]}
        selectedIds={["cp-1"]}
        onSelectionChange={vi.fn()}
      />
    );

    expect(screen.getByRole("checkbox", { name: /GPT-4o Direct/ })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /Claude Direct/ })).toBeInTheDocument();
  });

  test("shows All/1 toggle when more than one provider", () => {
    const cp1 = makeCP({ id: "cp-1", name: "GPT-4o Direct" });
    const cp2 = makeCP({ id: "cp-2", name: "Claude Direct", llm_provider: "anthropic" });
    const p1 = makeProvider({ name: "openai" });
    const p2 = makeProvider({ name: "anthropic" });

    render(
      <ChatProviderSelector
        chatProviders={[cp1, cp2]}
        providers={[p1, p2]}
        selectedIds={["cp-1"]}
        onSelectionChange={vi.fn()}
      />
    );

    // When not all selected, shows "All"
    expect(screen.getByRole("button", { name: "Select all providers" })).toBeInTheDocument();
  });

  test("toggle button shows '1' when all selected", () => {
    const cp1 = makeCP({ id: "cp-1", name: "GPT-4o Direct" });
    const cp2 = makeCP({ id: "cp-2", name: "Claude Direct", llm_provider: "anthropic" });
    const p1 = makeProvider({ name: "openai" });
    const p2 = makeProvider({ name: "anthropic" });

    render(
      <ChatProviderSelector
        chatProviders={[cp1, cp2]}
        providers={[p1, p2]}
        selectedIds={["cp-1", "cp-2"]}
        onSelectionChange={vi.fn()}
      />
    );

    expect(screen.getByRole("button", { name: "Select first only" })).toBeInTheDocument();
  });

  test("calls onSelectionChange when a provider chip is clicked", () => {
    const cp1 = makeCP({ id: "cp-1", name: "GPT-4o Direct" });
    const cp2 = makeCP({ id: "cp-2", name: "Claude Direct", llm_provider: "anthropic" });
    const p1 = makeProvider({ name: "openai" });
    const p2 = makeProvider({ name: "anthropic" });
    const onSelectionChange = vi.fn();

    render(
      <ChatProviderSelector
        chatProviders={[cp1, cp2]}
        providers={[p1, p2]}
        selectedIds={["cp-1"]}
        onSelectionChange={onSelectionChange}
      />
    );

    fireEvent.click(screen.getByRole("checkbox", { name: /Claude Direct/ }));
    expect(onSelectionChange).toHaveBeenCalledWith(["cp-1", "cp-2"]);
  });

  test("does not deselect the last selected provider", () => {
    const cp1 = makeCP({ id: "cp-1", name: "GPT-4o Direct" });
    const p1 = makeProvider({ name: "openai" });
    const onSelectionChange = vi.fn();

    render(
      <ChatProviderSelector
        chatProviders={[cp1]}
        providers={[p1]}
        selectedIds={["cp-1"]}
        onSelectionChange={onSelectionChange}
      />
    );

    // The single selected chip should be disabled
    const chip = screen.getByRole("checkbox", { name: /GPT-4o Direct/ });
    expect(chip).toBeDisabled();
  });

  test("All toggle selects all providers", () => {
    const cp1 = makeCP({ id: "cp-1", name: "GPT-4o Direct" });
    const cp2 = makeCP({ id: "cp-2", name: "Claude Direct", llm_provider: "anthropic" });
    const p1 = makeProvider({ name: "openai" });
    const p2 = makeProvider({ name: "anthropic" });
    const onSelectionChange = vi.fn();

    render(
      <ChatProviderSelector
        chatProviders={[cp1, cp2]}
        providers={[p1, p2]}
        selectedIds={["cp-1"]}
        onSelectionChange={onSelectionChange}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Select all providers" }));
    expect(onSelectionChange).toHaveBeenCalledWith(["cp-1", "cp-2"]);
  });

  test("1 toggle deselects to first provider only", () => {
    const cp1 = makeCP({ id: "cp-1", name: "GPT-4o Direct" });
    const cp2 = makeCP({ id: "cp-2", name: "Claude Direct", llm_provider: "anthropic" });
    const p1 = makeProvider({ name: "openai" });
    const p2 = makeProvider({ name: "anthropic" });
    const onSelectionChange = vi.fn();

    render(
      <ChatProviderSelector
        chatProviders={[cp1, cp2]}
        providers={[p1, p2]}
        selectedIds={["cp-1", "cp-2"]}
        onSelectionChange={onSelectionChange}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Select first only" }));
    expect(onSelectionChange).toHaveBeenCalledWith(["cp-1"]);
  });

  test("does not show All/1 toggle when only one provider available", () => {
    const cp1 = makeCP({ id: "cp-1", name: "GPT-4o Direct" });
    const p1 = makeProvider({ name: "openai" });

    render(
      <ChatProviderSelector
        chatProviders={[cp1]}
        providers={[p1]}
        selectedIds={["cp-1"]}
        onSelectionChange={vi.fn()}
      />
    );

    expect(screen.queryByRole("button", { name: /Select/ })).not.toBeInTheDocument();
  });

  test("has group role with aria-label", () => {
    const cp1 = makeCP();
    const p1 = makeProvider();

    render(
      <ChatProviderSelector
        chatProviders={[cp1]}
        providers={[p1]}
        selectedIds={["cp-1"]}
        onSelectionChange={vi.fn()}
      />
    );

    expect(screen.getByRole("group", { name: "Chat providers" })).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// MessageBubble (stubs — requires markdown renderer mocking)
// ---------------------------------------------------------------------------

describe("MessageBubble", () => {
  test.todo("renders user message with correct styling");
  test.todo("renders assistant message with correct styling");
  test.todo("renders markdown content in assistant messages");
  test.todo("displays execution metrics when provided");
  test.todo("shows mode badge (direct/rlm/rag)");
  test.todo("shows loading state for in-progress messages");
});

// ---------------------------------------------------------------------------
// ModeSelector (stubs — requires radix tooltip mocking)
// ---------------------------------------------------------------------------

describe("ModeSelector", () => {
  test.todo("renders all mode options (auto, direct, rlm, rag, compare)");
  test.todo("highlights currently selected mode");
  test.todo("calls onChange when a mode is selected");
  test.todo("shows tooltip with mode description on hover");
});

// ---------------------------------------------------------------------------
// TraceInline (stubs)
// ---------------------------------------------------------------------------

describe("TraceInline", () => {
  test.todo("renders step count summary");
  test.todo("expands to show step details on click");
  test.todo("displays code blocks for code steps");
});

// ---------------------------------------------------------------------------
// ComparisonBanner (stubs)
// ---------------------------------------------------------------------------

describe("ComparisonBanner", () => {
  test.todo("renders side-by-side results for direct and RLM modes");
  test.todo("highlights the winning mode based on metrics");
});
