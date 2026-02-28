# Spec 04: Frontend — Chat Page Multi-Provider Column Layout

## Goal

Refactor the Chat page to replace the single-mode selector with a multi-select Chat Provider selector. When the user sends a query, execute it against all selected Chat Providers in parallel and display responses in a column layout with per-response metrics.

---

## 1. New Component: ChatProviderSelector

### File: `frontend/src/components/chat/chat-provider-selector.tsx`

A multi-select pill group replacing the old `ModeSelector`.

#### Props

```typescript
interface ChatProviderSelectorProps {
  chatProviders: ChatProviderConfig[];
  providers: ProviderInfo[];        // LLM provider info for status checking
  selectedIds: string[];
  onSelectionChange: (ids: string[]) => void;
  disabled?: boolean;
}
```

#### Behavior

1. **Filter**: Only show Chat Providers whose underlying `llm_provider` has status `"connected"` or `"configured"` in the `providers` list
2. **Pills**: Each pill shows:
   - Provider name (e.g., "DIRECT-CLAUDE")
   - Mode badge: `D` for direct, `R` for rlm, `G` for rag
   - Status dot: green (connected), amber (configured), gray (unavailable)
3. **Toggle**: Click to select/deselect. Prevent deselecting the last provider (minimum 1 must be selected)
4. **Accessibility**: Use `role="checkbox"` and `aria-checked` on each pill, `role="group"` on container
5. **Empty state**: Show "No Chat Providers available. Configure one in Settings."

---

## 2. Chat Page State

### File: `frontend/src/app/page.tsx`

#### Core State

```typescript
// Session
const [sessionId, setSessionId] = useState<string | null>(() => {
  return localStorage.getItem("rlmkit_active_session") || null;
});

// Selected Chat Providers (persisted to localStorage)
const [selectedChatProviderIds, setSelectedChatProviderIds] = useState<string[]>(() => {
  const saved = localStorage.getItem("rlmkit_selected_chat_providers");
  return saved ? JSON.parse(saved) : [];
});

// Conversation turns
const [turns, setTurns] = useState<ChatTurn[]>([]);

// Polling state
const [isAnyPolling, setIsAnyPolling] = useState(false);
const pollingStateRef = useRef<Record<string, ReturnType<typeof setInterval>>>({});
```

#### ChatTurn Interface

```typescript
interface ChatTurn {
  id: string;
  userMessage: {
    content: string;
    timestamp: string;
  };
  responses: Record<string, {       // keyed by chatProviderId
    chatProviderId: string;
    chatProviderName: string;
    executionId: string;
    content: string;
    isStreaming: boolean;
    metrics?: MessageMetrics;
  }>;
}
```

#### Data Fetching

```typescript
const { data: chatProviders = [] } = useSWR<ChatProviderConfig[]>("chat-providers", getChatProviders);
const { data: providers = [] } = useSWR<ProviderInfo[]>("providers", getProviders);
const { data: config } = useSWR<AppConfig>("config", getConfig);
```

---

## 3. Sending Messages (Parallel Execution)

When the user sends a message via `handleSend(text)`:

1. **Create a new turn** with empty responses for each selected provider:
   ```typescript
   const newTurn: ChatTurn = {
     id: crypto.randomUUID(),
     userMessage: { content: text, timestamp: new Date().toISOString() },
     responses: {},
   };
   for (const cpId of selectedChatProviderIds) {
     newTurn.responses[cpId] = {
       chatProviderId: cpId,
       chatProviderName: chatProviders.find(cp => cp.id === cpId)?.name || cpId,
       executionId: "",
       content: "",
       isStreaming: true,
     };
   }
   ```

2. **Submit in parallel** — one `POST /api/chat` per selected Chat Provider:
   ```typescript
   const promises = selectedChatProviderIds.map(async (cpId) => {
     const resp = await submitChat({
       query: text,
       content: uploadedFile ? null : text || null,
       file_id: uploadedFile?.id || null,
       chat_provider_id: cpId,
       session_id: sessionId,
     });
     // Update turn with execution_id
     // Start polling for this execution
     pollForResult(resp.execution_id, cpId, turnId);
   });
   await Promise.allSettled(promises);
   ```

3. **Track session ID** — on first message, capture `session_id` from the first successful response and set it. Skip session load to avoid re-fetching.

---

## 4. Polling for Results

For each execution, poll `GET /api/traces/{executionId}` every 1 second:

```typescript
const pollForResult = (executionId: string, chatProviderId: string, turnId: string) => {
  const interval = setInterval(async () => {
    const trace = await getTrace(executionId);

    if (trace.status === "complete") {
      clearInterval(interval);
      delete pollingStateRef.current[executionId];
      // Update turn's response with content and metrics
      updateTurnResponse(turnId, chatProviderId, {
        content: trace.result.answer,
        isStreaming: false,
        metrics: {
          input_tokens: trace.budget.tokens_used,
          output_tokens: 0,
          total_tokens: trace.budget.tokens_used,
          cost_usd: trace.budget.cost_used,
          elapsed_seconds: computeElapsed(trace),
          steps: trace.budget.steps_used,
        },
      });
    } else if (trace.status === "error") {
      clearInterval(interval);
      delete pollingStateRef.current[executionId];
      // Update turn's response with error
      updateTurnResponse(turnId, chatProviderId, {
        content: `Error: ${trace.result?.answer || "Execution failed"}`,
        isStreaming: false,
      });
    }
    // "running" status → keep polling
  }, 1000);

  pollingStateRef.current[executionId] = interval;
};
```

Track `isAnyPolling` based on whether `pollingStateRef.current` has any entries.

---

## 5. Loading Existing Sessions

When `sessionId` changes, fetch `getSession(sessionId)` and reconstruct turns:

### From `session.conversations` (preferred)

1. Iterate each `[chatProviderId, messages[]]` in `session.conversations`
2. For each user message, find or create a turn matching that content
3. For each assistant message, attach it to the corresponding turn's `responses[chatProviderId]`

### Fallback: from `session.messages` (legacy flat list)

1. Walk messages sequentially
2. Each `role: "user"` starts a new turn
3. Each `role: "assistant"` with `chat_provider_id` attaches to the current turn's responses

---

## 6. Message Display Layout

### Structure

```
┌────────────────────────────────────────────────────────┐
│ [ChatProviderSelector: multi-select pills]              │  ← Toolbar
├────────────────────────────────────────────────────────┤
│                                                        │
│  User Message (full width)                             │
│                                                        │
│  ┌──────────────┬──────────────┬──────────────┐        │
│  │ DIRECT-CLAUDE│ RLM-CLAUDE   │ RAG-OPENAI   │        │  ← Grid columns
│  │              │              │              │        │
│  │ [markdown]   │ [markdown]   │ [markdown]   │        │
│  │              │              │              │        │
│  │ ───────────  │ ───────────  │ ───────────  │        │
│  │ 386 tokens   │ 1.2k tokens  │ 520 tokens   │        │  ← Metrics
│  │ $0.002 · 2.1s│ $0.01 · 9.7s│ $0.005 · 3.2s│        │
│  └──────────────┴──────────────┴──────────────┘        │
│                                                        │
│  User Message (full width)                             │
│  ┌──────────────┬──────────────┬──────────────┐        │
│  │ ...          │ ...          │ ...          │        │
│  └──────────────┴──────────────┴──────────────┘        │
│                                                        │
├────────────────────────────────────────────────────────┤
│ [ChatInput: text field + file upload + send button]     │  ← Input bar
└────────────────────────────────────────────────────────┘
```

### CSS Grid

```typescript
<div
  className="grid gap-4"
  style={{ gridTemplateColumns: `repeat(${selectedChatProviderIds.length}, 1fr)` }}
>
  {selectedChatProviderIds.map((cpId) => {
    const resp = turn.responses[cpId];
    return (
      <div key={cpId} className="rounded-lg border bg-muted/30 p-4">
        {/* Provider name + mode badge */}
        <div className="mb-2 flex items-center gap-2">
          <Bot className="h-3 w-3" />
          <span className="text-xs font-medium text-muted-foreground">{resp?.chatProviderName}</span>
        </div>

        {/* Response content or loading */}
        {resp?.content ? (
          <>
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{resp.content}</ReactMarkdown>
            </div>
            {resp.metrics && (
              <div className="mt-3 flex gap-4 border-t pt-2 text-xs text-muted-foreground">
                <span>{resp.metrics.total_tokens} tokens</span>
                <span>${resp.metrics.cost_usd.toFixed(4)}</span>
                <span>{resp.metrics.elapsed_seconds.toFixed(1)}s</span>
              </div>
            )}
          </>
        ) : resp ? (
          <TypingIndicator />
        ) : null}
      </div>
    );
  })}
</div>
```

### Metrics Display

Each response card shows below the content:
- **Tokens**: `{total_tokens} tokens`
- **Cost**: `${cost_usd}` (4 decimal places)
- **Time**: `{elapsed_seconds}s` (1 decimal place)
- **Steps**: Only for RLM mode (optional)

---

## 7. Empty State

When no turns exist, show a centered welcome message with the ChatInput:

```
Welcome to RLM Studio
Upload a document and ask questions. The RLM engine will recursively
explore the content using code generation to find answers efficiently.

[ChatInput]
```

---

## 8. Session Management

- **Persist** `sessionId` to `localStorage.rlmkit_active_session`
- **Persist** `selectedChatProviderIds` to `localStorage.rlmkit_selected_chat_providers`
- **Auto-select**: If no providers selected but `chatProviders` is non-empty, select the first one
- **New Session**: Clear `sessionId`, `turns`, `uploadedFile`, stop all polling
- **File Upload**: Uses existing `uploadFile()` API, stores `FileUploadResponse` in state

---

## 9. Cleanup

On unmount, clear all polling intervals:

```typescript
useEffect(() => {
  return () => {
    Object.values(pollingStateRef.current).forEach(clearInterval);
  };
}, []);
```

---

## 10. Components Required

### Existing (from repo)
- `AppShell` — layout wrapper with sidebar (passes `activeSessionId`, `onSelectSession`, `onNewSession`)
- `ChatInput` — text input with file upload and send button (props: `onSend`, `onFileUpload`, `disabled`)
- `FileAttachment` — file info display (props: `name`, `sizeBytes`, `tokenCount`, `onRemove`)
- `TypingIndicator` — loading animation

### New
- `ChatProviderSelector` — multi-select pills (see Section 1)

### Not used (deprecated by this feature)
- `ModeSelector` — replaced by `ChatProviderSelector`
- `MessageBubble` — replaced by inline rendering in the column grid (simpler, more control)

---

## 11. Dependencies

```typescript
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { User, Bot } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
```

Ensure `react-markdown` and `remark-gfm` are in `package.json`. If not:
```bash
npm install react-markdown remark-gfm
```

---

## Acceptance Criteria

1. ChatProviderSelector shows available Chat Providers with mode badges and status dots
2. Only providers with connected/configured LLM providers are shown
3. Multi-select works: click toggles selection; cannot deselect last provider
4. Sending a message submits N parallel requests (one per selected Chat Provider)
5. Responses appear in a column grid with one column per selected provider
6. Each column shows: provider name, markdown-rendered response, metrics (tokens, cost, time)
7. Loading state shows `TypingIndicator` in each column while awaiting results
8. Error responses show inline error message
9. Session loading reconstructs turns correctly from `conversations` dict
10. Selected providers persist across page reloads (localStorage)
11. New Session button clears all state
12. When only 1 Chat Provider is selected, it renders as a single full-width column
13. When multiple providers are selected, columns are equal width
14. TypeScript compiles without errors (`npx tsc --noEmit`)
15. No regressions in existing components (file upload, session sidebar)
