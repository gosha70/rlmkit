# Spec 03: Frontend — Chat Providers Settings Tab

## Goal

Replace the old "Execution" tab in Settings with a "Chat Providers" management tab where users create, edit, and delete Chat Providers.

---

## 1. Tab Structure

### File: `frontend/src/app/settings/page.tsx`

The Settings page uses shadcn `Tabs` with these tabs:

```
Providers | Chat Providers | Budget | Profiles | Prompts | Appearance
```

The "Chat Providers" tab replaces the former "Execution" tab. All other tabs remain unchanged.

---

## 2. Data Fetching

Use SWR with key `"chat-providers"`:

```typescript
const { data: chatProviders = [], mutate: mutateChatProviders } = useSWR<ChatProviderConfig[]>(
  "chat-providers",
  getChatProviders
);
```

Also fetch providers list (needed for the LLM Provider and Model dropdowns):

```typescript
const { data: providers = [] } = useSWR<ProviderInfo[]>("providers", getProviders);
```

---

## 3. Chat Provider List

When the form is not open, show:

- A **"Create Chat Provider"** button at the top
- A list of existing Chat Providers, each rendered as a `Card`:
  - **Header**: Provider name (bold), subtitle `{llm_provider} · {llm_model}`
  - **Badges**: Execution mode pill (e.g., `direct`, `rlm`, `rag`)
  - **Actions**: Edit (pencil icon) and Delete (trash icon) buttons
- Empty state: "No Chat Providers configured yet. Create one to get started."

---

## 4. Create/Edit Form

Clicking "Create" or "Edit" shows an inline form (card) with these fields:

### 4.1 Required Fields

| Field | Input Type | Notes |
|-------|-----------|-------|
| **Name** | Text input | e.g., "DIRECT-CLAUDE", "RLM-GPT4". Must be unique. |
| **LLM Provider** | Select dropdown | Populated from `providers` list. Shows `display_name`. Changing this resets the Model field. |
| **Model** | Select dropdown | Populated from selected provider's `models[]` array. Disabled until a provider is selected. |
| **Execution Mode** | Radio group | Options: `direct`, `rlm`, `rag` |

### 4.2 Mode-Specific Settings (conditional)

**When mode is `rlm`**, show an "RLM Settings" section:

| Field | Input Type | Default |
|-------|-----------|---------|
| Max Steps | Number input, min=1 | 10 |
| Timeout (seconds) | Number input, min=1 | 60 |

**When mode is `rag`**, show a "RAG Settings" section:

| Field | Input Type | Default |
|-------|-----------|---------|
| Chunk Size | Number input, min=1 | 512 |
| Chunk Overlap | Number input, min=0 | 64 |
| Top K | Number input, min=1 | 5 |
| Embedding Model | Text input | `text-embedding-3-small` |

### 4.3 Runtime Settings (always shown)

Shown in a bordered section labeled "Runtime Settings":

| Field | Input Type | Default | Range |
|-------|-----------|---------|-------|
| Temperature | Number input, step=0.1 | 0.7 | 0–2 |
| Top P | Number input, step=0.1 | 0.9 | 0–1 |
| Max Output Tokens | Number input, min=1 | 2048 | — |
| Timeout (seconds) | Number input, min=1 | 30 | — |

### 4.4 Form Actions

- **Create/Update** button: Disabled when name, provider, or model is empty. Calls `createChatProvider()` or `updateChatProvider()`.
- **Cancel** button: Closes form, resets state.

### 4.5 Form State Management

```typescript
const [showCreateForm, setShowCreateForm] = useState(false);
const [editingId, setEditingId] = useState<string | null>(null);
const [formData, setFormData] = useState({
  name: "",
  llm_provider: "",
  llm_model: "",
  execution_mode: "direct" as "direct" | "rlm" | "rag",
  runtime_settings: { temperature: 0.7, top_p: 0.9, max_output_tokens: 2048, timeout_seconds: 30 },
  rag_config: { chunk_size: 512, chunk_overlap: 64, top_k: 5, embedding_model: "text-embedding-3-small" },
  rlm_max_steps: 10,
  rlm_timeout_seconds: 60,
});
```

---

## 5. CRUD Operations

### Create
```typescript
const payload: ChatProviderCreateRequest = {
  name: formData.name,
  llm_provider: formData.llm_provider,
  llm_model: formData.llm_model,
  execution_mode: formData.execution_mode,
  runtime_settings: formData.runtime_settings,
  rlm_max_steps: formData.rlm_max_steps,
  rlm_timeout_seconds: formData.rlm_timeout_seconds,
};
if (formData.execution_mode === "rag") {
  payload.rag_config = formData.rag_config;
}
await createChatProvider(payload);
mutateChatProviders();
```

### Edit
Pre-populate form from existing provider, then call `updateChatProvider(editingId, payload)`.

### Delete
Confirm with `window.confirm()`, then call `deleteChatProvider(id)` and `mutateChatProviders()`.

---

## 6. Components Used

From shadcn/ui:
- `Tabs`, `TabsContent`, `TabsList`, `TabsTrigger`
- `Card`, `CardContent`, `CardHeader`, `CardTitle`
- `Input`, `Label`, `Button`
- `Select`, `SelectContent`, `SelectItem`, `SelectTrigger`, `SelectValue`
- `RadioGroup`, `RadioGroupItem`

From lucide-react:
- `Plus`, `Edit2`, `Trash2`

---

## 7. Imports Required

```typescript
import {
  getChatProviders,
  createChatProvider,
  updateChatProvider,
  deleteChatProvider,
  type ChatProviderConfig,
  type ChatProviderCreateRequest,
  type RuntimeSettings,
  type RAGConfig,
} from "@/lib/api";
```

---

## Acceptance Criteria

1. "Chat Providers" tab is visible and functional in Settings
2. Empty state shown when no Chat Providers exist
3. Create form opens inline with all fields described above
4. Mode-specific settings appear/hide based on execution mode selection
5. Creating a Chat Provider calls the API, closes the form, and refreshes the list
6. Editing pre-populates the form and calls update API on save
7. Deleting prompts confirmation and removes from list
8. Provider dropdown shows all providers from the catalog (not just connected ones — user may configure API key later)
9. Model dropdown is populated from the selected provider's model list
10. TypeScript compiles without errors (`npx tsc --noEmit`)
