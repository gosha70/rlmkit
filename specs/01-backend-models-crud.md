# Spec 01: Backend — Chat Provider Models & CRUD API

## Goal

Define the `ChatProviderConfig` data model and expose CRUD endpoints at `/api/chat-providers`. Chat Providers are persisted in `.rlmkit_config.json` alongside existing config.

---

## 1. Data Models

### File: `src/rlmkit/server/models.py`

#### 1.1 New Models

Add these in a `# Chat Providers` section after the `ModeConfig` model:

```python
class ChatProviderConfig(BaseModel):
    """A named combination of LLM Provider + Execution Mode + Settings."""
    id: str
    name: str                                          # e.g. "DIRECT-CLAUDE", "RLM-OPENAI"
    llm_provider: str                                  # key from PROVIDERS_BY_KEY, e.g. "anthropic"
    llm_model: str                                     # e.g. "claude-sonnet-4-5"
    execution_mode: Literal["direct", "rlm", "rag"]
    runtime_settings: RuntimeSettings = Field(default_factory=RuntimeSettings)
    rag_config: RAGConfig | None = None                # populated only when execution_mode == "rag"
    rlm_max_steps: int = 16                            # used only when execution_mode == "rlm"
    rlm_timeout_seconds: int = 60                      # used only when execution_mode == "rlm"
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ChatProviderCreateRequest(BaseModel):
    name: str
    llm_provider: str
    llm_model: str
    execution_mode: Literal["direct", "rlm", "rag"]
    runtime_settings: RuntimeSettings | None = None
    rag_config: RAGConfig | None = None
    rlm_max_steps: int | None = None
    rlm_timeout_seconds: int | None = None


class ChatProviderUpdateRequest(BaseModel):
    name: str | None = None
    llm_model: str | None = None
    execution_mode: Literal["direct", "rlm", "rag"] | None = None
    runtime_settings: RuntimeSettings | None = None
    rag_config: RAGConfig | None = None
    rlm_max_steps: int | None = None
    rlm_timeout_seconds: int | None = None
```

#### 1.2 Modify Existing Models

**`ConfigResponse`** — add field:
```python
chat_providers: list[ChatProviderConfig] = Field(default_factory=list)
```

**`ConfigUpdateRequest`** — add field:
```python
chat_providers: list[ChatProviderConfig] | None = None
```

**`ChatRequest`** — add field (keep all existing fields for backward compat):
```python
chat_provider_id: str | None = None
```

**`ChatResponse`** — add field:
```python
chat_provider_id: str | None = None
```

**`SessionMessage`** — add fields:
```python
chat_provider_id: str | None = None
chat_provider_name: str | None = None
```

**`SessionDetail`** — add field (keep existing `messages` for backward compat):
```python
conversations: dict[str, list[SessionMessage]] = Field(default_factory=dict)
```

---

## 2. CRUD API Endpoints

### File: `src/rlmkit/server/routes/chat_providers.py` (new file)

Create a FastAPI router with prefix `/api/chat-providers` and tag `"chat-providers"`.

#### 2.1 List All

```
GET /api/chat-providers
Response: list[ChatProviderConfig]
```

Return `state.config.chat_providers`.

#### 2.2 Get One

```
GET /api/chat-providers/{chat_provider_id}
Response: ChatProviderConfig
Errors: 404 if not found
```

#### 2.3 Create

```
POST /api/chat-providers
Body: ChatProviderCreateRequest
Response: ChatProviderConfig (201 Created)
Errors: 400 if llm_provider not in PROVIDERS_BY_KEY catalog
Errors: 409 if name already exists (case-insensitive)
```

Logic:
1. Validate `req.llm_provider` exists in `PROVIDERS_BY_KEY`
2. Validate `req.name` is unique among existing chat_providers (case-insensitive)
3. Generate `id = str(uuid.uuid4())`
4. Set `created_at` and `updated_at` to `datetime.now(timezone.utc)`
5. Apply defaults: if `runtime_settings` is None, use `RuntimeSettings()`; if `rlm_max_steps` is None, use 16; etc.
6. Append to `state.config.chat_providers`
7. Call `state.save_config()`
8. Return the created `ChatProviderConfig`

#### 2.4 Update

```
PUT /api/chat-providers/{chat_provider_id}
Body: ChatProviderUpdateRequest
Response: ChatProviderConfig
Errors: 404 if not found
Errors: 409 if new name conflicts with another provider
```

Logic:
1. Find existing Chat Provider by ID
2. If `name` provided and differs, validate uniqueness
3. Apply non-None fields from request to existing config
4. Update `updated_at`
5. Call `state.save_config()`
6. Return updated config

#### 2.5 Delete

```
DELETE /api/chat-providers/{chat_provider_id}
Response: 204 No Content
Errors: 404 if not found
```

Logic:
1. Find and remove from `state.config.chat_providers`
2. Call `state.save_config()`

---

## 3. Router Registration

### File: `src/rlmkit/server/app.py`

Add import and include:
```python
from rlmkit.server.routes import chat_providers
app.include_router(chat_providers.router)
```

---

## 4. Auto-Migration

### File: `src/rlmkit/server/dependencies.py`

Add a `_migrate_chat_providers()` method to `AppState`, called at the end of `__init__` (after `_load_config()` and `_load_sessions()`).

**Behavior:**
- If `self.config.chat_providers` is already non-empty, do nothing (migration already happened)
- Otherwise, iterate `self.config.provider_configs` and for each **enabled** provider config, create a default Chat Provider:
  - `name = "DIRECT-{provider.upper()}"` (e.g., "DIRECT-ANTHROPIC")
  - `execution_mode = "direct"`
  - `llm_provider = pc.provider`
  - `llm_model = pc.model`
  - `runtime_settings` copied from the provider config
- Save config after migration

---

## 5. Config Persistence Format

The `.rlmkit_config.json` file structure (relevant parts):

```json
{
  "config": {
    "active_provider": "anthropic",
    "active_model": "claude-sonnet-4-5",
    "provider_configs": [...],
    "mode_config": {...},
    "chat_providers": [
      {
        "id": "uuid-here",
        "name": "DIRECT-ANTHROPIC",
        "llm_provider": "anthropic",
        "llm_model": "claude-sonnet-4-5",
        "execution_mode": "direct",
        "runtime_settings": {
          "temperature": 0.7,
          "top_p": 1.0,
          "max_output_tokens": 4096,
          "timeout_seconds": 30
        },
        "rag_config": null,
        "rlm_max_steps": 16,
        "rlm_timeout_seconds": 60,
        "created_at": "2026-02-28T00:00:00+00:00",
        "updated_at": "2026-02-28T00:00:00+00:00"
      }
    ]
  },
  "system_prompts": {...},
  "user_profiles": [...]
}
```

---

## 6. Frontend Types (parallel update)

### File: `frontend/src/lib/api.ts`

Add TypeScript interfaces mirroring the backend models:

```typescript
export interface ChatProviderConfig {
  id: string;
  name: string;
  llm_provider: string;
  llm_model: string;
  execution_mode: "direct" | "rlm" | "rag";
  runtime_settings: RuntimeSettings;
  rag_config?: RAGConfig | null;
  rlm_max_steps: number;
  rlm_timeout_seconds: number;
  created_at: string;
  updated_at: string;
}

export interface ChatProviderCreateRequest {
  name: string;
  llm_provider: string;
  llm_model: string;
  execution_mode: "direct" | "rlm" | "rag";
  runtime_settings?: RuntimeSettings | null;
  rag_config?: RAGConfig | null;
  rlm_max_steps?: number | null;
  rlm_timeout_seconds?: number | null;
}

export interface ChatProviderUpdateRequest {
  name?: string | null;
  llm_model?: string | null;
  execution_mode?: "direct" | "rlm" | "rag" | null;
  runtime_settings?: RuntimeSettings | null;
  rag_config?: RAGConfig | null;
  rlm_max_steps?: number | null;
  rlm_timeout_seconds?: number | null;
}
```

Update `AppConfig`:
```typescript
export interface AppConfig {
  // ... existing fields ...
  chat_providers: ChatProviderConfig[];
}
```

Update `ChatRequest`:
```typescript
export interface ChatRequest {
  // ... existing fields ...
  chat_provider_id?: string | null;
}
```

Update `ChatResponse`:
```typescript
export interface ChatResponse {
  // ... existing fields ...
  chat_provider_id?: string | null;
}
```

Update `SessionMessage`:
```typescript
export interface SessionMessage {
  // ... existing fields ...
  chat_provider_id?: string | null;
  chat_provider_name?: string | null;
}
```

Update `SessionDetail`:
```typescript
export interface SessionDetail {
  // ... existing fields ...
  conversations?: Record<string, SessionMessage[]>;
}
```

Add CRUD functions:
```typescript
export const getChatProviders = () =>
  fetchJSON<ChatProviderConfig[]>("/api/chat-providers");

export const getChatProvider = (id: string) =>
  fetchJSON<ChatProviderConfig>(`/api/chat-providers/${id}`);

export const createChatProvider = (req: ChatProviderCreateRequest) =>
  fetchJSON<ChatProviderConfig>("/api/chat-providers", {
    method: "POST",
    body: JSON.stringify(req),
  });

export const updateChatProvider = (id: string, req: ChatProviderUpdateRequest) =>
  fetchJSON<ChatProviderConfig>(`/api/chat-providers/${id}`, {
    method: "PUT",
    body: JSON.stringify(req),
  });

export const deleteChatProvider = (id: string) =>
  fetchJSON<void>(`/api/chat-providers/${id}`, { method: "DELETE" });
```

---

## Acceptance Criteria

1. `GET /api/chat-providers` returns `[]` on fresh install, then returns migrated providers after first startup with existing provider_configs
2. `POST /api/chat-providers` creates and persists a new Chat Provider; rejects duplicate names and invalid provider keys
3. `PUT /api/chat-providers/{id}` updates fields; validates name uniqueness
4. `DELETE /api/chat-providers/{id}` removes and persists
5. Server restart preserves all Chat Providers (loaded from `.rlmkit_config.json`)
6. Auto-migration creates one "DIRECT-{PROVIDER}" Chat Provider per enabled provider config on first load
7. All existing tests pass without regression (`uv run pytest`)
8. TypeScript types compile without errors (`npx tsc --noEmit`)
