# Spec 02: Backend — Chat Execution & Session Storage Refactor

## Goal

Refactor chat execution so that each request can target a specific Chat Provider, using that provider's LLM, mode, and settings. Maintain per-Chat-Provider conversation history to fix the bug where every question is treated as a new first question.

---

## 1. Session Storage Changes

### File: `src/rlmkit/server/dependencies.py`

#### 1.1 SessionRecord

The `SessionRecord` dataclass must have both a legacy flat `messages` list and a `conversations` dict keyed by `chat_provider_id`:

```python
@dataclass
class SessionRecord:
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    messages: list[dict[str, Any]] = field(default_factory=list)
    conversations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
```

#### 1.2 AppState Helper Methods

**`get_chat_provider(chat_provider_id: str) -> ChatProviderConfig | None`**
- Linear scan through `self.config.chat_providers` by `id`

**`get_conversation(session_id: str, chat_provider_id: str) -> list[dict]`**
- Return `session.conversations.get(chat_provider_id, [])`
- Return `[]` if session not found

**`add_message(session_id: str, message: dict, chat_provider_id: str | None = None) -> None`**
- Always append to `session.messages` (legacy flat list)
- If `chat_provider_id` is not None, also append to `session.conversations[chat_provider_id]` (create list if missing)

**`create_llm_adapter_for_chat_provider(chat_provider_id: str) -> LiteLLMAdapter`**
- Look up the Chat Provider by ID (raise ValueError if not found)
- Use `cp.llm_provider` and `cp.llm_model` to build the LiteLLM model name (via `_litellm_model_name()`)
- Use `cp.runtime_settings` for temperature, max_tokens, timeout
- Resolve `api_base` from the provider catalog entry if present
- Return a configured `LiteLLMAdapter`

#### 1.3 Persistence

**`_load_sessions()`** must deserialize the `conversations` field:
```python
rec = SessionRecord(
    id=s["id"],
    name=s["name"],
    created_at=datetime.fromisoformat(s["created_at"]),
    updated_at=datetime.fromisoformat(s["updated_at"]),
    messages=s.get("messages", []),
    conversations=s.get("conversations", {}),  # NEW
)
```

**`save_sessions()`** must serialize it:
```python
{
    "id": s.id,
    "name": s.name,
    "created_at": s.created_at.isoformat(),
    "updated_at": s.updated_at.isoformat(),
    "messages": s.messages,
    "conversations": s.conversations,  # NEW
}
```

---

## 2. Chat Execution Refactor

### File: `src/rlmkit/server/routes/chat.py`

#### 2.1 REST Endpoint: `POST /api/chat`

**Request flow:**

1. Validate `content` or `file_id` is provided
2. Resolve content from file if needed
3. Get or create session
4. **If `chat_provider_id` is provided:**
   - Look up Chat Provider via `state.get_chat_provider(chat_provider_id)`
   - 404 if not found
   - Override `mode` with `cp.execution_mode`
5. **If only legacy params** (`mode`, `provider`, `model`):
   - Use them as-is (backward compat)
6. Create `ExecutionRecord`
7. Build user message dict with `chat_provider_id` field
8. Call `state.add_message(session.id, user_msg, chat_provider_id)` — this writes to both flat and per-provider lists
9. Spawn background `_run_execution()`
10. Return `ChatResponse` with `execution_id`, `session_id`, `chat_provider_id`

#### 2.2 Background Execution: `_run_execution()`

**Signature:**
```python
async def _run_execution(
    state: AppState,
    execution: ExecutionRecord,
    content: str,
    query: str,
    mode: str,
    chat_provider_id: str | None = None,
) -> None:
```

**Key changes from the original:**

1. **LLM Adapter selection:**
   - If `chat_provider_id` is set: `state.create_llm_adapter_for_chat_provider(chat_provider_id)`
   - Else: `state.create_llm_adapter()` (global active provider)

2. **Conversation history (the critical fix):**
   - If `chat_provider_id` is set, get previous messages from `state.get_conversation(session_id, chat_provider_id)`
   - Exclude the current user message (it was just appended, so it's the last item)
   - Build conversation context as a list of `{role, content}` dicts
   - Prepend to the query as a formatted conversation context:
     ```
     Previous conversation:
     User: <msg1>

     Assistant: <msg2>

     User: <msg3>

     Current question: <actual query>
     ```
   - This ensures each Chat Provider has its own continuous conversation thread

3. **Run config from Chat Provider:**
   - If Chat Provider has `execution_mode == "rlm"`, use `cp.rlm_max_steps` and `cp.rlm_timeout_seconds`
   - If Chat Provider has `execution_mode == "rag"` and `cp.rag_config`, use those RAG settings (when RAG use case is implemented)

4. **Store assistant response:**
   - Include `chat_provider_id` and `chat_provider_name` (from `cp.name`) in the assistant message dict
   - Call `state.add_message(session_id, assistant_msg, chat_provider_id)` — writes to both flat and per-provider

5. **Error handling:**
   - Error messages also include `chat_provider_id` so they're stored in the right conversation

#### 2.3 WebSocket Endpoint

Same logic as REST for Chat Provider resolution:
- Extract `chat_provider_id` from the `query` message data
- Resolve mode from Chat Provider if provided
- Add user message with `chat_provider_id`
- Use Chat Provider-specific LLM adapter in `_ws_execute()`
- Include `chat_provider_id` and `chat_provider_name` in `complete` and `error` events

---

## 3. Session Retrieval

### File: `src/rlmkit/server/routes/sessions.py`

**`GET /api/sessions/{session_id}`** must return:

```python
SessionDetail(
    id=session.id,
    name=session.name,
    created_at=session.created_at,
    updated_at=session.updated_at,
    messages=messages,              # Legacy flat list (all messages)
    conversations=conversations,    # Dict keyed by chat_provider_id
)
```

Where `conversations` is built by iterating `session.conversations` and deserializing each message list using `_deserialize_message()`.

The `_deserialize_message()` helper must include `chat_provider_id` and `chat_provider_name` from the raw dict.

---

## 4. Parallel Execution (Frontend-Driven)

The backend does NOT need a batch endpoint. The frontend sends N separate `POST /api/chat` requests — one per selected Chat Provider — each with a different `chat_provider_id` but the same `session_id` and `query`. Each executes independently and returns its own `execution_id`.

This keeps the backend simple and stateless per-request. The frontend handles parallelism and result aggregation.

---

## 5. Message Schema

Every message stored in session (both user and assistant) follows this dict shape:

```python
# User message
{
    "id": "uuid",
    "role": "user",
    "content": "the query text",
    "file_id": "optional-file-id" | None,
    "mode": "direct",                       # resolved mode
    "chat_provider_id": "cp-uuid" | None,   # which Chat Provider this was sent to
    "timestamp": "2026-02-28T12:00:00+00:00"
}

# Assistant message
{
    "id": "uuid",
    "role": "assistant",
    "content": "the response text",
    "mode_used": "direct",                  # actual mode executed
    "provider": "anthropic",                # LLM provider key
    "execution_id": "exec-uuid",
    "chat_provider_id": "cp-uuid" | None,
    "chat_provider_name": "DIRECT-CLAUDE" | None,
    "metrics": {
        "input_tokens": 150,
        "output_tokens": 350,
        "total_tokens": 500,
        "cost_usd": 0.002,
        "elapsed_seconds": 2.1,
        "steps": 1
    },
    "timestamp": "2026-02-28T12:00:02+00:00"
}
```

---

## Acceptance Criteria

1. Sending `POST /api/chat` with `chat_provider_id` uses that provider's LLM and mode
2. Sending `POST /api/chat` without `chat_provider_id` works as before (backward compat)
3. Follow-up questions in the same session+chat_provider have conversation context (not treated as new)
4. `GET /api/sessions/{id}` returns both `messages` (flat) and `conversations` (per-provider dict)
5. User messages are stored in both `messages[]` and `conversations[chat_provider_id]`
6. Assistant messages include `chat_provider_id`, `chat_provider_name`, and `metrics`
7. Session persistence survives server restart (conversations dict saved/loaded correctly)
8. WebSocket path also supports `chat_provider_id` in query messages
9. Multiple concurrent `POST /api/chat` requests with different `chat_provider_id` values but same `session_id` execute in parallel without interference
10. All existing tests pass (`uv run pytest`)
