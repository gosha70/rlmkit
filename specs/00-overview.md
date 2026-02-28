# Chat Providers — Feature Specification Overview

## Problem Statement

RLM Studio currently ties chat execution directly to a single active LLM Provider. The execution mode (Direct/RLM/RAG) is selected via toggles in the chat toolbar, and only one provider can be active at a time. This limits users from comparing responses across different provider+mode combinations side by side.

Additionally, each user question is treated as a new first question — there is no conversation history passed to the LLM, breaking multi-turn chat.

## Solution: Chat Provider Abstraction

A **Chat Provider** is a named configuration that bundles:

- An **LLM Provider** (e.g., Anthropic, OpenAI, Ollama)
- An **Execution Mode** (Direct, RLM, or RAG)
- **Mode-specific settings** (RLM max_steps/timeout, RAG chunk_size/top_k, etc.)
- **Runtime settings** (temperature, top_p, max_output_tokens, timeout)

Users create and manage Chat Providers in Settings, then select one or more in the Chat tab to run queries against in parallel.

## Spec Documents

Execute these specs in order. Each spec is self-contained with file paths, data models, API contracts, and acceptance criteria.

| # | Spec | What it covers |
|---|------|----------------|
| 01 | [Backend: Models & CRUD API](./01-backend-models-crud.md) | Pydantic models, CRUD endpoints, config persistence |
| 02 | [Backend: Chat Execution & Session Storage](./02-backend-chat-execution.md) | Per-provider conversation history, LLM adapter routing, chat history fix |
| 03 | [Frontend: Settings Tab](./03-frontend-settings.md) | Replace Execution tab with Chat Providers management UI |
| 04 | [Frontend: Chat Page](./04-frontend-chat-page.md) | Multi-provider selector, column layout, parallel execution, metrics display |

## Architecture Constraints

- **Stack**: FastAPI (Python 3.10+), Next.js 16 / React 19 / TypeScript, Tailwind CSS, shadcn/ui
- **State**: In-memory AppState singleton with JSON file persistence (`.rlmkit_config.json`, `.rlmkit_sessions.json`)
- **Dependency injection**: `get_state()` FastAPI dependency returns AppState
- **Data fetching**: SWR on frontend, REST + polling for chat execution
- **LLM access**: LiteLLM via `LiteLLMAdapter` class
- **Provider catalog**: `rlmkit.ui.data.providers_catalog.PROVIDERS_BY_KEY` maps provider keys to metadata

## Current State of Codebase

The previous implementation attempt added Chat Provider support to the backend models, dependencies, API types, and route files. **However, the implementation was done directly in code without proper testing or verification.** These specs define the correct behavior; use them as the source of truth. Review existing code against these specs and fix/rewrite as needed.

## Key Files Reference

### Backend
- `src/rlmkit/server/models.py` — Pydantic request/response models
- `src/rlmkit/server/dependencies.py` — AppState, SessionRecord, persistence
- `src/rlmkit/server/routes/chat.py` — Chat REST + WebSocket endpoints
- `src/rlmkit/server/routes/chat_providers.py` — Chat Provider CRUD
- `src/rlmkit/server/routes/sessions.py` — Session retrieval
- `src/rlmkit/server/app.py` — FastAPI app + router registration

### Frontend
- `frontend/src/lib/api.ts` — Typed API client
- `frontend/src/lib/use-chat.ts` — WebSocket chat hook
- `frontend/src/app/page.tsx` — Chat page
- `frontend/src/app/settings/page.tsx` — Settings page
- `frontend/src/components/chat/` — Chat components directory

## Post-Implementation

After all four specs are implemented and verified:
- Review the **Traces** page (`frontend/src/app/traces/page.tsx`, `src/rlmkit/server/routes/traces.py`) for compatibility with chat_provider_id tracking
- Review the **Dashboard** page (`frontend/src/app/dashboard/page.tsx`, `src/rlmkit/server/routes/metrics.py`) for per-Chat-Provider metrics aggregation
