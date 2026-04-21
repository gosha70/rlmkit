# Changelog

All notable changes to RLMKit are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

Learn tab V2 — a self-contained surface that teaches the RLM loop
through a scrubbable, step-by-step replay. Shipped in three slices
(V1 → V2a → V2b) across PRs #20, #23, and #24.

### Added
- **Learn tab (V1, PR #20)**: new `/learn` surface with Concepts,
  Cookbook, and Troubleshooting sub-pages. Mode chooser, diagnostics
  strip, and markdown-backed guides for `rlm-concepts`, `rlm-prompt-tuning`,
  `rlm-studio-guide`, and `lessons-from-ai-copilots`.
- **Replay walkthrough widget (V2a, PR #23)**: 6-node SVG diagram,
  play / pause / step / reset controls at 1× / 1.5× / 2× speeds,
  three-pane layout (controls / diagram / step list + detail). Mounts
  on the Concepts page with a bundled demo replay. Advanced details
  tray is folded into the right pane and resets to collapsed on
  step change.
- **Trace-backed replay (V2b, PR #24)**: deep-link from any execution
  to its own walkthrough.
  - `GET /api/replays/{execution_id}` — pure-service converter
    (`src/rlmkit/application/services/trace_to_replay.py`) that turns
    a canonical `TraceResponse` into a `LearnReplay`. Returns a bare
    `LearnReplay` (no wrapper); 404 with the standard error envelope
    when the id is unknown. Omits null optional fields on the wire to
    match bundled replay JSON.
  - `/learn/replay/[executionId]` — SWR-based page that mirrors
    Concepts' loading / error / ready pattern; reuses the shared
    `ReplayWalkthrough` widget exactly (no second renderer).
    Surfaces a truncation banner when `metadata.truncated` fires.
  - Traces row CTA — small outline "Replay in Learn" button per row
    navigates to the new page; keyboard-safe (Enter/Space on the
    button no longer bubbles into the row's open-trace handler).
  - Shared canonical trace loader — fixes the live-execution path
    so both `/api/traces/{id}` and `/api/replays/{id}` see the same
    `inspect | subcall | final | error` action-type enum.
  - Contract refinement: code-less `inspect` steps (the RLM
    controller's "Runtime fingerprint" preamble) are dropped from
    trace-sourced replays rather than rendered as empty code cards.

### Tests
- +5 converter unit tests for the code-less-inspect refinement
  (real-trace fixture + whitespace-only / null / post-skip-subcall
  edges).
- +5 e2e tests for the replay route (unknown id → 404, telemetry
  round-trip, in-memory canonicalization regression guard, failed
  run with `metadata.failed` + folded error, null-omission wire
  shape, bare `LearnReplay` response).
- +34 unit tests for the `trace_to_replay` service across bookends,
  kind inference, metadata, failure handling, truncation, step
  bounds, ids/titles, and final-step folding.
- +8 frontend tests (5 for the page, 3 for the Traces CTA) covering
  loading / error / ready / truncated states and the keyboard /
  mouse stopPropagation contract.

### Fixed
- Failed-run replays now preserve the trace-side failing output in
  `answer.details.output` instead of duplicating the run-level
  error string from `summary`.
- Truncation banner copy now matches the metadata it cites
  ("The full replay would have had N steps" — `originalStepCount`
  is the pre-truncation replay length, not the raw trace step count).

## [1.0.0] - 2026-03-28

Cycle 3 (Quality, DevOps & Release) + Cool-down 3.

### Changed
- **BREAKING:** `ExecutionConfig.default_safe_mode` now defaults to `True` — sandbox restrictions (blocked modules, restricted builtins via RestrictedPython) are active by default. Pass `safe_mode=False` explicitly to opt into unrestricted execution.
- **BREAKING:** `RecursiveController.max_depth` now defaults to `1` (aligned with the RLM paper's evaluated regime). Deeper recursion remains fully supported — pass `max_depth=N` explicitly. Previous default was `5`.

### Added
- CI pipeline: Codecov coverage upload, pip-audit vulnerability scan, Docker sandbox build + smoke-test job
- Coverage gate raised to 80%; omit list scoped to Streamlit-bound files only — pure-Python `ui/data/` and `ui/services/` remain measured
- Pre-commit mypy hook (`uv run mypy src/rlmkit/ --ignore-missing-imports`); ruff scope expanded to all of `src/rlmkit/`
- `execute_async()` on all three sandbox adapters (`LocalSandboxAdapter`, `RestrictedSandboxAdapter`, `DockerSandboxAdapter`) — satisfies `SandboxPort` protocol
- `complete_async()` and `complete_stream_async()` on all legacy LLM adapters (`MockLLMAdapter`, `OpenAIAdapter`, `AnthropicAdapter`, `OllamaAdapter`) — satisfies `LLMPort` protocol; streaming wrappers delegate to `complete_async` via `asyncio.to_thread` to remain non-blocking
- Async API variants: `interact_async()`, `complete_async()` with `api_base` and `timeout` parameters
- Storybook 10 with 27 component stories across 7 components
- Frontend test suite: 175 passing + 21 skipped tests (vitest, Node ≥ 22)
- D1: Unified `interact()` / `interact_async()` API consolidation
- D2: Dashboard charts polished (chart types, color palette, tooltips)
- D3: Frontend cleanup — dead chat components removed, dead API functions removed, error toasts added (`sonner`), Dashboard→Traces deep-link, Traces "Load more" pagination

### Fixed
- Security (CodeQL `py/clear-text-storage-sensitive-data`): replaced bespoke `_update_env_file()` with `SecretStore` dispatch — OS keyring when available, JSON file (`~/.rlmkit/api_keys.json`, chmod 600) as fallback; startup `_reload_stored_api_keys()` restores persisted keys with correct precedence (real env > SecretStore > legacy `.env`)
- `_create_llm_adapter()` return type narrowed from `object` to `LLMPort`; branch-local variable names fixed to prevent mypy incompatible-assignment errors
- `ExecutionSlot.mode` cast to `Literal["rlm","direct","rag"]` in configuration panel
- Anthropic default model updated to `claude-sonnet-4-6` across `api.py` and `RLMKitClient`
- Local providers (`ollama`, `lmstudio`) now require an explicit `model=` argument; no silent default (avoids "model not found" errors on deployments with different models pulled)
- ESLint `globalIgnores` includes `storybook-static/`

### Known Limitations
- **No timeout enforcement in non-main threads (e.g., Streamlit):** When the REPL environment runs outside the main thread, signal-based timeouts are unavailable and process-based timeouts would break variable persistence. Code execution proceeds without a timeout guard in these contexts. A warning is now logged when this path is taken. Out-of-process sandbox execution is planned for v1.1.0.

## [0.2.0-alpha.3] - 2026-02-11

### Added
- WCAG 2.1 AA accessibility: 32 issues fixed across 20 files (ARIA labels, keyboard navigation, focus management, color contrast, reduced motion support)
- End-to-end WebSocket streaming with `ExecutionEventEmitter` protocol
- Token-by-token response streaming in chat UI
- Live step progress during RLM execution
- `execute_async()` on all use cases with event emitter support

## [0.2.0-alpha.2] - 2026-02-10

### Added
- Next.js 16 frontend with 4 pages: Chat, Dashboard, Traces, Settings
- 38 custom React components built on shadcn/ui and Radix UI
- Dark/light theme with system preference detection via next-themes
- SWR-based data fetching with real-time dashboard refresh
- WebSocket reconnection with exponential backoff

### Fixed
- 8 of 9 server code review findings resolved (error format, validation, race conditions)

## [0.2.0-alpha.1] - 2026-02-10

### Added
- FastAPI backend with 12 REST + 1 WebSocket endpoint
- 30+ Pydantic v2 request/response models
- AppState singleton with dependency injection
- Docker Sandbox Adapter implementing `SandboxPort`
- Async port methods: `complete_async()`, `complete_stream_async()`, `execute_async()`
- Two-model switching in `RunRLMUseCase` (root + recursive model)

## [0.1.0] - 2026-02-09

Initial release after Cycle 1: Foundation & Core Alignment.

### Added
- Clean Architecture: domain, application, infrastructure, and public API layers
- `interact()` and `complete()` as main programmatic entry points
- Three execution modes: Direct, RAG, RLM with Auto selection
- LiteLLM integration for 100+ LLM providers through a single adapter
- Two-model cost optimization (root model + recursive model)
- Deep recursion support with configurable `max_depth`
- RestrictedPython sandbox for safe code execution
- Budget controls: token limits, cost caps, step limits, timeouts
- Execution tracing with per-step metrics
- `RLMKitClient` public client with backward compatibility
- 814+ test functions across unit, integration, and E2E suites
- CI/CD pipeline with GitHub Actions (lint, typecheck, test, security)
- Pre-commit hooks (ruff, trailing whitespace, YAML/JSON validation)
