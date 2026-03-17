# Changelog

All notable changes to RLMKit are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Async API variants: `interact_async()`, `complete_async()` with `api_base` and `timeout` parameters
- Storybook 10 with 27 component stories across 7 components (Badge, StatusIndicator, ProviderBadge, MetricCard, TypingIndicator, MessageBubble, Timeline)
- Frontend test suite: 134 passing tests covering chat, settings, dashboard, traces, and accessibility
- Node.js >= 22 requirement for frontend (`.nvmrc` + `engines` field)

### Fixed
- Anthropic default model aligned across `api.py` and `RLMKitClient` (`claude-sonnet-4-5-20250514`)
- Ollama default model aligned (`llama3`)
- ESLint `globalIgnores` includes `storybook-static/` to prevent lint failures after Storybook builds

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
