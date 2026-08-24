---
feature_id: interop-official-rlm
spec_mode: full
spec: ./spec.md
status: draft
date: 2026-08-15
origin:
  urls:
    - https://github.com/alexzhang13/rlm
  transcripts:
    - "Owner, 2026-08-15: accepted 'official RLM interop mode in the Compare matrix' as a pre-launch differentiator."
  origin_claim: |
    Inherited from spec.md — run the paper authors' `rlms` implementation as a
    selectable engine so Studio can benchmark it against its own loop, Direct
    and RAG with identical traces/telemetry.
---

# Plan — Official-engine interop

Depends on `specs/rebrand-rlm-studio` (paths below use the post-rename package `rlmstudio`; if built earlier, substitute `rlmkit`). Size: **M** (≈6–8 working days). Circuit breaker: if not mergeable by end of week 3 of the cycle, ship without it — it becomes the v1.1 headline.

## 1. Architecture
```
server/routes (chat, compare, engines) ──> application/use_cases/run_rlm_official.py ──> application/ports/rlm_engine_port.py (RLMEnginePort)
                                                       │                                              ▲
                                                       └── budgets, classification, RunResultDTO      │ implements (structural)
                                                                                                      │
                                                    infrastructure/engines/rlms_adapter.py ── lazy `import rlm` (extra: interop)
```
Reuse: `RunConfigDTO`/`RunResultDTO` (`application/dto.py:156`), mode constants (`application/sandbox_vars.py:57-61`), raw trace shape (`sandbox_vars.py:101-111`) + normaliser (`server/routes/_helpers.py:50 _canonical_action_type`, `_save_trajectory`), `TraceStep` (`domain/entities.py:52`), budget + outcome classification helpers used by `RunRLMUseCase` (`application/use_cases/run_rlm.py`), matrix dispatch (`run_matrix_comparison.py:248-291`, `_SUPPORTED_MODES:48`, `SlotMode:32`), slot construction from the slot's LLM Provider in `server/routes/compare_matrix.py:~400-450` (build the engine there, next to where the LiteLLM adapter is built), ranking `_rank:294` and telemetry write at `compare_matrix.py:~805` (unchanged), cost table in `infrastructure/llm/`. Budget mapping: `max_steps` → rlms `max_iterations`, `max_recursion_depth` → rlms `max_depth`, wall-clock guard ours, cost post-hoc from tokens.

## 2. Allowlist
`src/rlmstudio/application/{constants.py|dto.py,ports/rlm_engine_port.py,use_cases/run_rlm_official.py,use_cases/run_matrix_comparison.py}`, `src/rlmstudio/infrastructure/engines/**`, `src/rlmstudio/server/{dependencies.py,models.py,routes/chat.py,routes/compare.py,routes/engines.py,app.py}`, `src/rlmstudio/telemetry/**` (mode enum only), `pyproject.toml` (`interop` extra), `frontend/src/{lib/api.ts,lib/types*,app/compare/**,components/**(compare/chat/traces)}`, `tests/**`, `docs/{rlm-studio-guide.md,rlm-concepts.md,hosts/README.md}`, `README.md`, `CHANGELOG.md`.

## 3. Steps
1. **Constant + literals** (FR-1): add `MODE_RLM_OFFICIAL` to the mode constants module (create `application/constants.py` if none — check where `SlotMode`/mode literals live first and centralise); widen `SlotMode`, `server/models.py` literals; frontend type mirror. Tests: literal round-trip 422→200.
2. **Port** (FR-2): `RLMEnginePort` + `tests/test_port_compliance.py` structural test with a fake.
3. **Fake engine + use case** (FR-4): `RunRLMOfficialUseCase` with budgets/classification; tests with `FakeRLMEngine` returning a scripted trajectory (happy path, error, timeout, budget breach).
4. **Adapter** (FR-3): `rlms_adapter.py` — client factory from Studio provider config (openai/anthropic native; `openai_compatible` base_url for vLLM/LM Studio/Ollama; document what is unsupported), trajectory → `TraceStep` mapping, usage → tokens/cost, `is_available()` (import + version). One `--runslow` test on 3.11 against an OpenAI-compatible stub server (reuse existing test stubs if present in `tests/`).
5. **Matrix + routes** (FR-5/6): dispatch, validation, `GET /api/engines`, 400 path; wiring in `dependencies.py`; e2e tests in `tests/e2e/test_api_endpoints.py`.
6. **Extra** (FR-4): `interop = ["rlms>=0.1.3"]`; `all` includes it; `uv lock`; CI matrix already runs 3.10–3.12 → 3.10 exercises the unavailable path, 3.11+ the available path (install extra only there).
7. **Frontend** (FR-7): option + availability gating + badge; vitest for the picker and gating.
8. **Telemetry + docs** (FR-8/9); CHANGELOG.

## 4. Test strategy
- Unit: use case with fake engine (4 scenarios), adapter mapping (pure function tests over a recorded `rlms` trajectory fixture), availability on/off via monkeypatched import.
- API: `/api/engines`, compare with `rlm_official` slot (fake engine injected via `AppState`), 400 when unavailable.
- Slow: one real `rlms` run (3.11) against a stub OpenAI-compatible server; asserts step types and that Studio's wall-clock timeout fires (AC-3).
- Frontend: vitest — option disabled when unavailable; badge renders.

## 5. Risks
| Risk | Mitigation |
|---|---|
| `rlms` API churn (0.1.x) | pin `>=0.1.3,<0.2`; adapter isolates the import; availability reports version |
| `rlms` hard deps (`openai>=2.14`, `anthropic>=0.75`, `google-genai`, `portkey-ai`, `pytest>=9.0.2` at runtime) conflict with the LiteLLM stack or bloat `[all]` | Resolve `[studio,interop]` in a fresh 3.11 venv in step 6 before committing; if unresolvable, keep `interop` **out of `all`** and document a separate venv; CI job on 3.11 installs `[studio,interop]` |
| `rlms` `local` environment executes in-process, unsandboxed | default banner + docs; map Studio `docker` sandbox → rlms `environment="docker"`; never advertise the local env as isolated |
| Provider parity is only honest with the same model + base_url per engine | the slot builder passes identical model/base_url/api_key to every engine; note in `BENCHMARKS.md` methodology |
| Engine gives no token usage for some clients | show `—`, rank last (OQ-2); note in docs |
| Local rlms env is not sandboxed | banner + docs; Docker mapping when Studio sandbox = docker |
| Time overrun | circuit breaker end of week 3; Compare-only fallback (OQ-3) before dropping the bet |
