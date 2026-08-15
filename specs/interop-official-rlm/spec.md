---
feature_id: interop-official-rlm
spec_mode: full
status: draft
date: 2026-08-15
origin:
  urls:
    - https://github.com/alexzhang13/rlm            # official implementation, PyPI `rlms`, Python >= 3.11
    - https://arxiv.org/abs/2512.24601              # Zhang, Kraska, Khattab — Recursive Language Models
  transcripts:
    - "Owner, 2026-08-15 planning session: appetite '~4–6 weeks: hygiene + 1–2 differentiators' — accepted the proposed differentiator 'an official RLM / dspy.RLM interop mode in the Compare matrix' with primary audiences 'AI engineers evaluating RLM vs RAG vs Direct' and 'local-LLM self-hosters'."
  origin_claim: |
    Before the first public release, RLM Studio needs one capability that no
    other RLM tool offers and that makes its "measure whether RLM helps on your
    data" positioning defensible: run the paper authors' own implementation
    (`rlms`) as a selectable engine inside the Compare matrix and Chat, so a
    user can benchmark Official-RLM vs Studio-RLM vs Direct vs RAG on the same
    document, provider and budget, with the same traces and telemetry.
spec_mode_justification: >
  New use case, new port, new adapter, new optional extra with a Python-version
  marker, API/DTO literal widening, frontend option, and telemetry mapping —
  cross-layer feature with a public API surface. Full spec.
---

# Spec — Official-engine interop (`rlm_official` slot)

## 1. Problem
RLM Studio's Compare matrix ranks `direct` / `rag` / `rlm` across providers, but the `rlm` slot is Studio's *own* loop. Skeptical engineers (and the paper authors) will ask whether Studio's numbers reflect the RLM idea or Studio's implementation. Adding the reference implementation as a first-class engine answers that in-product and turns Studio into the neutral bench.

## 2. Goals / non-goals
**Goals**
- G1 New slot mode `rlm_official` selectable in Compare (matrix + unified requests) and in Chat provider config, executing the `rlms` library (`from rlm import RLM`) against the slot's provider/model.
- G2 Same output contract as `RunRLMUseCase` (`RunResultDTO` with `TraceStep`s, tokens, cost, timing) so Traces, replay, Dashboard, telemetry store, ranking, judge, and JSONL export work unchanged.
- G3 Studio budgets (wall-clock, max steps, token/cost caps) enforced around the engine even where `rlms` only exposes iteration limits.
- G4 Optional extra `interop` (`rlms; python_version >= "3.11"`); a clear, classified "engine unavailable" outcome on 3.10 or when not installed; the frontend hides/disables the option when the backend reports it unavailable.
- G5 Provider mapping: `rlms` speaks OpenAI/Anthropic/OpenRouter/Portkey/vLLM natively; Studio maps its provider config to those clients where possible and otherwise routes through an OpenAI-compatible endpoint (LM Studio, vLLM, Ollama's OpenAI endpoint). Documented matrix of what works.

**Non-goals**
- Not wrapping `dspy.RLM` in this cycle (Deno runtime dependency) — v1.1 candidate; the port is designed so it can be a second adapter.
- Not exposing `rlms`' cloud sandboxes (Modal/E2B/Daytona/Prime); the engine runs with `rlms`' local or Docker environment only, chosen by Studio's sandbox setting where a mapping exists (`docker` → rlms Docker env, everything else → rlms local env with a documented warning that the official local env is not isolated).
- No changes to Studio's own RLM loop.

## 3. Functional requirements
| ID | Requirement |
|---|---|
| FR-1 | Constant `MODE_RLM_OFFICIAL = "rlm_official"` added next to `MODE_DIRECT/MODE_RLM/MODE_RAG/…` in `application/sandbox_vars.py:57-61` (the existing mode-constants home) and imported by DTOs, use case, server models, telemetry, frontend type mirror. `SlotMode` (`run_matrix_comparison.py:32`) and `_SUPPORTED_MODES` (`:48`) widened; `server/models.py` `CompareMatrix*Request` mode/`slots[].mode` literals (`:219, :638, :682, :711`) and `ChatRequest.mode` widened. `auto` never selects `rlm_official`. |
| FR-2 | Port `RLMEnginePort` (`application/ports/rlm_engine_port.py`, `@runtime_checkable`): `run(content, query, config) -> RunResultDTO`, `run_async(...)`, `is_available() -> tuple[bool, str]`. Structural checks only. |
| FR-3 | Adapter `infrastructure/engines/rlms_adapter.py`: lazy-imports `rlm`; builds the client from Studio's provider settings (FR G5); runs `RLM(...).completion(...)`; captures the trajectory via `RLMLogger` / `RLMChatCompletion.metadata` (iterations → `code_blocks[code, result.stdout/stderr, rlm_calls]`, `usage_summary`, `execution_time`). Emits Studio's **raw trace shape** (flat dicts with `role ∈ {assistant, execution, system}` + content/code/tokens/model/elapsed, see `application/sandbox_vars.py:101-111`) — one `assistant` (code) + one `execution` (result) per block, nested `rlm_query` calls as extra `execution` entries with `recursion_depth` — so the existing normaliser `server/routes/_helpers.py:_canonical_action_type` (assistant→inspect, execution→subcall, last-success→final) and `_save_trajectory` produce `TraceStep`s unchanged. Token/cost aggregation from `usage_summary` via the same cost table used by `LiteLLMAdapter`; `ttft_ms`/decode = `None` (no streaming from the engine) and a `known_limitation` note. |
| FR-4 | Use case `application/use_cases/run_rlm_official.py`: `RunRLMOfficialUseCase(engine: RLMEnginePort)` — enforces `RunConfigDTO` budgets (max_steps → engine iteration limit; wall-clock via timeout wrapper; token/cost caps checked post-run and classified `budget_exceeded` if breached), attaches `mode="rlm_official"` and engine version to the result. |
| FR-5 | `RunMatrixComparisonUseCase._execute_slot` dispatches `rlm_official`; slot validation requires an engine; `_copy_config_for_slot` handles the new mode; ranking treats it like `rlm`. |
| FR-6 | Server: dependency wiring in `server/dependencies.py` (`get_rlm_engine()`), `GET /api/engines` returns availability `{rlm_official: {available, reason, version}}`; chat and compare routes accept the mode; 400 with the reason if unavailable. |
| FR-7 | Frontend: engine option in Compare slot picker and Chat provider mode select; badge "official rlms x.y.z" in Traces/Compare; disabled with tooltip when `/api/engines` says unavailable. Type mirror for the new literal. |
| FR-8 | Telemetry: `mode` column stores `rlm_official`; Dashboard groups by it; JSONL export unchanged. |
| FR-9 | Docs: `docs/rlm-studio-guide.md` (Compare section), `docs/rlm-concepts.md` ("what RLM Studio adds beyond the paper" gets a paragraph on running the reference implementation), README comparison table row, `docs/hosts/README.md` provider-mapping matrix. CHANGELOG entry. |

## 4. Constraints
- Dependency rule: `domain` untouched except if a new value object is strictly required; `application` imports nothing from `infrastructure`/`server`. `RunResultDTO` (`application/dto.py:156`) is already the contract — no new domain port; the built-in `RunRLMUseCase` may be wrapped as an engine so all engines are peers behind `RLMEnginePort` (optional refactor, only if it simplifies dispatch).
- **Dependency reality of `rlms` (verified on PyPI 2026-08-15):** `requires_python >= 3.11`; hard deps `anthropic>=0.75.0`, `google-genai>=1.56.0`, `openai>=2.14.0`, `portkey-ai>=2.1.0`, `pytest>=9.0.2` (a test framework as a runtime dep), `python-dotenv`, `requests`, `rich`. The `[interop]` extra must be resolvable together with `[studio]` (LiteLLM's own `openai`/`anthropic` pins) — verify in a fresh 3.11 venv before committing the extra; if unresolvable, ship interop as a separately documented `pip install rlm-studio[interop]` in its own venv and say so.
- No inline prompts (the engine's own prompts are the engine's; Studio passes none).
- Python 3.10 remains supported for everything except this extra.
- Tests use a fake engine; exactly one `--runslow` test exercises real `rlms` on 3.11+ with a local OpenAI-compatible stub or a recorded fixture — no live API calls.

## 5. Acceptance criteria
- AC-1 On 3.11 with `[interop]`, a Compare run `{openai/gpt-4o-mini, ollama/qwen} × {direct, rlm, rlm_official}` completes; each `rlm_official` slot has ≥1 `inspect`, ≥0 `subcall`, exactly 1 `final` step; ranking by cost/latency/TTFT/judge includes the slots.
- AC-2 On 3.10 or without the extra: `/api/engines` reports unavailable with a reason; Compare/Chat with the mode return 400 + reason; frontend disables the option.
- AC-3 A `while True: pass` inside the engine's REPL is stopped by Studio's wall-clock budget and classified `timeout`.
- AC-4 Trace/replay/Dashboard/telemetry pages render `rlm_official` runs without code changes beyond FR-7/FR-8.
- AC-5 `git grep '"rlm_official"'` finds only the constants module(s) and tests.

## 6. Open questions (defaults)
- OQ-1 Sandbox mapping default: rlms *local* env with a warning banner; Docker mapping if Studio sandbox = docker. Default as stated.
- OQ-2 Cost when engine reports no usage: show `—` and exclude the slot from cost ranking (rank last), not zero. Default as stated.
- OQ-3 Chat mode exposure: default yes (behind the same availability flag); could be Compare-only if time is short — Compare is the must.
