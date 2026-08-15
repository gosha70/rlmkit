---
feature_id: interop-official-rlm
spec: ./spec.md
plan: ./plan.md
status: draft
date: 2026-08-15
---

# Tasks — Official-engine interop

Branch `feat/interop-official-rlm` off `master` after `feat/rebrand-rlm-studio` merges. One commit per phase; CI green each time. Allowlist in `plan.md` §2 is binding. Apply defaults for OQ-1..3 in `spec.md` §6.

## Phase 0 — confirmations (no edits)
- [ ] T0.1 `pip index versions rlms` — record latest 0.1.x and its Python floor; read `rlm.RLM` signature and logger/trajectory API for the pinned version.
- [ ] T0.2 Locate where mode literals/constants live today (`grep -rn 'Literal\["auto"' src`; `SlotMode`); decide the single constants module.

## Phase 1 — constants + literals
- [ ] T1.1 `MODE_RLM_OFFICIAL` constant; widen `SlotMode`, `server/models.py` literals; frontend type mirror.
- [ ] T1.2 Tests: request validation accepts the mode; `auto` never routes to it.

## Phase 2 — port + fake + use case
- [ ] T2.1 `application/ports/rlm_engine_port.py` (`RLMEnginePort`).
- [ ] T2.2 `tests/fakes/fake_rlm_engine.py` (scripted trajectories).
- [ ] T2.3 `application/use_cases/run_rlm_official.py`; tests: happy, error, timeout, budget breach → correct outcome categories.

## Phase 3 — adapter
- [ ] T3.1 `infrastructure/engines/rlms_adapter.py`: lazy import, `is_available()`, client factory (native openai/anthropic; openai-compatible base_url; unsupported → clear error), run, trajectory→`TraceStep` mapping, usage→tokens/cost.
- [ ] T3.2 Recorded trajectory fixture + pure mapping tests.
- [ ] T3.3 `--runslow` test on 3.11 against an OpenAI-compatible stub; AC-3 timeout test.

## Phase 4 — matrix + routes + wiring
- [ ] T4.1 `_execute_slot` dispatch + slot validation + `_copy_config_for_slot`.
- [ ] T4.2 `server/routes/engines.py` (`GET /api/engines`), register in `app.py`; chat/compare 400 path; `get_rlm_engine()` in `dependencies.py`.
- [ ] T4.3 e2e tests in `tests/e2e/test_api_endpoints.py`.

## Phase 5 — packaging
- [ ] T5.1 `interop` extra with version marker; `all` includes it; `uv lock`; CI installs the extra on 3.11+ only.

## Phase 6 — frontend
- [ ] T6.1 Compare slot picker + Chat mode option; availability gating from `/api/engines`; engine badge in Traces/Compare.
- [ ] T6.2 vitest coverage for gating + badge.

## Phase 7 — telemetry + docs
- [ ] T7.1 Telemetry mode enum/labels; Dashboard grouping check.
- [ ] T7.2 Docs: studio guide, concepts, hosts provider-mapping matrix, README table row; CHANGELOG.

## Phase 8 — acceptance
- [ ] T8.1 Run AC-1..AC-5 from `spec.md`; record results in `doc_internal/v1.0.0-rlm-studio/MANUAL_TEST_PLAN.md`.
