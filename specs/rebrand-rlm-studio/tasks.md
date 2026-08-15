---
feature_id: rebrand-rlm-studio
spec: ./spec.md
plan: ./plan.md
status: draft
date: 2026-08-15
---

# Tasks — Rebrand RLMKit → RLM Studio + packaging hygiene

Execute in order on `feat/rebrand-rlm-studio`; one commit per phase; CI green before moving on. Allowlist in `plan.md` §2 is binding. Open questions OQ-1..3 in `spec.md` §6 have defaults — apply them and note it.

## Phase 0 — confirmations (no edits)
- [ ] T0.1 `git grep -iw rlmkit | wc -l` and per-group counts match `plan.md` §1 (±10%); note drift.
- [ ] T0.2 Confirm `rlm-studio` still free on PyPI (`curl -s -o /dev/null -w "%{http_code}" https://pypi.org/pypi/rlm-studio/json` → 404) and GitHub.
- [ ] T0.3 Confirm master CI status and the `cryptography` pip-audit finding.

## Phase 0b — prep refactor (old name still in place; CI green)
- [ ] T0.4 Fix `tests/e2e/test_docs.py:85` to assert on a repo marker file, not the checkout dir name.
- [ ] T0.5 Centralise the ~12 `Path.home() / ".rlmkit"` sites onto one `STATE_DIR` accessor (list in `plan.md` §3 step 0); tests still green.

## Phase 1 — constants + env accessor (`src/rlmkit/branding.py`, callers, tests)
- [ ] T1.1 Create `branding.py` with the constants and `env()` accessor (FR-3).
- [ ] T1.2 Replace all `RLMKIT_*` reads with `branding.env(...)`; leave `.env.example`/docs for Phase 7.
- [ ] T1.3 `tests/test_branding_env.py` — canonical wins; legacy fallback warns once; default path.

## Phase 2 — state dir migration (`server/dependencies.py`, `config.py`, `storage/database.py`, `infrastructure/storage/sqlite_adapter.py`, `server/routes/providers.py`, tests)
- [ ] T2.1 `ensure_state_dir()` implementing FR-4; single source for the path.
- [ ] T2.2 `tests/test_state_dir_migration.py` — fresh HOME; legacy-only HOME (copies, logs once, originals untouched); both present (no-op); `RLM_STUDIO_DIR` override.

## Phase 3 — package rename
- [ ] T3.1 `git mv src/rlmkit src/rlmstudio`; sed imports in `src tests examples scripts benchmarks docs`.
- [ ] T3.2 `pyproject.toml` FR-2 (name, scripts, package-data, tool paths, urls, classifier, keywords).
- [ ] T3.3 `.github/workflows/ci.yml` paths (`ruff`, `mypy`, `bandit`, `--cov=rlmstudio`, import smoke).
- [ ] T3.4 Logger names (5), `prompts/templates.py` package strings + `prompts/README.md`, `__init__.py` `_pkg_version("rlm-studio")`, `cli/main.py` help/usage + `"rlmstudio.server.app:app"`, `ui_bundle.py` docstrings, `.pre-commit-config.yaml` paths, the 181 `patch("rlmkit.…")` test strings.
- [ ] T3.5 Telemetry JSONL extras keys → `rlm_studio_*` + tests (FR-16); keyring service `"rlm-studio"` with read-fallback + re-save (FR-17); default config filename + legacy search list (FR-18); `DEFAULT_SANDBOX_IMAGE` constant used by `envs/sandbox.py` and `docker_sandbox_adapter.py` (FR-19).
- [ ] T3.6 Full suite + `ruff format`.

## Phase 4 — extras + Streamlit removal + audit fix
- [ ] T4.1 Delete `ui/app.py`, `ui/pages/`, `ui/components/`, `ui/session.py`, `ui/charts.py` + their tests; drop coverage omit entries; **keep** `ui/services/`, `ui/data/`; tag `archive/streamlit-ui` on the parent commit (owner pushes tag later).
- [ ] T4.2 Extras `studio`, `eval`, `interop` (empty), `dev`; `all` = `studio,eval,interop`; remove streamlit/plotly/gitpython/pillow pins; `cryptography>=50.0.0`; `uv lock`.
- [ ] T4.3 Update dependency-pin-guard tests; `uv run pip-audit` green; fold dependabot #60/#61 versions.

## Phase 5 — Docker (`docker-compose.yml`, `docker/`, `.env.example`, CI docker job)
- [ ] T5.1 Rename images/containers/volume/project/home path; Dockerfile.api user `rlmstudio` + venv `/opt/rlm-studio-venv`; `docker/README.md`; CI `docker-sandbox` image name; OPERATIONS.md note on manual copy of the old `rlmkit-data` volume.
- [ ] T5.2 Local `docker compose up --build`; Direct chat completes (AC-5).

## Phase 6 — frontend (`frontend/src/**` brand strings, `package.json`, `next.config.ts`)
- [ ] T6.1 "RLMKit" → "RLM Studio" where still present (`sidebar.tsx:126,132`, `settings/page.tsx:1498`, `lib/api.ts:2,15`, comments); `NEXT_PUBLIC_RLM_STUDIO_PERF_UI` with legacy fallback (`traces`, `compare` pages); one-shot `localStorage` key migration `rlmkit_*` → `rlm_studio_*` in `app/page.tsx`; refresh `frontend/public/logo.png`.
- [ ] T6.2 `npm test`, `npm run build:bundle`; copy `out/` to `src/rlmstudio/_ui/` locally and check `/studio` renders (do not commit `_ui/` payload).

## Phase 7 — docs + logo + changelog
- [ ] T7.1 README install block, brand, logo path; `RELEASING.md` paths; `docs/OPERATIONS.md` env/state/migration; `SECURITY.md`; `CONTRIBUTING.md`; `AGENTS.md`; `CLAUDE.md`; `.github/ISSUE_TEMPLATE/*`; `docs/hosts/*` env names.
- [ ] T7.2 `git mv docs/RLMKit_Logo.png docs/RLM_Studio_Logo.png`; `git mv docs/RLMKit_Design_Document.md docs/RLM_Studio_Design_Document.md` (+ links); `benchmarks/sample_benchmark.yaml` `created_by`; `github.com/gosha70/rlmkit` URLs → `rlm-studio` (3 files) and `.github/ISSUE_TEMPLATE/*` path refs.
- [ ] T7.3 `CHANGELOG.md` `[Unreleased]`: **BREAKING** rename entry + Migration callout (env vars, state dir, extras, removed Streamlit).

## Phase 8 — hygiene
- [ ] T8.1 `git rm debug_parsing.py` (or `git mv` to `scripts/dev/`); `.gitignore` += `build/`, `htmlcov/`, `frontend/out/`.

## Phase 9 — CI wheel-smoke + release workflow
- [ ] T9.1 `ci.yml` job `wheel-smoke` (FR-10), matrix 3.10/3.11/3.12, Node 22.
- [ ] T9.2 `.github/workflows/release.yml` (FR-11): `v*` → PyPI, `v*-rc.*` → TestPyPI, `id-token: write`, no secrets.
- [ ] T9.3 AC-6 grep gate as a CI lint step.

## Phase 10 — owner actions (manual, after merge)
- [ ] T10.1 Configure trusted publisher on TestPyPI + PyPI (playbook `steps/04`), then push `v1.0.0-rc.1` and verify AC-7.
- [ ] T10.2 Rename GitHub repo → `rlm-studio`; update remotes; description/topics.
- [ ] T10.3 Reply on #34/#37 with roadmap position; close #38/#39/#41.
