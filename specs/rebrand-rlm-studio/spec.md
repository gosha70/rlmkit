---
feature_id: rebrand-rlm-studio
spec_mode: full
status: draft
date: 2026-08-15
origin:
  urls:
    - https://pypi.org/project/rlmkit/            # taken 2026-04-24 by an unrelated project
    - https://github.com/shyamsn97/rlmkit
  transcripts:
    - "Owner, 2026-08-15 planning session: 'Review the project and the delivery/publishing plan; re-evaluate usefulness; refresh the plan to target the right wider audience.' Owner then chose: 'Rebrand to RLM Studio now (PyPI rlm-studio, import rlmstudio, repo gosha70/rlm-studio)'; audiences 'local-LLM self-hosters' + 'AI engineers evaluating RLM vs RAG vs Direct'; appetite '4–6 weeks: hygiene + 1–2 differentiators'."
  origin_claim: |
    The PyPI distribution name `rlmkit` was claimed on 2026-04-24 by an
    unrelated, actively maintained RLM library, so the README's
    `pip install "rlmkit[all]"` installs someone else's package and both
    projects would install a top-level `rlmkit` import package. The owner
    decided to rebrand the project to "RLM Studio" before its first public
    release, positioning it as the evaluation workbench (the differentiator)
    rather than as one more RLM library. The same release must fix packaging
    hygiene that blocks a credible `pip install` (dev tools and a legacy
    Streamlit app inside `[all]`, a failing pip-audit, stray tracked files)
    and add the PyPI publishing workflow the playbook still lists as
    "future state".
spec_mode_justification: >
  Touches every layer (package rename, CLI, env vars, on-disk state migration,
  Docker, CI/CD, frontend brand strings, docs) and changes user-visible
  contracts (env-var names, state directory, install command). Full spec so
  the compat shims and migration behaviour are explicit and testable.
---

# Spec — Rebrand RLMKit → RLM Studio + packaging hygiene

## 1. Problem

1. `pip install rlmkit` resolves to a different project (see origin). The name is not recoverable.
2. The current name anchors the project as "an RLM library", a niche now owned by the paper authors' `rlms` and DSPy's first-party `dspy.RLM`. The differentiator is the workbench (Compare matrix, telemetry, traces, budgets, one-click UI); the name should say so.
3. Packaging is not release-grade: `all` includes `dev`; the `ui` extra installs a legacy Streamlit app (`src/rlmkit/ui/`) plus streamlit/plotly/gitpython/pillow; `debug_parsing.py` is tracked; classifier claims Production/Stable; `[project.urls] Documentation` points at `blob/main/`.
4. `security` CI job fails: `cryptography 49.0.0` → PYSEC-2026-3552 (fix ≥ 50.0.0).
5. No PyPI publish workflow; no wheel-with-bundle smoke test in CI.

## 2. Goals / non-goals

**Goals**
- G1 New identity everywhere: distribution `rlm-studio`, import `rlmstudio`, console script `rlm-studio` (subcommands unchanged: `studio`, `version`), GitHub `gosha70/rlm-studio`, brand string "RLM Studio", logo renamed.
- G2 Existing users of a source checkout keep working across the upgrade: env vars and the state directory migrate automatically for one release cycle.
- G3 Release-grade extras: `studio` (server + file extraction), `eval`, `interop` (placeholder for `specs/interop-official-rlm`), `dev`; `all` = `studio,eval,interop` (no `dev`). Legacy Streamlit removed from the package.
- G4 CI green including `security`; a new CI job builds the bundled wheel and smoke-tests `rlm-studio --help` + `import rlmstudio` on Python 3.11/3.12/3.13.
- G5 `release.yml`: on tag `v*` build bundle → copy to `_ui/` → `uv build` → publish via PyPI trusted publisher (OIDC); rc tags publish to TestPyPI. Optional: push sandbox image to GHCR.

**Non-goals**
- No behaviour changes to modes, sandboxes, telemetry, or the frontend beyond brand strings.
- No `rlmkit` compatibility shim package on PyPI (the name belongs to someone else) — a clean break, documented.
- The playbook stays MANUAL-ONLY; `release.yml` is the automatable sub-step it already anticipates, triggered by the owner's tag push.

## 3. Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | `src/rlmkit/` → `src/rlmstudio/`; all imports in `src/`, `tests/`, `examples/`, `scripts/`, `benchmarks/`, docs code blocks updated. `git grep -iw rlmkit` after the change matches only CHANGELOG history, the compat-shim lines, and `doc_internal/`. |
| FR-2 | `pyproject.toml`: `name = "rlm-studio"`, `[project.scripts] rlm-studio = "rlmstudio.cli:main"`, package-data `"rlmstudio._ui" = ["**/*"]`, coverage/ruff/mypy/bandit paths, URLs → `gosha70/rlm-studio` (`blob/master/`), classifier `4 - Beta`, keywords add `rlm-studio`, `evaluation`, `benchmark`. |
| FR-3 | Env vars: canonical `RLM_STUDIO_*`; the config accessor reads `RLM_STUDIO_X`, then falls back to `RLMKIT_X` with a one-time deprecation warning. Covers all 14 names in use (`_DIR, _PORT, _HOST, _CONFIG_PATH, _PERF_UI, _STREAMED_COMPLETE, _HISTORY_MAX_BYTES, _CONNECTION_TEST_INTERVAL_SECONDS_OVERRIDE, _OPENAI_DEFAULT_MODEL, _ANTHROPIC_DEFAULT_MODEL, _OLLAMA_DEFAULT_MODEL, _VERBOSE, _TIMEOUT, _MAX_STEPS`) and the frontend `NEXT_PUBLIC_RLMKIT_PERF_UI`. |
| FR-4 | State dir: canonical `~/.rlm-studio/`. On first boot, if `~/.rlm-studio/` is absent and `~/.rlmkit/` exists, copy (not move) `config.json`, `api_keys.json`, `conversations.db` and any other files, log one INFO line, and continue. `RLM_STUDIO_DIR` (or legacy `RLMKIT_DIR`) overrides as today. |
| FR-5 | Docker: image names `rlm-studio-api`, `rlm-studio-frontend`, `rlm-studio-sandbox`; compose project `rlm-studio`, volume `rlm-studio-data`, container user/home path updated; `.env.example` and `docs/OPERATIONS.md` backup command updated. |
| FR-6 | Frontend: brand string "RLM Studio" in nav/metadata/settings/learn (13 files), `frontend/package.json` name `rlm-studio-frontend`, `NEXT_PUBLIC_RLM_STUDIO_PERF_UI` with fallback. |
| FR-7 | Extras per G3. Delete only the **Streamlit-bound** files: `ui/app.py`, `ui/pages/`, `ui/components/`, `ui/session.py`, plus `ui/charts.py` (sole plotly user, unreferenced) — with their tests and coverage-omit entries. **Keep** `ui/services/` (`secret_store`, `profile_store`, … — imported by `server/app.py`, `server/routes/*`, `server/dependencies.py`, `application/services/provider_tester.py`) and `ui/data/providers_catalog.py` in place; relocating them out of `ui/` is v1.1 cleanup, not this feature. Remove `streamlit`, `plotly`, `gitpython`, `pillow` pins from `pyproject.toml` (re-run `pip-audit` — the CVE comments for those packages go away). |
| FR-8 | `cryptography>=50.0.0`; dependabot #60/#61 content folded in or rebased; `security` job green. |
| FR-9 | `debug_parsing.py` untracked (moved to `scripts/dev/` or deleted); `build/`, `htmlcov/`, `frontend/out/` in `.gitignore`. |
| FR-10 | CI: new `wheel-smoke` job — `npm ci && npm run build:bundle`, copy to `_ui/`, `uv build`, install the wheel in a fresh venv per Python 3.11/3.12/3.13, run `rlm-studio --help`, `rlm-studio version`, `python -c "import rlmstudio; from rlmstudio import interact"`, and assert `_ui/index.html` is inside the wheel. |
| FR-11 | `.github/workflows/release.yml` per playbook `steps/04` "future state": tag `v*` → build → `pypa/gh-action-pypi-publish` with `id-token: write`; `v*-rc.*` → TestPyPI. No long-lived credentials. |
| FR-12 | Docs: README, `docs/*`, `docs/hosts/*`, `CONTRIBUTING.md`, `SECURITY.md`, `RELEASING.md`, `AGENTS.md`, `CLAUDE.md`, `.github/ISSUE_TEMPLATE/*`, `CODE_OF_CONDUCT.md` renamed/updated; `docs/RLMKit_Logo.png` → `docs/RLM_Studio_Logo.png`. |
| FR-13 | Prompt YAML/JSON files contain **no** "RLMKit" mentions (verified); only `prompts/README.md` import examples and the package-resource strings in `prompts/templates.py` (`files("rlmkit.prompts")` ×3) change. Prompt template *names/keys* unchanged (cross-module constants). |
| FR-14 | Logger names `rlmkit.*` → `rlmstudio.*` (5 explicit sites: `server/app.py:86`, provider_tester, envs, connection_test_thread, nav); telemetry SQLite schema/table names unchanged (no data migration). |
| FR-15 | GitHub: repo renamed to `rlm-studio`; issues #34/#37 receive a roadmap reply; PRs #38/#39/#41 closed with a note; repo description/topics updated; `github.com/gosha70/rlmkit` URLs (3 files) and `.github/ISSUE_TEMPLATE/*` path refs updated. |
| FR-16 | Telemetry JSONL export extras keys `rlmkit_query / _success / _total_tokens / _total_cost / _elapsed_seconds / _raw_steps_count` (`telemetry/store.py:673-681`) → `rlm_studio_*`. These are exported keys — rename now while nothing is published; the reader side (if any) and tests (`test_telemetry_store.py:287-291,389`) updated. |
| FR-17 | Keyring service name `"rlmkit"` (`ui/services/secret_store.py:139`) → `"rlm-studio"` with **read fallback** to the old service name and re-save under the new one (not a plain sed). |
| FR-18 | `src/rlmkit/rlmkit_config.default.yaml` → `src/rlmstudio/rlm_studio_config.default.yaml`; package-data key updated; `config.py:305-308` search list reads `./rlm_studio_config.{yaml,json}` and `~/.rlm-studio/config.{yaml,json}` first, then the legacy `./rlmkit_config.*` / `~/.rlmkit/config.*` names (deprecated, one cycle). |
| FR-19 | Docker sandbox image default `"rlmkit-sandbox"` baked into `envs/sandbox.py:204,213` and `infrastructure/sandbox/docker_sandbox_adapter.py:33` → one constant `DEFAULT_SANDBOX_IMAGE = "rlm-studio-sandbox"` imported by both; `docker/README.md`, CI, compose aligned. Dockerfile.api Linux user `rlmkit` → `rlmstudio`, venv `/opt/rlmkit-venv` → `/opt/rlm-studio-venv`. |
| FR-20 | Remaining renames: `docs/RLMKit_Design_Document.md` → `docs/RLM_Studio_Design_Document.md` (+ links), `frontend/public/logo.png` refreshed with the new mark, `.pre-commit-config.yaml` paths (3 lines), `benchmarks/sample_benchmark.yaml` `created_by`, `uv.lock` regenerated. Frontend `localStorage` keys `rlmkit_*` (7 keys in `app/page.tsx`): one-shot read-old/write-new migration on load, then new keys only. |
| FR-21 | Tests that assert on the string/path: fix `tests/e2e/test_docs.py:85` **first** (asserts the *checkout directory name* contains "rlmkit" — breaks on any clone name; assert on a repo marker file instead); update `tests/test_cli.py` (12 lines), `test_database.py:10`, `test_trace_writers.py:21`, `test_subprocess_sandbox.py:200,428`, `test_public_client.py:263-264`, `test_save_config_atomicity.py`, the 181 `patch("rlmkit.…")` strings across 22 test files (sed-safe). |

## 4. Constraints
- Dependency rule and no-magic-strings rules apply: env-var names and state paths are defined once (constants module) and imported.
- Every step keeps CI green; the rename lands as a sequence of reviewable commits on `feat/rebrand-rlm-studio` (see plan §3).
- Playbook remains manual; the owner pushes the tag.
- No behaviour regressions: full backend + frontend suites pass; coverage ≥ 80% on `application/`, `infrastructure/`.

## 5. Acceptance criteria
- AC-1 Fresh clone: `uv sync --extra all && uv run pytest -n auto` green; `ruff`, `mypy`, `bandit`, `pip-audit` green.
- AC-2 `uv build` wheel installs on 3.11/3.12/3.13; `rlm-studio --no-browser` serves `/studio` with the Dashboard and no console errors.
- AC-3 With `RLMKIT_PORT=8123` set and no `RLM_STUDIO_PORT`, the server binds 8123 and logs a deprecation warning once.
- AC-4 With an existing `~/.rlmkit/` and no `~/.rlm-studio/`, first boot copies state; providers and telemetry appear unchanged in the UI.
- AC-5 `docker compose up --build` boots both services; a Direct-mode chat completes.
- AC-6 `git grep -iw rlmkit -- ':!CHANGELOG.md' ':!doc_internal' ':!specs'` returns only the compat-shim constants and their tests.
- AC-7 `release.yml` dry-run publishes `v1.0.0-rc.1` to TestPyPI and `pip install -i https://test.pypi.org/simple/ rlm-studio` works in a clean venv.
- AC-8 `pip install rlm-studio[all]` pulls no `pytest`/`mypy`/`ruff`/`streamlit`.

## 6. Open questions (defaults apply if unanswered)
- OQ-1 Version number — owner decides; two defensible options:
  - **1.0.0** (default in the release plan): "first public release" story; requires folding the RLMKit `[1.0.0] — 2026-03-28` + `[Unreleased]` entries under the new header as "Pre-public history (as RLMKit)" so no second 1.0.0 header exists.
  - **2.0.0** (recommended by the rename inventory pass): CHANGELOG already carries `[1.0.0]`, README/OPERATIONS/compose describe "v1.1" and a 1.0→1.1 schema migration; import path, CLI, env vars, state dir, Docker names and repo URL all change — textbook major. Keeps history verbatim ("1.x = unpublished RLMKit era"), less CHANGELOG surgery, at the cost of a first public release numbered 2.0.
  - 1.1.0 is defensible only with the same CHANGELOG rewrite as 1.0.0 and is the least honest of the three.
- OQ-4 State-dir migration mode: default **copy** (`~/.rlmkit/` left intact, INFO log); alternative `shutil.move` with a marker file. Copy is safer for users who run old and new side by side during the cycle.
- OQ-2 Import name: default `rlmstudio` (single word, matches distribution). Alternative `rlm_studio`.
- OQ-3 Keep the Streamlit app anywhere? Default: delete (it is unreferenced and coverage-excluded); tag the last commit containing it as `archive/streamlit-ui` for history.
