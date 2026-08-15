---
feature_id: rebrand-rlm-studio
spec_mode: full
spec: ./spec.md
status: draft
date: 2026-08-15
origin:
  urls:
    - https://pypi.org/project/rlmkit/
  transcripts:
    - "Owner, 2026-08-15: chose 'Rebrand to RLM Studio now'."
  origin_claim: |
    Inherited from spec.md — the PyPI name collision forces a rename before the
    first public release; the owner chose RLM Studio (dist rlm-studio, import
    rlmstudio, repo gosha70/rlm-studio) and a hygiene pass that makes
    `pip install` credible.
---

# Plan — Rebrand RLMKit → RLM Studio + packaging hygiene

Implements `spec.md`. Branch: `feat/rebrand-rlm-studio` off `master`. Estimated effort: 3–4 working days (mostly mechanical) + ½ day for `release.yml` bootstrap outside a release window.

## 1. Inventory (measured 2026-08-15 on `7fd093e`)

| Group | Count / where | Nature |
|---|---|---|
| Files mentioning `rlmkit` (any case, tracked) | 266 | mixed |
| Python import sites `from/import rlmkit` | 187 files / 490 lines (`src` 79, `tests` 102, `examples` 4, `scripts` 1) + 363 non-import `rlmkit` lines in `.py` (181 are `patch("rlmkit.…")` strings in 22 test files; `files("rlmkit.prompts")` ×3 in `prompts/templates.py`; `_pkg_version("rlmkit")` in `__init__.py:117`; `"rlmkit.server.app:app"` in `cli/main.py:153`) | mechanical (`\brlmkit\b` sed, excluding `.rlmkit`, `RLMKIT_`, `rlmkit_`, `rlmkit-` patterns which are handled by name) |
| Frontend brand strings | 13 files; still "RLMKit": `sidebar.tsx:126,132`, `settings/page.tsx:1498`, `lib/api.ts:2,15`; already "RLM Studio": `layout.tsx` title, learn pages, `app-shell.tsx`; `localStorage` keys `rlmkit_*` ×7 in `app/page.tsx`; `frontend/public/logo.png` | mechanical + one-shot key migration |
| Prompts | **0** mentions in YAML/JSON; only `prompts/README.md` + `prompts/templates.py` package strings | mechanical |
| Telemetry JSONL extras keys `rlmkit_*` | `telemetry/store.py:673-681` (6 keys) + tests | mechanical, but exported — do it now |
| Keyring service name `"rlmkit"` | `ui/services/secret_store.py:139` (+ `~/.rlmkit/api_keys.json` file path `:83`) | **judgment** (read-fallback + re-save) |
| Default config filename | `src/rlmkit/rlmkit_config.default.yaml`, package-data, `config.py:305-308` search list | **judgment** (legacy names still read one cycle) |
| Sandbox image default | `envs/sandbox.py:204,213`, `infrastructure/sandbox/docker_sandbox_adapter.py:33`, `docker/README.md`, CI | mechanical → one constant |
| Misc | `docs/RLMKit_Design_Document.md`, `.pre-commit-config.yaml` (3 lines), `benchmarks/sample_benchmark.yaml:6`, `Dockerfile.api` user/venv, `.github/ISSUE_TEMPLATE/*`, `github.com/gosha70/rlmkit` URLs (3 files) | mechanical |
| Tests asserting on strings/paths | `tests/e2e/test_docs.py:85` (checkout dir name — fix first), `test_cli.py` (12), `test_database.py:10`, `test_trace_writers.py:21`, `test_subprocess_sandbox.py:200,428`, `test_public_client.py:263-264`, `test_save_config_atomicity.py`, `test_telemetry_store.py:287-291,389` | mechanical after FR-21 |
| Docs mentioning rlmkit | 17 files (README, `docs/`, CONTRIBUTING, SECURITY, RELEASING, AGENTS, `.github/`) | mechanical + copy edits |
| Env vars `RLMKIT_*` | 14 distinct names; read in `src/rlmkit/config.py`, `server/dependencies.py`, `server/routes/*`, `cli/main.py`, `docker-compose.yml`, `.env.example`, CI, frontend `NEXT_PUBLIC_RLMKIT_PERF_UI` | **judgment** (compat) |
| State paths | `~/.rlmkit/` (`server/dependencies.py:58-70`, `config.py:307`, `storage/database.py:84`, `infrastructure/storage/sqlite_adapter.py:20`, `routes/providers.py`), legacy CWD files `.rlmkit_config.json` etc. | **judgment** (migration) |
| Docker | `docker-compose.yml` (project, images, containers, volume, home path), `docker/Dockerfile*`, CI `docker-sandbox` job | mechanical |
| Logger names | 5 `getLogger("rlmkit…")` | mechanical |
| Legacy Streamlit | Streamlit-bound: `ui/app.py`, `ui/pages/*`, `ui/components/*`, `ui/session.py`; plotly-only: `ui/charts.py`. **Not** deletable: `ui/services/*`, `ui/data/providers_catalog.py` (imported by `server/`, `application/services/provider_tester.py`) | delete the first set only |

## 2. Allowlist (paths this feature may touch)
`src/**` (rename), `tests/**`, `examples/**`, `scripts/**`, `benchmarks/**`, `pyproject.toml`, `uv.lock`, `.gitignore`, `.github/**`, `docker/**`, `docker-compose.yml`, `.env.example`, `frontend/package.json`, `frontend/package-lock.json`, `frontend/next.config.ts`, `frontend/src/**` (brand strings + env name only), `README.md`, `CHANGELOG.md`, `RELEASING.md`, `SECURITY.md`, `CONTRIBUTING.md`, `AGENTS.md`, `CLAUDE.md`, `CODE_OF_CONDUCT.md`, `docs/**`, `debug_parsing.py` (remove). Nothing else.

## 3. Execution order (each step = one commit, CI green after each)

0. **Prep (no rename yet).** Fix `tests/e2e/test_docs.py:85` to assert on a repo marker (e.g. `pyproject.toml` exists) instead of the checkout directory name; centralise the ~12 independent `Path.home() / ".rlmkit"` sites (`server/dependencies.py:60-67`, `telemetry/store.py:213`, `storage/database.py:85`, `ui/services/secret_store.py:83`, `profile_store.py:131,195`, `llm_config_manager.py:63`, `chat_manager.py:89,623,777`, `config.py:305-308`, plus Streamlit files about to be deleted) onto one `STATE_DIR` accessor. CI green with the old name still in place.
1. **Constants first.** Add `src/rlmkit/branding.py` (to become `src/rlmstudio/branding.py`): `DIST_NAME="rlm-studio"`, `PACKAGE_NAME="rlmstudio"`, `PRODUCT_NAME="RLM Studio"`, `CLI_NAME="rlm-studio"`, `ENV_PREFIX="RLM_STUDIO_"`, `LEGACY_ENV_PREFIX="RLMKIT_"`, `STATE_DIR_NAME=".rlm-studio"`, `LEGACY_STATE_DIR_NAME=".rlmkit"`, and `env(name, default=None)` — the single accessor implementing FR-3 (canonical → legacy with one-time `warnings.warn`/log). Replace every `os.environ.get("RLMKIT_…")`/`os.getenv` with `branding.env("…")`. Tests: `tests/test_branding_env.py` (canonical wins, legacy fallback warns once, neither → default).
2. **State-dir migration.** In `server/dependencies.py` generalise the existing legacy-file migration (`_LEGACY_CONFIG_FILE` block) into `ensure_state_dir()` implementing FR-4; use it from `config.py`, `storage/database.py`, `sqlite_adapter.py`, `routes/providers.py` (import the path from one place — no duplicated `Path.home() / ".rlmkit"`). Tests with `tmp_path` + monkeypatched `HOME`.
3. **Package rename.** `git mv src/rlmkit src/rlmstudio`; `sed` imports across `src tests examples scripts benchmarks docs`; update `pyproject.toml` (FR-2), CI paths, logger names (FR-14), prompts YAML values (FR-13), `ui_bundle.py` docstrings, `cli/main.py` help text (`rlm-studio studio`). Run full suite.
3b. **Exported names + baked defaults** (FR-16..FR-19): telemetry JSONL extras keys → `rlm_studio_*`; keyring service `"rlm-studio"` with read-fallback + re-save; default config filename + legacy search list; `DEFAULT_SANDBOX_IMAGE` constant; `prompts/templates.py` package strings; `__init__.py` `_pkg_version("rlm-studio")`; `cli/main.py` app path string.
4. **Drop legacy Streamlit + extras** (FR-7): delete `ui/app.py`, `ui/pages/`, `ui/components/`, `ui/session.py`, `ui/charts.py` (+ their tests, coverage-omit entries); keep `ui/services/`, `ui/data/`; new extras `studio`, `eval`, `interop` (empty list for now), `dev`; `all = ["rlm-studio[studio,eval,interop]"]`; remove streamlit/plotly/gitpython/pillow pins; `uv lock`; `pip-audit` (FR-8: `cryptography>=50.0.0`).
5. **Docker + compose + env example** (FR-5). Rebuild locally: `docker compose up --build`.
6. **Frontend brand + env** (FR-6): `frontend/src/**` strings, `package.json` name, `NEXT_PUBLIC_RLM_STUDIO_PERF_UI` (read both during the cycle), `next.config.ts` comments. `npm test && npm run build:bundle`.
7. **Docs + logo** (FR-12): README install block → `pip install "rlm-studio[all]" && rlm-studio studio`; `RELEASING.md` bundle path `../src/rlmstudio/_ui/`; `docs/OPERATIONS.md` new env names, state dir, migration note; `SECURITY.md` supported versions; `CHANGELOG.md` `[Unreleased]` → "Changed — **BREAKING:** project renamed…" + Migration callout; `CLAUDE.md`/`AGENTS.md` paths.
8. **Hygiene** (FR-9): `git rm debug_parsing.py` (or move to `scripts/dev/`), `.gitignore` additions.
9. **CI wheel-smoke job** (FR-10) in `ci.yml` (matrix 3.10/3.11/3.12; needs Node 22 for the bundle).
10. **`release.yml`** (FR-11) — written now, exercised in M3 with `v1.0.0-rc.1` after the owner configures the trusted publisher on TestPyPI/PyPI (manual, playbook `steps/04`).
11. **GitHub rename + triage** (FR-15) — owner action after merge: rename repo, update local remotes, reply on #34/#37, close #38/#39/#41.

## 4. Test strategy
- Unit: `test_branding_env.py`, `test_state_dir_migration.py`, updated `tests/test_cli.py` (`rlm-studio version`, `--help`, "no UI bundled" path), dependency-pin-guard tests (extras composition: `all` excludes `dev`).
- Full suites: `uv run pytest -n auto --runslow`, `cd frontend && npm test`.
- Integration: AC-2 wheel smoke (CI job), AC-4 migration on a machine with an existing `~/.rlmkit/`, AC-5 compose.
- Grep gate: AC-6 command in `spec.md` run in CI as a lint step (allowlist the shim constants file).

## 5. Risks
| Risk | Mitigation |
|---|---|
| Missed `rlmkit` string in a runtime path (LiteLLM metadata tag, User-Agent, prompt) | AC-6 grep gate in CI; review prompts YAML diff by hand |
| Users on a source checkout lose state | FR-4 copy-migration + INFO log + OPERATIONS.md note; nothing is deleted from `~/.rlmkit/` |
| Frontend/backend env name drift (`NEXT_PUBLIC_*`) | both names read during the cycle; test in `frontend/src/__tests__` |
| Trusted-publisher misconfiguration on first real tag | rc dry-run to TestPyPI first (AC-7), outside the release window |
