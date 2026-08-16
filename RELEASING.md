# Releasing RLM Studio

RLM Studio releases are cut by the project maintainer following a manual playbook. This document is the public-facing pointer to the artifacts that travel with each release.

## Where to look

- **What shipped in each version** — [`CHANGELOG.md`](CHANGELOG.md). Format follows [Keep a Changelog](https://keepachangelog.com/); every release header lists added / changed / fixed / security / migration items. Breaking changes carry an explicit `**BREAKING:**` marker.
- **Reporting a vulnerability** — [`SECURITY.md`](SECURITY.md). Coordinated disclosure policy, contact address, and supported-version table. Please do not file public issues for security reports.
- **Operational details for a deployed instance** — [`docs/OPERATIONS.md`](docs/OPERATIONS.md). State paths, env vars, upgrade behavior between versions, backup and restore procedure.
- **Versioning** — [SemVer](https://semver.org/). Major releases break public API surface; minor releases are additive; patch releases are fixes and security bumps. Pre-releases use `vX.Y.Z-rc.N`.

## How releases are cut

For each release, the maintainer:

1. Confirms every CI gate passes on `master` and runs a manual test plan on a fresh checkout.
2. Builds the bundled Studio UI so `rlm-studio studio` ships a working one-click experience:
   ```bash
   cd frontend
   npm install
   npm run build:bundle             # produces frontend/out/

   # Wipe the previous bundle contents but PRESERVE ../src/rlmstudio/_ui/__init__.py.
   # That file is what makes `rlmstudio._ui` a discovered package under
   # [tool.setuptools.packages.find] — without it, `rlmstudio._ui` is not
   # found, `"rlmstudio._ui" = ["**/*"]` in package-data has nothing to
   # attach to, and the wheel ships without the bundled UI even though
   # the source tree looks correct locally.
   find ../src/rlmstudio/_ui -mindepth 1 -not -name __init__.py -delete
   cp -r out/. ../src/rlmstudio/_ui/   # the wheel ships this directory

   # Verify the package marker survived and the bundle landed.
   test -f ../src/rlmstudio/_ui/__init__.py
   test -f ../src/rlmstudio/_ui/index.html
   ```
   `build:bundle` (not `build`) is the right script — `build` produces a Next.js server build (`.next/`), `build:bundle` sets `BUNDLE=1` and emits a static export under `frontend/out/`.
3. Opens a release PR that bumps `pyproject.toml`, promotes the `[Unreleased]` section in `CHANGELOG.md` to a dated version header, and updates any docs whose user-facing copy changed.
4. After merge, pushes an annotated tag `vX.Y.Z` to `master` and publishes a GitHub release with the wheel and sdist attached.
5. Verifies the published artifacts install cleanly on Python 3.10, 3.11, and 3.12 from a fresh environment before announcing.

Between releases, `src/rlmstudio/_ui/` is intentionally empty (only the `__init__.py` placeholder is tracked) so the dev workflow stays clean. The `_ui/` payload is regenerated at every release cut.

CI gates and the seven enforced jobs are defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## For contributors

If you are preparing a change for an upcoming release:

- Add your CHANGELOG entry under the `## [Unreleased]` section in your PR. Use the Added / Changed / Fixed / Security / Known Limitations subsections to mirror the format of past releases.
- If your change is breaking, prefix the bullet with `**BREAKING:**` and add a one-line migration note describing what users need to do on upgrade.
- See [`CONTRIBUTING.md`](CONTRIBUTING.md) for code conventions, test requirements, and the PR template.

## Internal maintainer playbook

The detailed step-by-step procedure the maintainer follows — including announcement copy, channel sequencing, hotfix and yank flows — lives under `doc_internal/release/`. That folder is gitignored because it carries per-channel announcement drafts and contact lists that do not need to be in the public repo. Forks that need an equivalent playbook for their own releases can use [`CHANGELOG.md`](CHANGELOG.md) as the contract and the public sections of this document as the procedure.
