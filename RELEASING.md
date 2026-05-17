# Releasing RLMKit

RLMKit releases are cut by the project maintainer following a manual playbook. This document is the public-facing pointer to the artifacts that travel with each release.

## Where to look

- **What shipped in each version** — [`CHANGELOG.md`](CHANGELOG.md). Format follows [Keep a Changelog](https://keepachangelog.com/); every release header lists added / changed / fixed / security / migration items. Breaking changes carry an explicit `**BREAKING:**` marker.
- **Reporting a vulnerability** — [`SECURITY.md`](SECURITY.md). Coordinated disclosure policy, contact address, and supported-version table. Please do not file public issues for security reports.
- **Operational details for a deployed instance** — [`docs/OPERATIONS.md`](docs/OPERATIONS.md). State paths, env vars, upgrade behavior between versions, backup and restore procedure.
- **Versioning** — [SemVer](https://semver.org/). Major releases break public API surface; minor releases are additive; patch releases are fixes and security bumps. Pre-releases use `vX.Y.Z-rc.N`.

## How releases are cut

For each release, the maintainer:

1. Confirms every CI gate passes on `master` and runs a manual test plan on a fresh checkout.
2. Opens a release PR that bumps `pyproject.toml`, promotes the `[Unreleased]` section in `CHANGELOG.md` to a dated version header, and updates any docs whose user-facing copy changed.
3. After merge, pushes an annotated tag `vX.Y.Z` to `master` and publishes a GitHub release with the wheel and sdist attached.
4. Verifies the published artifacts install cleanly on Python 3.10, 3.11, and 3.12 from a fresh environment before announcing.

CI gates and the seven enforced jobs are defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## For contributors

If you are preparing a change for an upcoming release:

- Add your CHANGELOG entry under the `## [Unreleased]` section in your PR. Use the Added / Changed / Fixed / Security / Known Limitations subsections to mirror the format of past releases.
- If your change is breaking, prefix the bullet with `**BREAKING:**` and add a one-line migration note describing what users need to do on upgrade.
- See [`CONTRIBUTING.md`](CONTRIBUTING.md) for code conventions, test requirements, and the PR template.

## Internal maintainer playbook

The detailed step-by-step procedure the maintainer follows — including announcement copy, channel sequencing, hotfix and yank flows — lives under `doc_internal/release/`. That folder is gitignored because it carries per-channel announcement drafts and contact lists that do not need to be in the public repo. Forks that need an equivalent playbook for their own releases can use [`CHANGELOG.md`](CHANGELOG.md) as the contract and the public sections of this document as the procedure.
