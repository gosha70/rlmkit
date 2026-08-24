# Operations Guide

This doc covers what operators of a single-node RLM Studio deployment need to
know: where state lives on disk, how to back it up, how upgrades work,
and what workloads RLM Studio v1.0 does and does not support.

For install and run instructions, see the [README](../README.md). For
per-provider setup, see [docs/hosts/](hosts/).

---

## 1. What RLM Studio v1.0 is (and is not)

RLM Studio v1.0 is a **single-node self-hosted** product. One backend process,
one frontend process, one user or one small team sharing that single
backend. The [README § Deployment model & support boundary](../README.md#deployment-model--support-boundary)
lists what is supported and what is out of scope.

**The one sentence summary:** v1.0 is the right fit for developers,
prompt engineers, and small teams running on their own machine or on a
shared workstation. It is not a multi-tenant service, and concurrent
large-document workloads from different users are unsupported in v1.1.

**Why this matters for ops:** two operators uploading 8MB documents to
the same backend at the same time will both execute; RLM Studio does not
queue them. The inference backend behind RLM Studio (vLLM, Ollama, cloud API)
handles that concurrency, not RLM Studio itself. If that inference backend
saturates, both runs will slow or fail. If you need hard isolation
between users, run separate RLM Studio backends — one per user or team —
behind a reverse proxy.

---

## 2. Where state lives

Every piece of mutable state RLM Studio writes lives under `~/.rlm-studio/` on
the backend host, with one exception (API keys may live in the OS
keyring — see §2.5).

### 2.1 Filesystem layout

```
~/.rlm-studio/
├── config.json          # Chat providers, profiles, budget, prompts, theme
├── sessions.json        # Chat sessions (conversation history per session)
├── evaluations.json     # LLM-as-judge scores and pairwise comparisons
├── api_keys.json        # Fallback API-key store (chmod 600) — used when
│                        # no OS keyring is available
└── telemetry.db         # SQLite — runs, steps, provider_calls, ratings
```

On older installs you may also see these legacy files in the working
directory (CWD-relative):

```
./.rlmkit_config.json          # auto-migrated to ~/.rlm-studio/config.json on first launch
./.rlmkit_sessions.json        # auto-migrated to ~/.rlm-studio/sessions.json
./.rlmkit_evaluations.json     # auto-migrated to ~/.rlm-studio/evaluations.json
```

The migration is one-way: after the first successful move, the legacy
files are deleted. If you want to keep the old path for some reason,
symlink `~/.rlm-studio/<file>` to wherever you prefer before the first
launch.

### 2.1a Upgrading from RLMKit (`~/.rlmkit/`)

RLM Studio was previously published as **RLMKit**, which kept its state in
`~/.rlmkit/`. On the **first server boot** (`rlm-studio studio`,
`python -m rlmstudio.server`, uvicorn, or the Docker image) after the
upgrade, if `~/.rlm-studio/` does not exist yet and `~/.rlmkit/` does, the
whole legacy directory is **copied** to `~/.rlm-studio/` and one INFO line
is logged. Nothing is moved or deleted; `~/.rlmkit/` stays as-is and can be
removed by hand once you are happy with the new install. The copy is a
one-time cost proportional to the size of `telemetry.db`.

The copy runs only at server startup — never on `import rlmstudio`, in the
test suite, on `rlm-studio version`, or when `RLM_STUDIO_DIR` points the
state directory elsewhere.

Related one-cycle compatibility shims:

- Environment variables: the canonical prefix is `RLM_STUDIO_*`
  (`RLM_STUDIO_HOST`, `RLM_STUDIO_PORT`, `RLM_STUDIO_CONFIG_PATH`,
  `RLM_STUDIO_DIR`, `RLM_STUDIO_HISTORY_MAX_BYTES`,
  `RLM_STUDIO_STREAMED_COMPLETE`, …). The old `RLMKIT_*` names are still
  read as a fallback and log a one-time deprecation warning naming the
  new variable; they will be dropped in the next minor release.
- Config file: `./rlm_studio_config.{yaml,json}` is searched first, then
  the legacy `./rlmkit_config.{yaml,json}`, then `~/.rlm-studio/config.*`
  and `/etc/rlm-studio/config.*`.
- OS keyring: provider keys are stored under the service name
  `rlm-studio`; keys saved by RLMKit under `rlmkit` are read as a fallback
  and re-saved under the new service name on first use.
- Docker Compose: the named volume is now `rlm-studio-data`, mounted at
  `/home/rlmstudio/.rlm-studio`. Copy the old `rlmkit-data` volume once
  before the first `up` (recipe in the `docker-compose.yml` header).
- Frontend `localStorage`: the chat page migrates its `rlmkit_*` keys to
  `rlm_studio_*` on first load.
- Python: the import package is now `rlmstudio` (the old `rlmkit` package name is gone); the distribution is
  `rlm-studio`. There is no `rlmkit` shim on PyPI — that name belongs to an
  unrelated project.

### 2.2 `config.json`

Configuration entered through Settings (LLM Providers, Chat Providers,
Profiles, budget, system prompts, theme, scheduled-connection-testing
interval) is persisted atomically on every change. Size: typically
< 100 KB.

### 2.3 `sessions.json`

Chat sessions, including per-session conversation history. One entry
per session with timestamps and message arrays. Size scales with
session count and history length — expect single-digit to low-double
-digit MB after months of active use.

### 2.4 `evaluations.json`

Judge scores (pointwise rubric v2.0) and pairwise comparisons produced
by the LLM-as-judge workflow. Referenced by the Dashboard and by the
Compare page's judge-score ranking.

### 2.5 `api_keys.json` (fallback) and the OS keyring

On startup RLM Studio probes for an OS keyring via the [`keyring`](https://pypi.org/project/keyring/)
library:

- **macOS:** Keychain.
- **Windows:** Windows Credential Manager.
- **Linux:** Secret Service (GNOME Keyring, KWallet).

If a keyring is available, **API keys persisted through Settings go into
the keyring**, and `~/.rlm-studio/api_keys.json` is never written. If no
keyring is available (most headless Linux servers, CI runners), RLM Studio
falls back to a `chmod 600` JSON file at `~/.rlm-studio/api_keys.json`.

Precedence on startup, highest wins:

1. Real process environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …).
2. SecretStore (keyring or `api_keys.json`).
3. Legacy `.env` file at the repo root, if present.

An existing env var always wins. This is intentional — ops teams
deploying via systemd, Kubernetes, Docker Compose, etc. typically inject
secrets via the environment, and those should override anything
RLM Studio's Settings UI wrote.

### 2.6 `telemetry.db`

SQLite database holding every run, step, provider call, and rating.
Default path: `~/.rlm-studio/telemetry.db`. Enabled by default; disable it
by creating an in-memory `SQLiteStorageAdapter` programmatically (see
`src/rlmstudio/telemetry/store.py`).

Size grows with usage — roughly 1–5 KB per RLM step, 0.5–1 KB per
direct call. A year of moderate use (hundreds of runs per week) stays
under 500 MB. WAL mode is enabled for concurrent reads while writing.

**Schema versioning** via SQLite's `PRAGMA user_version`. Current
version as of v1.1 is `2`. See §4 for upgrade behavior.

---

## 3. Backup and restore

### 3.1 What to back up

The single directory `~/.rlm-studio/` captures everything except API keys
held in the OS keyring. For a complete backup:

```bash
# Stop the backend first — SQLite WAL checkpoints cleanly on a clean exit.
systemctl stop rlm-studio        # or: docker compose down, or Ctrl-C the process

# Tarball the directory
tar -czf rlm-studio-backup-$(date +%F).tar.gz -C "$HOME" .rlm-studio/

# If you use the OS keyring for API keys, export those separately —
# keyring content is not in ~/.rlm-studio/. On macOS:
#   security export -k ~/Library/Keychains/login.keychain-db -o rlm-studio-keys.p12
# On Linux Secret Service: use `secret-tool search rlm-studio` and save the output.
```

A weekly cron job with a one-week retention is adequate for the
workload v1.1 targets. If you care about evaluation history or
long-running experiments, move to daily.

### 3.2 What to restore

```bash
# Backend must be stopped before restore.
systemctl stop rlm-studio
rm -rf ~/.rlm-studio/
tar -xzf rlm-studio-backup-2026-04-24.tar.gz -C "$HOME"
systemctl start rlm-studio
```

On first launch after restore, RLM Studio re-reads `config.json`,
`sessions.json`, `evaluations.json`, and the SQLite DB. If the backup
was taken from an older version that used schema v1, the v2 migration
runs automatically on first launch — see §4.

### 3.3 What not to back up

- The `uv`/`pip` virtual environment — rebuild it from `pyproject.toml`
  / `uv.lock`.
- Anything under `frontend/node_modules/` — rebuild from
  `package-lock.json`.
- The running process's in-memory state — by design, stopping the
  backend cleanly flushes everything that matters to disk.

### 3.4 Disaster recovery RTO

On commodity hardware a restore from a ~100 MB tarball plus a
`uv sync` plus a `npm ci` and first-boot takes under five minutes. For
single-node v1.1, that's the recovery-time objective.

---

## 4. Upgrade behavior

> Version numbers in this section (`v1.0.x`, `v1.1`) refer to the
> pre-public **RLMKit** lineage — in particular the telemetry schema step
> from version 1 to 2. The first public RLM Studio release carries that
> schema forward unchanged; the RLMKit → RLM Studio path itself is
> described in §2.1a.

### 4.1 SQLite schema auto-migration

RLM Studio uses `PRAGMA user_version` as the schema-version store. The
`_MIGRATIONS` dict in `src/rlmstudio/telemetry/store.py` is the source of
truth for every post-v1 schema change.

**Upgrading from v1.0.x to v1.1.0:**

1. Stop the backend.
2. Install the new version (`uv sync`, `docker compose pull`, etc.).
3. Start the backend.

On first launch the telemetry store reads `PRAGMA user_version`, sees
it is below the target (2 in v1.1), and applies the v2 migration in a
single transaction: six new columns on `steps` (`prompt_tokens`,
`completion_tokens`, `ttft_ms`, `decode_ms`, `cached_tokens`,
`cache_write_tokens`) and one new column on `runs`
(`outcome_category`). Existing rows are preserved and get defaults or
`NULL`. The pragma is bumped after the transaction commits.

**No manual SQL, no downtime beyond the process restart, no
data migration scripts.** Runs recorded before the upgrade keep
rendering in the Dashboard and Traces; they just have `NULL` or `0`
values in the new columns.

Re-opening an already-migrated database is a no-op — the pragma
already matches the target.

**The migration is forward-only.** There are no `DOWN` migrations. If
you need to revert to v1.0.x, restore from a pre-upgrade backup (§3)
rather than trying to un-migrate the DB.

### 4.2 Config-file shape

`config.json`, `sessions.json`, and `evaluations.json` are
version-tolerant: missing keys fall back to defaults, unknown keys are
preserved. A v1.1 backend reads a v1.0 config without ceremony.

### 4.3 API compatibility

The REST API under `/api/...` is semver-compatible within the v1.x
line: v1.0 clients continue to work against v1.1 backends. Breaking
changes require a major version bump or a versioned route prefix.
See [CONTRIBUTING.md](../CONTRIBUTING.md) for the versioning stance.

### 4.4 Downgrade

Downgrading from v1.1 to v1.0 is not supported. The v1.0 telemetry
store reads v2 schema columns it doesn't know about as NULL (harmless),
but any `outcome_category` classification performed by v1.1 will be
ignored. If you genuinely need to downgrade, restore from a
pre-upgrade backup rather than running v1.0 against a v1.1 database.

---

## 5. Disk-space hygiene

The things that grow unboundedly with use are:

- `telemetry.db` (runs + steps). Low-write: trace deletion via the
  Traces UI reclaims space on the next `VACUUM`. To force a manual
  reclaim:

  ```bash
  sqlite3 ~/.rlm-studio/telemetry.db "VACUUM;"
  ```

  Safe to run while the backend is stopped; do not run during active
  writes.

- `sessions.json`. Delete sessions from the Chat sidebar to trim.
  The file is rewritten atomically on every save.

- Application logs. RLM Studio writes to stderr by default; if you redirect
  to a file (systemd `StandardOutput=append:/var/log/rlmstudio.log`, etc.),
  rotate it with `logrotate` or equivalent.

For a single-node v1.1 deployment a yearly `VACUUM` + ad-hoc session
cleanup is enough.

---

## 6. Health checks

- **Liveness:** `GET /health` returns `{status, version, uptime_seconds}`
  when the process is up. Use this as a Kubernetes liveness probe or
  a systemd watchdog target.
- **Provider connectivity:** the scheduled-connection-testing
  background thread (opt-in; `connection_test_interval_minutes` in
  Settings) updates each LLM Provider's status. The UI surfaces those
  statuses on the Settings page. A deeper provider-status summary on
  `/health?detail=true` is tracked as a later enhancement.

---

## 7. Running under Docker

See the Dockerfiles at [docker/Dockerfile.api](../docker/Dockerfile.api)
and [docker/Dockerfile.frontend](../docker/Dockerfile.frontend), and
the compose stack at [docker-compose.yml](../docker-compose.yml). A
one-command smoke test:

```bash
cp .env.example .env          # fill in at least one provider key
docker compose up --build
```

Backend is then reachable on `http://localhost:8000`, frontend on
`http://localhost:3000`. State persists to the
`rlm-studio-data` named volume (mapped to `/home/rlmstudio/.rlm-studio` inside the
backend container) — back it up the same way as a bare-metal deployment
(§3), e.g.:

```bash
docker run --rm -v rlm-studio-data:/data -v "$PWD:/backup" alpine \
  tar -czf /backup/rlm-studio-backup-$(date +%F).tar.gz -C /data .
```

---

## 8. Known limitations (v1.0)

Carried over from the CHANGELOG:

- **No timeout enforcement in non-main threads.** Signal-based
  timeouts are unavailable outside the main thread; code execution
  proceeds without a timeout guard in that path. A warning is logged
  when the path is taken. Out-of-process sandbox execution is planned
  for a later milestone.
- **No automatic provider failover.** If the configured provider goes
  offline, the run fails with a classified error (see Dashboard
  outcome classification); it does not transparently retry against a
  different provider.
- **No cross-process queue.** Concurrent heavy workloads rely on the
  inference backend's own queuing (vLLM has prefix-caching and KV-cache
  eviction; Ollama serializes). RLM Studio does not add an application-
  level queue in v1.1.

These items are tracked as separate milestones, not v1.1 issues.
