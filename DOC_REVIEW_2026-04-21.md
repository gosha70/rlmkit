# Docs review — RLMKit, last 30 days

**Scope:** code + doc changes on `gosha70/rlmkit` between 2026-03-22 and 2026-04-21
(232 commits, 4 feature/fix PRs: #20, #22, #23, #24, plus 3 dependabot: #18, #19, #21).
The rest of the work landed as direct pushes to `master`.
**Inputs:** the repo itself, plus two uploaded setup notes
(`DGX_Spark_RLMKit_Setup_Cookbook_v2.md`, `DGX_Spark_Setup_Cookbook_v4b.md`).

The short version: the last 30 days were **feature-heavy and doc-light**. The
Learn tab ships a Cookbook UI that surfaces real user-facing docs
(`docs/hosts/*.md`, `docs/troubleshoot.yaml`), but several *other* big
features from the same period — LLM Tuner, LLM-as-judge, outcome classifier,
scheduled connection testing, conversation memory — landed without any
matching updates to the canonical user guide. A few long-lived files
(`CLAUDE.md`, `AGENTS.md`, `CONTRIBUTING.md`, `.env.example`,
`docs/rlm-studio-guide.md`) have drifted out of sync with the product.

---

## 1. What actually changed (last 30 days)

Grouped by theme, with the commits that matter most:

| Theme | Representative commits | User-visible? |
|---|---|---|
| **Learn tab V1→V2→V2b** (PRs #20, #23, #24) | `cfcba81`, `e47836e`, `656ade4`, `bbd633c`, `f1fdb5d`, `e92c2b4`, `c86672c`, `1ba30b9` | Yes — new `/learn` surface: Concepts + Cookbook + Troubleshooting + scrubbable Replay walkthrough. Deep-link from any Traces row via "Replay in Learn". |
| **Cookbook host guides seeded** (inside the Learn work) | `3ddf631`, `7cdffd3`, `c9d482d`, `252d403`, `5d1c29c` | Yes — `docs/hosts/{anthropic,openai,ollama,lmstudio,vllm,dgx-spark}.md` plus `docs/troubleshoot.yaml`. DGX Spark guide was expanded with full vLLM memory-tuning playbook. |
| **LLM Tuner / Compare-matrix V2** | `4b25541`, `4617b2a`, `ed20084`, `2c59a91`, `20c0902`, `1e75686`, `cdbb4c2` | Yes — new `/compare` page (Provider × Mode grid) backed by `POST /api/chat/compare-matrix`. |
| **LLM-as-judge rubric v2.0** | `722140f`, `6162a35`, `a518786`, `669242c`, `94cf849`, `baf4f86`, `dacd44c`, `131de35` | Yes — pointwise + pairwise prompts, server-side `overall_score`, unjudged-slots sort below judged, non-usable outcomes auto-scored. |
| **Outcome classifier + failure metrics** | `245da80`, `baf4f86`, `3c218eb`, `5f85f49`, `01a6884` | Yes — `GET /api/metrics/failures/{session_id}`, failure charts in dashboard. |
| **Scheduled connection testing** | `ddf5339`, `a5f683b`, `5791483`, `d9e70d2`, `d1fc076`, `f9c8f0a` | Yes — global `connection_test_interval_minutes` setting, background daemon tests providers, new Settings UI. |
| **Conversation memory** | `c0b8508`, `c29a3eb`, `3c317ae`, `5130ac8`, `b877c34`, `6aca17a`, `2684627` | Yes — `conversation_memory_enabled` on ChatProvider, budgeted replay service, history bound as REPL variable in RLM sandbox, native chat messages for Direct/Compare/RAG. |
| **Trace bulk delete** | `ca38737`, `a403360` | Yes — Gmail-style single + bulk delete with confirm dialog, single-SQL bulk path. |
| **LLM-provider UX polish** | `64898f8`, `cabe94a`, `fe8ce7d` | Yes — Test Connection in edit form, logo in sidebar, fixed model-dropdown bug. |
| **Stability / correctness fixes** | `4bb13c4`, `c96366a`, `4d6f347`, `fbe64f0`, `36adcc8`, `b5d74b8` | Yes — stall breaker accepts plain-text after all files inspected; JSON parser tolerates trailing braces / prose / `<think>` blocks; 50MB+ PDF upload fix; Anthropic top_p/temperature guard; timeout warnings. |
| **Build/test hygiene** | `4554cb2`, `a09ca23`, `2c59a91`, `139b229`, `3912881`, `70576e4`, #19, #21 | Internal — pytest-xdist parallelism, CI hotfixes, dependabot bumps. |
| **Refactor: eliminate magic strings** | `459b3de`, `e80f8e4`, `040f7a6`, `fd8fe2a`, `0f45aa0`, `b992546`, `2b41cfc` | Internal — mode and trace-key literals centralized; lowers future drift. |

**Frontend surface now has 6 pages**, not 4: chat (`app/page.tsx`), plus
`compare`, `dashboard`, `learn`, `traces`, `settings` subdirectories under
`frontend/src/app/`.

---

## 2. Does the current doc set capture the product — in depth and clearly?

### 2.1 What's covered well

- **The RLM paradigm itself** — `docs/rlm-concepts.md` and
  `docs/RLMKit_Design_Document.md` are excellent: paper citation, core
  insight, loop algorithm, circuit breakers, tool API, JSON action protocol
  v2.0. If anything, these are the strongest docs in the repo.
- **Prompt tuning** — `docs/rlm-prompt-tuning.md` stays narrow and practical.
- **Per-host setup cookbook** — `docs/hosts/{ollama,vllm,lmstudio,openai,
  anthropic,dgx-spark}.md` each follow the same 6-section template
  (Install → Start → Model → Add to RLM Studio → Test → Common errors).
  DGX Spark is the flagship and genuinely thorough.
- **README.md** — recently expanded with "Where RLMKit Shines / Not a fit",
  which was a real gap. Now the front door tells you when *not* to use
  RLMKit, which is unusually honest and useful.
- **Troubleshoot catalog** — `docs/troubleshoot.yaml` is well-structured and
  rendered live by the Learn tab.

### 2.2 Gaps (significant — should be fixed)

**G1. `docs/rlm-studio-guide.md` still describes a 4-page app.**
The guide is the canonical "how to use RLM Studio" doc and it never mentions
the **Learn** tab or the **LLM Tuner / Compare** page — both of which
shipped in the last 30 days and are prominently in the sidebar today. The
README got a line about each; the guide did not.

**G2. No doc for LLM-as-judge.**
The rubric v2.0 (pointwise + pairwise), `overall_score` math, how to pick a
judge provider in Settings, how "auto-score non-usable outcomes" interacts
with user-visible metrics — none of it is documented outside commit
messages and internal specs. Users who see a `judge_score` column on
Traces or a sort order on Compare have no way to understand what it means.

**G3. No doc for the outcome classifier + failure dashboard.**
There's a new `GET /api/metrics/failures/{session_id}` endpoint and failure
charts, but the user guide's Dashboard section still lists only the
pre-classifier cards (Total Tokens, Total Cost, Avg Latency, Token
Savings). The "non-usable" vs "degraded" vs "complete" classes are not
named anywhere in public docs.

**G4. No doc for scheduled connection testing.**
The `connection_test_interval_minutes` setting ships with a full background
daemon (consecutive-failure threshold, per-test timeout, stale-result
guard, shutdown hook). `doc_internal/specs/scheduled-connection-testing.md`
exists but is not user-facing. The Studio guide's **Providers** section
doesn't mention it.

**G5. No doc for conversation memory.**
`conversation_memory_enabled` on ChatProviderConfig, the fact that RLM binds
`history` as a REPL variable (see system_prompt_v2_1), and the
budgeted-replay service are all user-relevant. The Studio guide only says
"Conversation history is maintained per Chat Provider" without
acknowledging that there's now a toggle to disable it, or that RLM/RAG see
history differently than Direct/Compare.

**G6. No doc for trace deletion.**
Single and bulk delete (Gmail-style) landed with a confirm dialog; the
guide's Traces section still says "Click any row to load its full trace"
with no mention of deletion affordances.

**G7. Stale model identifiers in `docs/rlm-studio-guide.md`.**
Seven occurrences of `claude-sonnet-4-5` remain; the rest of the codebase
standardized on `claude-sonnet-4-6` in v1.0.0 (`a91f9...` CHANGELOG entry
for Cycle 3).

**G8. Stale `.env.example`.**
References `claude-3-5-sonnet-20241022` and "Claude 3 Opus, Sonnet, Haiku",
neither of which matches current provider defaults.

**G9. `CLAUDE.md` / `AGENTS.md` describe a different product.**
Both are templated as "ML/AI — RAG + Knowledge Graph", list Neo4j + Cypher
as core stack, prescribe `src/rag/` and `src/graph/` subdirectories, and
define "Knowledge Graph Engineer" and "RAG Engineer" agent roles. RLMKit's
actual stack is RestrictedPython sandbox + LiteLLM + FastAPI; there is no
`src/graph/`, no Neo4j, no Cypher. Any agent that reads these files as
ground truth will plan the wrong work.

**G10. `CONTRIBUTING.md` inherits the same drift.**
Tells new contributors to `docker compose up -d` to start "Neo4j and
vector DB services", which do not exist in this project.

**G11. No deployment-topology doc.**
This is the gap your uploaded v2 cookbook actually addresses. Nowhere in
the repo docs does it say: *"Run the RLMKit backend + frontend on your dev
laptop, point the Chat Provider at an Ollama server running on a DGX Spark
at `http://<spark-ip>:11434`."* That pattern is the single most useful
thing in `DGX_Spark_RLMKit_Setup_Cookbook_v2.md` and it isn't in
`docs/hosts/dgx-spark.md` — the Spark guide jumps straight from "install
Ollama" to "Add to RLM Studio" without naming the topology.

**G12. No "index" for `docs/hosts/`.**
Six provider guides sit side-by-side with no landing page explaining the
choice between them. The Learn tab's Cookbook page solves this in the UI,
but a plain-markdown reader on GitHub sees a flat directory and has no
entry point.

### 2.3 Minor gaps

- The `Starting RLM Studio` block in README uses `uv run python -m
  rlmkit.server --reload`; the Studio guide uses `uv run uvicorn
  rlmkit.server.app:app --reload`; the uploaded v2 cookbook uses the
  uvicorn form. Pick one and propagate.
- `docker/README.md` documents a `DockerExecutor` import path
  (`rlmkit.envs.sandbox`) — verify against current `src/rlmkit/`
  layout; the architecture diagram in README.md shows
  `infrastructure/sandbox/` as the canonical location.
- No CHANGELOG entry yet for the LLM Tuner, LLM-as-judge, scheduled
  connection testing, or conversation memory features — they all landed
  under `[Unreleased]` but the entry there only covers the Learn tab.
  Consider a second bullet group for "Evaluation & Ops" before cutting
  the next release.

---

## 3. Uploaded DGX Spark cookbooks vs. the repo's DGX Spark guide

`docs/hosts/dgx-spark.md` is **already a superset** of
`DGX_Spark_Setup_Cookbook_v4b.md` in almost every respect. It keeps the
machine sanity-check, the Ollama install + systemd override, the full
vLLM build recipe, the memory-tuning playbook, the three failure patterns,
the practical model-sizing ranking, the explicit KV cache workaround, and
it adds an external-references section the uploads lack.

**What the uploads still add that the repo guide is missing:**

1. **Topology diagram (v2)** — "RLMKit on dev machine + Ollama on DGX
   Spark." One-line architecture description, plus the end-to-end health
   check: `curl http://<dgx-spark-ip>:11434/api/tags` *and* `curl
   http://localhost:8000/health | python3 -m json.tool`.
2. **Open WebUI as optional smoke-test client (v4b §"Open WebUI against
   host Ollama")** — handy when you want to confirm Ollama works from
   something other than RLM Studio.
3. **Hostname / network sanity (v4b §"Key DGX Spark checks")** —
   `hostnamectl`, `hostname -I`, `df -h`. The repo guide has most of
   these but not `hostname -I`, which is exactly what a user needs to
   fill in `<dgx-spark-ip>`.

Everything else in both uploads is already in `docs/hosts/dgx-spark.md`,
often in more depth (the external references, the vLLM Ollama-compatible
API caveat, the "base model on chat endpoint returns garbage" error, the
VPN/SSH tunnel security note).

**Recommendation:** treat the uploaded cookbooks as *superseded* by
`docs/hosts/dgx-spark.md`, then fold the three items above back into the
repo guide. Concretely:

- Add a new §2a "Deployment topology" before the install section,
  stating the dev-laptop + Spark-Ollama pattern, with the two-curl
  health check.
- Add `hostname -I` to §1.
- Append a new §4a "Optional: Open WebUI as a smoke-test client" with
  the docker run command from v4b.

After that, the uploads can be deleted (or archived under
`doc_internal/archive/`); nothing in them would be lost.

---

## 4. Proposed: consolidate hosting + "how to use an LLM with RLMKit"

You specifically asked about a single doc that covers both *how to host
an LLM* and *how to use it with RLMKit*. That consolidation is the right
move — right now the information is split across six files, the Studio
guide, the README, and `.env.example`, and each of them tells part of
the story.

### 4.1 Recommended shape

Introduce **`docs/hosts/README.md`** as the landing doc for the
`hosts/` subtree. Outline:

```
# Connecting an LLM to RLMKit

## 1. Pick a backend
   - Decision tree (cloud vs local vs GPU-host)
   - Links to the six provider-specific guides

## 2. Deployment topologies
   - All-local (laptop runs RLMKit + local LLM)
   - Laptop + remote GPU host (DGX Spark, workstation)
   - All-cloud (RLMKit locally, cloud API)
   - Self-hosted behind a VPN/SSH tunnel

## 3. What RLM Studio needs to talk to your LLM
   - Backend id (ollama / vllm / lmstudio / openai / anthropic)
   - Base URL (with defaults per backend)
   - Model id (how to find it per backend)
   - API key (cloud only; local backends are keyless — see security note)
   - Test Connection: what it checks, failure interpretations

## 4. Configuration surfaces
   - Settings → LLM Providers (runtime, GUI)
   - .env and env-var overrides (startup)
   - How both interact; precedence (real env > SecretStore > legacy .env)

## 5. Security & network boundaries
   - Secret storage (OS keyring / ~/.rlmkit/api_keys.json chmod 600)
   - Why vLLM's --api-key isn't exposed in Settings
   - SSH tunnel vs VPN vs 127.0.0.1-only for local backends

## 6. Operational hygiene
   - Scheduled connection testing (new — link to setting)
   - Watching cost and outcome classification
   - When connections go stale

## 7. Go deeper — per-provider guides
   - Links out to anthropic.md, openai.md, ollama.md, lmstudio.md,
     vllm.md, dgx-spark.md
```

### 4.2 What moves, what stays

| Content | Currently in | Goes to |
|---|---|---|
| Decision tree between backends | nowhere (implicit in README) | §1 of new `hosts/README.md` |
| Topology ("RLMKit locally + Ollama on Spark") | uploaded v2 cookbook only | §2 of new `hosts/README.md` **and** §2a of `dgx-spark.md` |
| Backend / Base URL / Model / API key table | repeated in every host guide | Consolidated once in §3, then *referenced* (not duplicated) in host guides |
| SecretStore / keyring note | `CHANGELOG.md` v1.0.0 only | §5 of new `hosts/README.md` |
| vLLM `--api-key` caveat | `vllm.md`, `dgx-spark.md` | §5 of new `hosts/README.md` (shared) |
| Scheduled connection testing | internal spec only | §6 of new `hosts/README.md` **and** new section in `rlm-studio-guide.md` |
| `.env.example` env-var set | `.env.example` (partially stale) | Refreshed + linked from §4 |

### 4.3 What stays per-provider

Each of the six `hosts/*.md` guides keeps its tight 6-section structure
(Install → Start → Model → Add to RLM Studio → Test → Common errors).
The redundant "Add to RLM Studio" tables become a one-liner each that
says "see hosts/README.md §3 for the shape; specific values for this
backend below" — this keeps the guides scannable without copy-pasting
the same shape six times.

### 4.4 What this does *not* do

- It does not replace `rlm-studio-guide.md`. That doc stays focused on
  *using* the Studio surfaces (Chat, Compare, Dashboard, Traces, Learn,
  Settings). The hosts subtree stays focused on *connecting* an LLM.
- It does not merge the two uploaded cookbooks into a single mega-doc.
  The right move is to delete them once their three useful deltas are
  folded into `dgx-spark.md` (§3 above).

---

## 5. Concrete recommendations, ranked

Ordered by impact-per-effort:

**P0 — land with the next release notes, ~2 hours each**
1. Add a **Learn** section to `docs/rlm-studio-guide.md` (mirroring the
   existing Chat/Dashboard/Traces/Settings treatment): Concepts,
   Cookbook, Troubleshooting, Replay walkthrough, deep-link from Traces.
2. Add a **Compare / LLM Tuner** section to the same guide: Provider ×
   Mode grid, ranking metrics, ephemeral Chat Providers, expected
   latency (synchronous endpoint).
3. Refresh model identifiers in `docs/rlm-studio-guide.md`
   (`claude-sonnet-4-5` → `claude-sonnet-4-6`) and in `.env.example`
   (`claude-3-5-sonnet-20241022` → `claude-sonnet-4-6`).
4. Fold the Neo4j / Cypher / `src/graph/` scaffolding out of `CLAUDE.md`
   and `AGENTS.md`, and out of `CONTRIBUTING.md`'s "Docker (for Neo4j
   and vector DB services)" line. Replace with the real stack:
   RestrictedPython sandbox, LiteLLM, FastAPI, Next.js. This one
   directly affects any AI agent that reads the repo.

**P1 — do before v1.1.0, ~half a day each**
5. Write `docs/hosts/README.md` per §4.1 above.
6. Extend `CHANGELOG.md [Unreleased]` with a second "Evaluation & Ops"
   bullet group covering LLM Tuner, LLM-as-judge rubric v2.0, outcome
   classifier, failure metrics, scheduled connection testing,
   conversation memory toggle, trace deletion.
7. Add a short **LLM-as-judge** doc (either `docs/rlm-judge.md` or a
   subsection in the Studio guide): rubric v2.0 anchors, pointwise vs
   pairwise, picking a judge provider, auto-scoring non-usable
   outcomes, how `overall_score` is computed.
8. Update the DGX Spark guide per §3 above (topology, `hostname -I`,
   optional Open WebUI smoke-test).

**P2 — nice to have, ~1 hour each**
9. Retire the two uploaded cookbooks once §3 items are folded in.
   Move them under `doc_internal/archive/` with a README note.
10. Standardize the "start the backend" command across README, Studio
    guide, and any remaining references to avoid the `uvicorn`
    vs `python -m rlmkit.server` drift.
11. Add a one-line index comment at the top of `docs/troubleshoot.yaml`
    noting it's also rendered by the Learn tab (already says this, good
    — but add a backlink from `rlm-studio-guide.md`'s Learn section).

---

## 6. Sources

- Repo at `/Users/gosha/dev/repo/rlmkit` (commit scan + doc inventory)
- `/Users/gosha/dev/repo/rlmkit/CHANGELOG.md` (v1.0.0 + Unreleased)
- `/Users/gosha/dev/repo/rlmkit/docs/hosts/dgx-spark.md`
- `/Users/gosha/dev/repo/rlmkit/docs/rlm-studio-guide.md`
- Uploaded: `DGX_Spark_RLMKit_Setup_Cookbook_v2.md`
- Uploaded: `DGX_Spark_Setup_Cookbook_v4b.md`
- [gosha70/rlmkit on GitHub](https://github.com/gosha70/rlmkit)
