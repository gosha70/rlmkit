---
feature_id: dgx-spark-docs-refresh
spec_mode: full
status: draft
origin:
  urls:
    - file:///Users/gosha/Downloads/files002/pitch-rlmkit-spark-docs.md
  transcripts:
    - "Team-lead delegation message, 2026-07-27: 'Read the pitch at /Users/gosha/Downloads/files002/pitch-rlmkit-spark-docs.md IN FULL, then read every file it touches, then produce SDD artifacts that are ready to hand to the build agent. Feature id: dgx-spark-docs-refresh.'"
  origin_claim: |
    "docs/hosts/dgx-spark.md contains model and sizing guidance written before
    NVFP4 quantization was working on Spark's sm_121a. Its 'Practical ranking'
    (§7) caps at '7B–14B fits comfortably / 30B–32B with tuning / 70B not
    recommended' — which now directly contradicts our own
    docs/hosts/dgx-spark-vllm.md, a verified operator manual that runs a ~44 GB
    NVFP4 model with a 128K context on the same hardware. §3 also pulls
    llama3.2 / gpt-oss:20b / qwen2.5:14b, all two generations stale.
    Separately, there are newer candidate models reported to run on a single
    128 GB Spark that we want documented — but they have not been tested on
    our hardware."
spec_mode_justification: >
  Docs-only by file type, which the spec-workflow skill would normally classify
  as `none`. Escalated to `full` because the change touches more than two files
  (two host docs + an executable start script + optionally CHANGELOG), modifies
  an executable artifact whose behaviour must stay byte-identical, and edits a
  document whose entire value is a hardware-verification claim that this change
  must not weaken.
date: 2026-07-27
---

# Spec — DGX Spark model-guidance refresh

## 1. Problem

Two docs in `docs/hosts/` now contradict each other about what fits on a
128 GB DGX Spark:

- `docs/hosts/dgx-spark.md` §7 ("Common errors" → "**Practical ranking for
  this Spark setup:**", lines 272–277) caps guidance at *7B–14B fits
  comfortably / some 30B–32B with tuning / 70B–72B not recommended*. §3
  (lines 88–90) pulls `llama3.2`, `gpt-oss:20b`, `qwen2.5:14b`, and its
  trailing prose (lines 94–95) claims a 14B–20B GGUF ceiling.
- `docs/hosts/dgx-spark-vllm.md` is a hardware-verified operator manual that
  runs `RedHatAI/Qwen3-Coder-Next-NVFP4` (~44 GB NVFP4 weights) at
  `--max-model-len 131072`, `--gpu-memory-utilization 0.72` on the same box.

The ranking is expressed in **parameter count**; the real constraint is
**weight footprint in GB**. A reader who trusts §7 will not attempt the
configuration the repo already proves works.

Separately, three newer models are reported by third parties to run on a
single 128 GB Spark. We want them documented without pretending they were
tested here.

## 2. User scenarios

1. **Operator sizing a model.** Reads `dgx-spark.md` §7, gets a GB-based
   band table, and is pointed at `dgx-spark-vllm.md` as the verified example
   in the first band — instead of being told 30B is the ceiling.
2. **Operator evaluating a candidate model.** Reads `dgx-spark-vllm.md` §9,
   sees clearly which parts of the verified §3 config carry over unchanged,
   which must be re-verified per model, and what to measure to fill in the
   placeholders. Nothing in §9 reads as if it had been booted.
3. **Operator serving a non-Qwen3-Coder model.** Runs
   `scripts/dgx-spark/vllm/start-qwen3-coder-next.sh` with
   `VLLM_TOOL_CALL_PARSER=<other>` and gets the right parser, without the
   default behaviour changing for anyone else.
4. **Reviewer of the PR.** Reads a PR description split into "staleness
   fixes, safe to merge" and "unverified guidance, needs a hardware run", and
   can merge the first half without endorsing the second.

## 3. Requirements

### FR-1 — `docs/hosts/dgx-spark.md` §3 de-stale
Replace the `ollama pull` list with current-generation models
(`qwen3.6:35b-a3b`, `qwen3.5:27b`) plus one tiny model kept for smoke tests.
Fix the trailing prose that asserts a 14B–20B GGUF ceiling: the constraint is
weight footprint, not parameter count.

### FR-2 — `docs/hosts/dgx-spark.md` §7 "Practical ranking" rewritten in GB
Three bands keyed on **weights in GB**:
- ≤ 45 GB — comfortable. Cross-reference `dgx-spark-vllm.md` as the verified
  example in this band.
- 45–90 GB — possible with tuning and a reduced context.
- \> 100 GB — needs multi-node.

The band table must state explicitly that MoE sparsity (`A10B` = 10 B active)
reduces **compute**, not **KV cache** — KV is sized by total layers × KV
heads × context. This misconception is the likely reason someone would
over-provision context on a 122B MoE model.

### FR-3 — `docs/hosts/dgx-spark.md` §7 failure pattern #2 made measurable
Failure pattern #2 ("Model loads, then no KV cache room left", lines
263–266) currently cites `Qwen/Qwen3-32B` at 0.4, which understates the
problem for large-weight MoE models. Add that the `GPU KV cache size:` line
in the vLLM boot log reports the token budget directly, so the fix is
measurable rather than trial-and-error.

### FR-4 — `docs/hosts/dgx-spark-vllm.md` gains a quarantined §9
A new `## 9. Candidate models (UNVERIFIED — 2026-07-27)` section containing:
1. An explicit statement that §3 is verified only for Qwen3-Coder-Next and
   that everything in §9 is a hypothesis.
2. **Carries over unchanged** — Blockers #1, #3, #4, plus `--enforce-eager`,
   the `pkill` / `drop_caches` preamble, and `--trust-remote-code`, with a
   per-item reason (client-side, transport-level, or toolchain-level — hence
   model-independent). Note that Blocker #1 gets *worse* with larger weights;
   suggest `MAX_JOBS=1` for a ~75 GB first boot.
3. **Must be re-verified per model** — a table covering at least
   `--tool-call-parser` (`qwen3_coder` is specific to Qwen3-Coder-Next's XML
   emission), `--gpu-memory-utilization`, `--max-model-len`, and Blocker #2's
   "omit `--reasoning-parser`" guidance, **which likely inverts** for
   Qwen3.5/3.6 (they have thinking modes; tool-calling reliability is
   reported to depend on thinking being enabled). Each row names the §6 step
   that tests it.
4. Starting-point commands with placeholder values, each flagged as a guess,
   each delta from §3 justified in prose.
5. Promotion criteria: §6 steps 1–3 all PASS, plus an `aider-polyglot
   python/bowling` n=3 result recorded alongside the existing anchor
   benchmark, before anything moves from §9 into §3/§5.

### FR-5 — `docs/hosts/dgx-spark-vllm.md` §5 caveat
Add a note (no existing row values changed) that the table's rows assume
~44 GB of weights, that each additional GB of weights costs roughly 1 GB of
KV budget, and that a ~75 GB model therefore subtracts ~31 GB at every row —
enough to put the `131072` row out of reach. Point at §9 for re-deriving per
model.

### FR-6 — `start-qwen3-coder-next.sh` parameterized
1. Add `VLLM_TOOL_CALL_PARSER` (default `qwen3_coder`) and
   `VLLM_MAX_NUM_BATCHED_TOKENS` (default `8192`).
2. Add `VLLM_REASONING_PARSER`, defaulting to **empty/absent**, so today's
   verified behaviour (no `--reasoning-parser` flag at all) is unchanged.
3. `verify-tool-calls.sh` model name: see DIVERGENT D-1 — already an env var;
   verify only.
4. Do **not** rename the script (see §4 Constraints).

### FR-7 — `docs/hosts/dgx-spark-vllm.md` §8 corollary
Add a corollary narrowing the known limitation: the three hardcoded `openai/`
call sites block **RLM Studio**, not Claude Code served via vLLM's native
Anthropic-compatible endpoint (`--enable-anthropic-api`), which removes
LiteLLM from the path entirely. Flag that `--enable-anthropic-api` is
version-dependent and should be verified against the local vLLM build. Note
that this may lower the priority of the tracked follow-up — do not act on it.

### FR-8 — PR description
Produce a PR description that separates "staleness fixes, safe to merge"
from "unverified guidance, needs a hardware run".

## 4. Constraints / what NOT to build

**C-1 (highest).** `docs/hosts/dgx-spark-vllm.md` §3, §5, and §7 are
verified-on-hardware content, backed by the line-3 claim *"Verified
2026-05-17 on DGX Spark with vLLM 0.6+ and flashinfer 0.6.6, sm_121a
target. … Every flag in §3 is load-bearing."* **Do not modify any existing
value in those sections.** Additive, clearly-marked notes are permitted where
a requirement calls for one (FR-5 in §5, FR-6's env-var list in the §3
wrapper sentence at line 62). All new *configuration* goes into §9.

**C-2.** No fabricated measured values, benchmark numbers, dates, or
"verified" claims. Placeholders must look like placeholders — use
`<TBD-measure>`. If you find yourself writing a specific
`--gpu-memory-utilization` or `--max-model-len` for a model nobody has
booted, stop and write the placeholder plus the command that would measure
it.

**C-3.** Preserve verbatim error strings. This binds in **both** docs:
`dgx-spark-vllm.md` §7 (the four-row troubleshooting matrix, lines 145–150)
and `dgx-spark.md` §7 ("Common errors", lines 233–322, e.g.
`externally-managed-environment`, `numa.h: No such file or directory`,
`NVFP4 / .e2m1x2 fails for sm_121`). Operators grep for exact text.

**C-4.** Minimal diffs. No reflowing, reformatting, or link-style changes to
lines not otherwise being touched. `dgx-spark-vllm.md` is written with long
unwrapped lines; `dgx-spark.md` is hard-wrapped at ~70 columns. Match the
local style of the file you are editing.

**C-5.** `start-qwen3-coder-next.sh` behaviour must be byte-identical when no
new env var is set. Scope of "byte-identical": the exported environment and
the assembled `python -m vllm.entrypoints.openai.api_server` argv. The
informational banner gains two lines for the new knobs (see Open question
OQ-3 in plan.md). Verified with `bash -n` plus an argv-diff dry run.

**C-6.** Out of scope: `src/`, `frontend/`, `tests/`. The three hardcoded
`openai/` → `hosted_vllm/` call sites are a known separate follow-up — do
**not** fix them. FR-7 documents a corollary about them and nothing more.

**C-7.** No `git push`. Commit to the feature branch and stop.

**C-8.** Do not add new models, flags, or numbers to `docs/troubleshoot.yaml`
or any file outside the scope list in plan.md §2, even where the same stale
guidance appears there (see Discovered fact DF-1).

## 5. Key entities (files and anchors)

| Entity | Path | Anchor at HEAD |
|---|---|---|
| Generic Spark guide | `docs/hosts/dgx-spark.md` | §3 L85–95; §7 L233–322; ranking L272–277; failure #2 L263–266 |
| Verified operator manual | `docs/hosts/dgx-spark-vllm.md` | claim L3; §3 L20–62; §5 L72–85; §6 L87–139; §7 L141–150; §8 L152–200; References L202–207 |
| Start wrapper | `scripts/dgx-spark/vllm/start-qwen3-coder-next.sh` | header L13–20; defaults L28–33; banner L64–70; exec L72–83 |
| Smoke test | `scripts/dgx-spark/vllm/verify-tool-calls.sh` | `VLLM_MODEL` L26; usage L17–19 |
| Changelog | `CHANGELOG.md` | `## [Unreleased]` L6 — see OQ-1 |

## 6. Success criteria (acceptance criteria, from the pitch)

**AC-1.** `docs/hosts/dgx-spark.md` no longer contradicts `dgx-spark-vllm.md`
on sizing. Grep both for parameter-count-based ceilings; there should be none
left that aren't explicitly qualified as BF16-only.

**AC-2.** Every new configuration value is either inside §9, or marked as a
placeholder, or both.

**AC-3.** `start-qwen3-coder-next.sh` runs identically with no env vars set,
and can serve a model needing a different tool-call parser.

**AC-4.** Markdown links resolve — check the relative paths between
`docs/hosts/*` and `scripts/dgx-spark/vllm/*`.

**AC-5.** A PR description that separates "staleness fixes, safe to merge"
from "unverified guidance, needs a hardware run" so the reviewer can merge
the first half without endorsing the second.

## 7. Pitch claims verified

The pitch was written from a partial read of the repo. Every factual claim it
makes about file contents was checked against HEAD
(`e278ca2`, branch `docs/dgx-spark-vllm-qwen3-coder`).

### 7.1 CONFIRMED

| # | Claim | Evidence |
|---|---|---|
| C-a | `dgx-spark.md` §3 pulls `llama3.2`, `gpt-oss:20b`, `qwen2.5:14b` | `docs/hosts/dgx-spark.md:88-90` |
| C-b | §3 trailing prose asserts a 14B–20B GGUF ceiling | `dgx-spark.md:94-95` — "quantized models (GGUF) fit comfortably up to the 14B–20B class" |
| C-c | §7 "Practical ranking" caps at 7B–14B / 30B–32B / 70B not recommended | `dgx-spark.md:272-277` |
| C-d | §7 failure pattern #2 uses a `Qwen/Qwen3-32B` example | `dgx-spark.md:263-266` — "Seen with `Qwen/Qwen3-32B` at 0.4. Weights fit, cache does not." |
| C-e | `dgx-spark-vllm.md` opens with the quoted verification claim | `dgx-spark-vllm.md:3` — quoted text matches verbatim, including "Every flag in §3 is load-bearing" |
| C-f | The verified model is ~44 GB NVFP4 at 128K context | `dgx-spark-vllm.md:74` (~44 GB weights) and `:82` (131072 / 0.72 row marked **Verified**) |
| C-g | `--gpu-memory-utilization 0.72` is the verified value | `dgx-spark-vllm.md:41`, `:57` |
| C-h | §3 hardcodes `--tool-call-parser qwen3_coder` and Blocker #2 says to omit `--reasoning-parser` | `dgx-spark-vllm.md:58` and `:148` |
| C-i | The start script is documented as overridable via the six named env vars | script `:13-20`; doc `dgx-spark-vllm.md:62` — both lists match exactly |
| C-j | The start script's tool-call parser is **not** overridable | `start-qwen3-coder-next.sh:81` — literal `--tool-call-parser qwen3_coder` |
| C-k | `--max-num-batched-tokens 8192` is hardcoded in the script | `start-qwen3-coder-next.sh:78` |
| C-l | §8 names three hardcoded `openai/` call sites in `api.py`, `server/routes/providers.py`, `server/dependencies.py` | doc `:188-192`; verified in source: `src/rlmkit/api.py:191`, `src/rlmkit/server/routes/providers.py:38`, `src/rlmkit/server/dependencies.py:1340` — all three read `"vllm": "openai/"` |
| C-m | §6 has exactly three verification steps | `dgx-spark-vllm.md:91`, `:103`, `:121` |
| C-n | §3 links the start script by relative path from `docs/hosts/` | `dgx-spark-vllm.md:62` → `../../scripts/dgx-spark/vllm/start-qwen3-coder-next.sh` (resolves) |
| C-o | Blockers #1/#3/#4 are toolchain-, client-envelope-, and transport-level respectively | `dgx-spark-vllm.md:147` (flashinfer JIT nvcc concurrency), `:149` (agent request envelope), `:150` (LiteLLM route selection) |

### 7.2 DIVERGENT

**D-1 — Task D.3 is already done.** The pitch says to "check
`verify-tool-calls.sh` for a hardcoded model name; if present, make it an
argument or env var". It is *already* an env var:
`verify-tool-calls.sh:26` reads
`VLLM_MODEL="${VLLM_MODEL:-RedHatAI/Qwen3-Coder-Next-NVFP4}"`, documented in
the usage block at `:19`. **Plan does instead:** verify only, change nothing
in that file. Recorded as a no-op task so the build agent does not invent a
change.

**D-2 — "Append §9" would land after an unnumbered section.**
`dgx-spark-vllm.md` ends with `## References` (L202–207), which follows §8.
Literally appending §9 to the file would place it *after* References.
**Plan does instead:** insert §9 between the end of §8 (L200) and
`## References` (L202), preserving the numbered-sections-then-references
structure.

**D-3 — There is no "anchor table".** Task B.5 says to add an
`aider-polyglot python/bowling` n=3 result "to the anchor table". The anchor
benchmark is a single bullet in `## References` (`dgx-spark-vllm.md:207`),
not a table. **Plan does instead:** promotion criteria say the new n=3 result
must be recorded alongside the existing anchor-benchmark bullet in
References; no table is created, and the References bullet itself is not
edited by this change (there is no measured result yet — C-2).

**D-4 — "§7" refers to two different sections in two different files.** The
hard rule "preserve the verbatim error strings in the §7 troubleshooting
matrix" describes `dgx-spark-vllm.md` §7, which *is* titled "Troubleshooting
matrix" and *is* a four-row table (L141–150). But Tasks A.2 and A.3 target
"§7 Practical ranking" and "§7 failure pattern #2", which live in
`dgx-spark.md` §7, titled "**Common errors**" — prose bullets, not a matrix.
**Plan does instead:** treats the preservation rule as binding on the
verbatim strings in *both* §7s (constraint C-3), and routes Task A edits only
to `dgx-spark.md` §7.

**D-5 — the §8 sentence the pitch quotes carries a qualifier the pitch
drops.** The pitch renders §8 as "an operator who points the built-in vllm
provider at this Spark setup will still hit Blocker #4". The actual sentence
(`dgx-spark-vllm.md:194`) is: "An operator who points the built-in `vllm`
provider at this Spark setup will therefore still hit Blocker #4 **once
LiteLLM is ≥ 1.50**, even though every flag in §3 is correct." **Plan does
instead:** the FR-7 corollary keeps the LiteLLM-version qualifier intact and
does not restate the sentence in the pitch's shortened form.

**D-6 — the three Ollama models in §3 are not all "two generations stale" in
the same way.** `gpt-oss:20b` is still the model
`scripts/dgx-spark/README.md:19` tells operators to pull in its quick-start.
**Plan does instead:** FR-1 replaces the §3 list as instructed, and flags in
plan.md (risk R-4) that `scripts/dgx-spark/README.md` will then name a model
the host doc no longer lists. That file is outside the pitch's scope list, so
the plan leaves it alone and records the inconsistency as a follow-up rather
than widening the diff.

### 7.3 TAKEN ON TRUST (unverifiable from the repo)

| # | Claim | Why it cannot be checked here |
|---|---|---|
| T-a | `RedHatAI/Qwen3.5-122B-A10B-NVFP4` ≈ 75 GB weights (NVIDIA developer forum thread) | Third-party hardware report; no local artifact |
| T-b | Qwen3.6-35B-A3B (FP8) ≈ 35 GB, ~68 tok/s on Spark (community blog) | Third-party benchmark; must be reproduced before it is a repo claim |
| T-c | Qwen3.5-27B-FP8 ≈ 31 GB (community blog) | Third-party report |
| T-d | Ollama registry tags `qwen3.6:35b-a3b` and `qwen3.5:27b` exist and resolve | Requires network access to the Ollama registry |
| T-e | vLLM exposes `--enable-anthropic-api`, and its availability is version-dependent | Flag is not referenced anywhere in this repo; only the vLLM Claude Code integration doc is linked (`dgx-spark-vllm.md:205`) |
| T-f | Qwen3.5/3.6 have thinking modes (unlike Qwen3-Coder-Next) and their tool-calling reliability depends on thinking being enabled — i.e. Blocker #2's guidance inverts | Model-behaviour claim; nothing in the repo tests it |
| T-g | The vLLM boot log emits a `GPU KV cache size:` line reporting the token budget | The string appears nowhere in the repo (grepped). Wording varies by vLLM version — FR-3 must hedge, not quote it as verbatim-verified |
| T-h | `qwen3_coder` is specific to Qwen3-Coder-Next's XML emission and will not serve other families | The repo confirms the parser is required *for this model* (`dgx-spark-vllm.md:58`); the negative claim about other models is external |
| T-i | MoE `A10B` denotes 10 B active parameters, and KV cache is sized by total layers × KV heads × context (unaffected by sparsity) | Architecture fact, correct as stated, but not derivable from repo contents |
| T-j | "Each additional GB of weights costs roughly 1 GB of KV budget" | Follows arithmetically from a fixed unified-memory pool, but is not a measured result — FR-5 must present it as a derivation, not a measurement. The 75 − 44 = ~31 GB figure is arithmetic on two numbers the docs already contain |

### 7.4 Discovered — not claimed by the pitch

**DF-1.** `docs/troubleshoot.yaml:75` ("Lower --gpu-memory-utilization to 0.4
for 7B-class models, 0.7 for 32B-class") and `:88` ("try 0.7 for 32B-class
models") repeat the same parameter-count framing AC-1 targets. That file is
outside the pitch's scope list and is schema-validated by
`tests/e2e/test_docs.py`. Left untouched; recorded as a follow-up.

**DF-2.** `docs/hosts/dgx-spark.md:256` ("Start at 0.4 for 7B-class models,
0.7 for 32B-class") is a parameter-count-keyed *recommendation* inside the
§7 text AC-1 greps. It is not a ceiling, but it will show up in the AC-1
grep. Plan FR-2 adds a BF16/unquantized qualifier in place rather than
rewriting it.

**DF-3.** `docs/hosts/dgx-spark.md:267-270` (failure pattern #3, "70B-class
is too large for Spark") is **already** BF16-qualified — "In single-GPU BF16
without quantization". It satisfies AC-1 as written; do not touch it.

**DF-4.** `scripts/dgx-spark/vllm/README.md` documents five scripts and does
**not** mention `start-qwen3-coder-next.sh` or `verify-tool-calls.sh` at all.
Pre-existing gap; the pitch's scope line says `scripts/dgx-spark/vllm/*.sh`.
Left untouched; recorded as a follow-up (see plan.md OQ-2).

**DF-5.** `docs/hosts/dgx-spark.md` is served to the RLM Studio Learn tab as
cookbook slug `hosts-dgx-spark`
(`src/rlmkit/server/routes/docs.py:56`); `docs/hosts/dgx-spark-vllm.md` is
**not** in the allowlist. Edits to `dgx-spark.md` therefore change
user-visible frontend content. `tests/e2e/test_docs.py` asserts only that the
file exists and is non-empty, so no test breaks — but the Learn-tab renderer
resolves no relative links, which is why AC-4 is scoped to repo-relative
resolution.

**DF-6.** The script hardcodes more than the two flags the pitch names:
`--enforce-eager`, `--max-num-seqs 1`, `--enable-auto-tool-choice`,
`--trust-remote-code`, `--host 0.0.0.0`, and the
`VLLM_USE_FLASHINFER_MOE_FP4` / `MAX_JOBS` / `NINJA_JOBS` exports. Deliberately
left hardcoded — FR-6 adds exactly the three variables named, no more.

**DF-7.** CHANGELOG convention: the most recent docs-only commit (`f336ffc`,
"docs(hosts): add Qwen3-Coder-Next on DGX Spark + vLLM operator manual",
which added `dgx-spark-vllm.md` and both scripts) left no CHANGELOG entry —
grepping `CHANGELOG.md` for `spark|dgx|qwen3-coder|operator manual` returns
only line 48–51, which is a docs paragraph delivered as part of feature
Phase 6. Documented default: no CHANGELOG entry (plan.md OQ-1).
