---
feature_id: dgx-spark-docs-refresh
spec: ./spec.md
plan: ./plan.md
status: draft
date: 2026-07-28
---

# Tasks — DGX Spark model-guidance refresh

Execute **one item at a time, in order**. Each task names the file it may
touch; a task must not modify anything outside that file. The allowlist in
`plan.md` §2 is binding — a path not on it is out of scope, always.

Line numbers are against HEAD (`e278ca2`) and shift as edits land; re-anchor
by searching for the quoted text, not by trusting the number.

Drafts for every prose change are in `plan.md` §3. Open questions OQ-1/2/3
already have documented defaults in `plan.md` §5 — apply the default and
note it; do not stop to ask.

---

## Phase 1 — confirmations (no edits)

### T1 — Confirm `verify-tool-calls.sh` needs no change
- **Action:** verify the model name is already an env var; change nothing.
- **File:** `scripts/dgx-spark/vllm/verify-tool-calls.sh` (read-only)
- **FR:** FR-6.3 (spec DIVERGENT D-1)
- **Done when:** `rg -n 'VLLM_MODEL' scripts/dgx-spark/vllm/verify-tool-calls.sh`
  shows `VLLM_MODEL="${VLLM_MODEL:-RedHatAI/Qwen3-Coder-Next-NVFP4}"` at L26
  and the usage line at L19, and the finding is recorded in the build notes.
  **This task's deliverable is a confirmation, not a diff.** If you find
  yourself editing this file, stop — the pitch's Task D.3 is already done.

### T2 — Capture the AC-1 baseline
- **Action:** run the two AC-1 greps from `plan.md` §6 against HEAD and save
  the output, so the post-change sweep has something to compare to.
- **File:** none (scratchpad output only)
- **FR:** AC-1 support
- **Done when:** baseline output saved; the current count (16 hits in
  `dgx-spark.md`, 0 in `dgx-spark-vllm.md` for the first grep) is recorded.

---

## Phase 2 — staleness fixes, `docs/hosts/dgx-spark.md` (safe half)

File is hard-wrapped at ~70 columns. Match it. Do not reflow untouched lines.

### T3 — Replace the §3 `ollama pull` list and its trailing prose
- **Action:** swap `gpt-oss:20b` / `qwen2.5:14b` for `qwen3.6:35b-a3b` /
  `qwen3.5:27b`; keep `llama3.2` as the tiny smoke-test pull; replace the
  14B–20B GGUF ceiling sentence with the weight-footprint framing.
- **File:** `docs/hosts/dgx-spark.md` (L87–92 fence, L94–95 prose)
- **FR:** FR-1
- **Done when:** `rg -n 'fit comfortably up to' docs/hosts/dgx-spark.md`
  returns nothing; no throughput, size, or "verified" claim is attached to
  either new tag (C-2); wrap width matches neighbours.

### T4 — Fix the §5 RLM Studio Ollama example models
- **Action:** one line — replace `qwen2.5:14b` with `qwen3.5:27b` in the
  Ollama path table so it stops naming a model §3 no longer pulls.
- **File:** `docs/hosts/dgx-spark.md` (L211)
- **FR:** FR-1 corollary (plan R-5)
- **Done when:** the table's column padding is unchanged and no other row
  is touched.

### T5 — Qualify the `--gpu-memory-utilization` starting points as BF16
- **Action:** append the BF16/unquantized qualifier in place; do not rewrite
  the sentence.
- **File:** `docs/hosts/dgx-spark.md` (L256)
- **FR:** FR-2 (spec DF-2)
- **Done when:** the original clause "Start at 0.4 for 7B-class models, 0.7
  for 32B-class" is still present verbatim, now followed by the qualifier.

### T6 — Make §7 failure pattern #2 measurable
- **Action:** append to the bullet — the boot log reports the KV-cache token
  budget, so this is measurable rather than trial-and-error; note that
  large-weight MoE models understate it relative to the `Qwen3-32B` example.
- **File:** `docs/hosts/dgx-spark.md` (L264–266)
- **FR:** FR-3
- **Done when:** the three existing sentences are byte-identical (C-3); the
  boot-log line is **described, not quoted** as verbatim-verified (spec T-g
  — the literal string does not exist anywhere in this repo); failure
  pattern #3 (L267–270) is untouched (DF-3).

### T7 — Rewrite the §7 "Practical ranking" in GB
- **Action:** replace the ranking block with the three GB-keyed bands, the
  verified cross-reference to `dgx-spark-vllm.md` in band 1, the
  90–100 GB "untested" clause, and the MoE compute-vs-KV-cache paragraph.
- **File:** `docs/hosts/dgx-spark.md` (L272–277)
- **FR:** FR-2
- **Done when:** the second AC-1 grep returns nothing; the band table states
  explicitly that `A10B` sparsity reduces compute but not KV cache; the link
  text names the file so it stays readable in the Learn tab (plan R-9); the
  only numbers used are 45 / 90 / 100 (from the spec) and ~44 GB / 131072 /
  0.72 (from `dgx-spark-vllm.md`).

### T8 — Commit 1
- **Action:** `git status` + `git diff`, show them, get explicit approval,
  then commit as
  `docs(hosts): size DGX Spark model guidance by weight footprint`.
- **File:** `docs/hosts/dgx-spark.md` only
- **FR:** AC-5 support
- **Done when:** `git show --stat` lists exactly one file; no push (C-7);
  approval for this commit is **not** carried forward to T12 or T17.

---

## Phase 3 — script parameterization (safe half)

### T9 — Add the three env vars to `start-qwen3-coder-next.sh`
- **Action:** header comment lines (after L20), the three `:-` defaults
  (after L33), the two banner lines (OQ-3 default = yes), and the
  `VLLM_ARGS` array replacing the inline `exec` argument list, with
  `--reasoning-parser` appended only when non-empty.
- **File:** `scripts/dgx-spark/vllm/start-qwen3-coder-next.sh`
- **FR:** FR-6.1, FR-6.2, FR-6.4
- **Done when:** exactly three new variables exist (`VLLM_TOOL_CALL_PARSER`,
  `VLLM_MAX_NUM_BATCHED_TOKENS`, `VLLM_REASONING_PARSER`) — `MAX_JOBS`,
  `NINJA_JOBS`, `--max-num-seqs`, `--enforce-eager`, `--trust-remote-code`
  and the host binding stay hardcoded (DF-6); the script is not renamed;
  the existing six header lines are not realigned (C-4).

### T10 — Prove the script is unchanged by default and overridable
- **Action:** run `bash -n`, `shellcheck` if available, and the full
  before/after argv + exported-env dry-run recipe from `plan.md` §6 (AC-3),
  including the override run.
- **File:** none (harness lives in the scratchpad)
- **FR:** FR-6 / AC-3 / C-5
- **Done when:** the default-path `diff` is empty; the override run shows
  `--tool-call-parser hermes`, `--reasoning-parser qwen3` and
  `--max-num-batched-tokens 4096` in the captured argv; the default run
  contains no `reasoning-parser` token at all. Paste the captured output
  into the build notes — do not assert this from reading the code.

### T11 — Document the new knobs in `dgx-spark-vllm.md` §3
- **Action:** append the three names to the env-var parenthetical at L62,
  with the one clause explaining that `VLLM_REASONING_PARSER` defaults to
  unset.
- **File:** `docs/hosts/dgx-spark-vllm.md` (L62 only)
- **FR:** FR-6 (permitted by C-1)
- **Done when:** the existing six names are in their original order, no
  value anywhere in §3 changed, and `git diff -U0` shows a single hunk in
  this file.

### T12 — Commit 2
- **Action:** show `git status` + `git diff`, get explicit approval, commit
  as `feat(scripts): parameterize tool-call/reasoning parser in the Spark vLLM starter`.
- **Files:** the start script + `docs/hosts/dgx-spark-vllm.md`
- **FR:** AC-5 support
- **Done when:** two files in `git show --stat`; no push.

---

## Phase 4 — unverified guidance, `docs/hosts/dgx-spark-vllm.md`

File uses long unwrapped lines. Match it — do not hard-wrap.
**Order matters:** T13 (corollary, end of §8) before T14 (§9), because both
insert at the same seam and §9 must come after the corollary.

### T13 — Add the §8 corollary
- **Action:** insert the corollary narrowing the `openai/` limitation to RLM
  Studio, with the `--enable-anthropic-api` version caveat and the
  verification command.
- **File:** `docs/hosts/dgx-spark-vllm.md` (after L200, inside §8)
- **FR:** FR-7
- **Done when:** the preceding sentence at L194 is untouched and its "once
  LiteLLM is ≥ 1.50" qualifier is intact (spec D-5); the corollary says the
  follow-up is narrowed, not removed; **no source file is edited** (C-6).

### T14 — Insert the quarantined §9
- **Action:** insert `## 9. Candidate models (UNVERIFIED — 2026-07-27)` with
  subsections 9.0–9.6 per `plan.md` §3 Task B.
- **File:** `docs/hosts/dgx-spark-vllm.md` (between the end of §8 and
  `## References`)
- **FR:** FR-4
- **Done when:** §9 sits **before** `## References` (spec D-2), not at EOF;
  the heading date is `2026-07-27` (matching spec FR-4 — do not change it to
  today); every configuration value is `<TBD-measure>` / `<TBD-parser>` or
  a value copied from §3 and labelled as such; every command block carries
  `# GUESS — not verified on this hardware`; 9.2 gives a per-item reason
  and the `MAX_JOBS=1` note for a ~75 GB first boot; 9.3's table has a §6
  step in every row and flags the `--reasoning-parser` inversion as a
  third-party, untested claim; 9.4 explains why the first-boot command uses
  the raw form rather than the wrapper (plan R-6 — `MAX_JOBS` is an
  unconditional export in the script); 9.6 records the `aider-polyglot
  python/bowling` n=3 promotion criterion as "alongside the existing anchor
  benchmark bullet in References" and **creates no table** (spec D-3); the
  References bullet at L207 is **not** edited.

### T15 — Add the §5 memory-table caveat
- **Action:** append the derivation note at the end of §5 — ~44 GB
  assumption, ~1 GB of KV budget per additional GB of weights, ~31 GB
  subtracted at every row for a ~75 GB model, `131072` out of reach, point
  at §9.
- **File:** `docs/hosts/dgx-spark-vllm.md` (after L85)
- **FR:** FR-5
- **Done when:** no table row value changed and no existing §5 sentence
  edited (C-1); the 1 GB-per-GB relationship is stated as a **derivation**,
  never as a measurement (spec T-j).

### T16 — Verify the quarantine held
- **Action:** run the AC-2 checks from `plan.md` §6 — the added-line flag
  grep, the placeholder grep, and the hunk-header audit.
- **File:** none
- **FR:** AC-2 / C-1 / C-2
- **Done when:** the flag-value grep returns nothing outside §9; the hunk
  headers in `dgx-spark-vllm.md` map to exactly four locations (L62, end of
  §5, end of §8, the §9 insert) and nothing else.

### T17 — Commit 3
- **Action:** show `git status` + `git diff`, get explicit approval, commit
  as `docs(hosts): add quarantined UNVERIFIED candidate-model section`.
- **File:** `docs/hosts/dgx-spark-vllm.md`
- **FR:** AC-5 support
- **Done when:** one file in `git show --stat`; no push (C-7).

---

## Phase 5 — deliverables and sweep

### T18 — Write the PR description
- **Action:** write the PR description with two top-level sections —
  "Staleness fixes (safe to merge)" (commits 1–2) and "Unverified guidance
  (needs a hardware run)" (commit 3) — plus a follow-ups list.
- **File:** `specs/dgx-spark-docs-refresh/pr-description.md` (new)
- **FR:** FR-8 / AC-5
- **Done when:** a reviewer could merge the first half without endorsing the
  second; the follow-ups section names R-4 (`scripts/dgx-spark/README.md`
  still pulls `gpt-oss:20b`), R-6 (`MAX_JOBS` is not overridable through the
  wrapper), OQ-2 (`vllm/README.md` omits both scripts), DF-1
  (`docs/troubleshoot.yaml` keeps the parameter-count framing) and the §8
  `openai/` → `hosted_vllm/` fix; and it states plainly that a default run
  of the start script now prints **two extra banner lines** (OQ-3) while the
  argv and exported environment are byte-identical, and that OQ-1's default
  was "no CHANGELOG entry".

### T19 — Final acceptance sweep
- **Action:** run every command in `plan.md` §6 end to end and capture the
  output: AC-1 greps + triage, AC-2 checks, AC-3 `bash -n` and dry-run diff,
  AC-4 link resolution + `uv run pytest tests/e2e/test_docs.py -q`, AC-5
  commit split, the `git diff --name-only` scope guard, and
  `git diff --exit-code -- scripts/dgx-spark/vllm/verify-tool-calls.sh`.
- **File:** none
- **FR:** AC-1..AC-5
- **Done when:** all five ACs demonstrated with pasted output; the scope
  guard lists only the five allowlisted paths; `verify-tool-calls.sh` is
  byte-identical (T1's no-op still holds); nothing was pushed. Report any
  AC that could not be demonstrated — do not declare done around it.
