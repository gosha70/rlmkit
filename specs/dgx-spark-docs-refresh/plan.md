---
feature_id: dgx-spark-docs-refresh
spec_mode: full
spec: ./spec.md
status: draft
date: 2026-07-28
origin:
  urls:
    - file:///Users/gosha/Downloads/files002/pitch-rlmkit-spark-docs.md
  transcripts:
    - "Team-lead delegation message, 2026-07-28: 'Finish plan.md and tasks.md against the existing spec.md. Feature id: dgx-spark-docs-refresh.'"
  origin_claim: |
    Inherited from spec.md. The origin is the pitch
    "Pitch — rlmkit: refresh DGX Spark model guidance", which asks that
    docs/hosts/dgx-spark.md stop contradicting the hardware-verified
    docs/hosts/dgx-spark-vllm.md on sizing, that newer candidate models be
    documented without being presented as tested, and that the vLLM start
    script become parameterizable — under the hard rule that no unverified
    configuration enters the verified sections.
---

# Plan — DGX Spark model-guidance refresh

Implements `spec.md`. Every change below traces to an FR in spec §3 and
respects the constraints in spec §4. Line numbers are against HEAD
(`e278ca2`, branch `docs/dgx-spark-vllm-qwen3-coder`) and were re-read
directly from the files, not copied from the spec's anchor table.

> **Anchor correction.** Spec §5 lists failure pattern #2 as
> `dgx-spark.md` L263–266. The bullet actually spans **L264–266**; L263
> is the tail of failure pattern #1. Every other anchor in spec §5
> verified exactly as written.

> **Date fields.** This plan is dated 2026-07-28. The §9 heading uses
> **2026-07-27**, matching spec FR-4 (the date the candidate set was
> assembled). That mismatch is intentional — do not "fix" it.

## 1. Approach

The work is sequenced by the **verified/unverified quarantine boundary**,
because that boundary — not file order — determines what can be reviewed
and merged independently.

`docs/hosts/dgx-spark-vllm.md` carries a hardware-verification claim on
line 3 that is the doc's entire value (spec C-1). `docs/hosts/dgx-spark.md`
carries no such claim, and `start-qwen3-coder-next.sh` carries a
*mechanically checkable* one (identical argv with no env vars set). So the
change splits cleanly into two halves:

- **Safe half** — everything whose correctness can be established without
  booting a model: FR-1/FR-2/FR-3 (de-staling a doc that makes no
  verification claim) and FR-6 (a script change provable by an argv diff).
- **Unverified half** — FR-4 (§9), FR-5 (§5 caveat), FR-7 (§8 corollary).
  All three land in the verified doc and all three are hypotheses,
  derivations, or version-dependent claims.

Sequencing consequences:

1. **Safe half first, and committed first.** This is not cosmetic. AC-5
   asks for a PR the reviewer can merge halfway; that is only true if the
   first commits stand alone. FR-2's band table cross-references
   `dgx-spark-vllm.md` as a *file*, not §9, so nothing in the safe half
   depends on §9 existing.
2. **Script before §9.** §9's starting-point commands name
   `VLLM_TOOL_CALL_PARSER` / `VLLM_REASONING_PARSER`. Writing the prose
   before the knobs exist invites documenting a variable that ends up
   spelled differently.
3. **§9 before the §5 caveat.** FR-5's note ends by pointing at §9 for
   per-model re-derivation. Insert order avoids a forward reference to a
   section that does not yet exist.
4. **§9 last inside `dgx-spark-vllm.md`, and inserted before
   `## References`** (spec D-2), so the verified numbered sections stay
   contiguous and References stays terminal.
5. **The `verify-tool-calls.sh` no-op is task T1, before anything else.**
   Spec D-1 established the change is already done. Confirming it first
   removes the temptation to invent a diff there later.

Nothing in the safe half touches §3/§5/§7 values of the verified doc. The
one edit that reaches into protected §3 is FR-6's env-var list at L62,
which spec C-1 explicitly permits and which adds names only.

## 2. Scope — files touched

**This table is an allowlist. A file not on it must not be modified by
this change, for any reason, including "while I was in there".**

| Path | FRs | Nature of edit |
|---|---|---|
| `docs/hosts/dgx-spark.md` | FR-1, FR-2, FR-3 | In-place prose. §3 pull list (L87–92) + trailing prose (L94–95); §5 Ollama example models (L211, FR-1 corollary — see R-5); §7 utilization default (L256, BF16 qualifier per DF-2); §7 failure pattern #2 (L264–266, additive sentences); §7 "Practical ranking" (L272–277, rewritten in GB) |
| `docs/hosts/dgx-spark-vllm.md` | FR-4, FR-5, FR-6, FR-7 | Append three names to the env-var parenthetical at L62 (names only, no value changed); additive note at end of §5 (after L85); additive corollary at end of §8 (after L200); new `## 9.` inserted between L200 and `## References` (L202) |
| `scripts/dgx-spark/vllm/start-qwen3-coder-next.sh` | FR-6 | Three new env vars with defaults; header comment block; two banner lines (OQ-3); argv assembly moved to an array so the reasoning-parser flag can be conditional |
| `scripts/dgx-spark/vllm/verify-tool-calls.sh` | FR-6.3 | **Read-only.** No-op verification (spec D-1). Must be byte-identical at the end of the change |
| `specs/dgx-spark-docs-refresh/pr-description.md` | FR-8 | New file — the PR description artifact (kept in-repo per the plan-artifact-locality rule) |

**Explicitly excluded**, each for a recorded reason:

- `CHANGELOG.md` — OQ-1 default is "no entry" (DF-7).
- `scripts/dgx-spark/vllm/README.md` — OQ-2 default is "no entry" (DF-4).
- `scripts/dgx-spark/README.md` — outside scope; the inconsistency FR-1
  creates there is recorded as R-4 and shipped as a follow-up, not fixed.
- `docs/troubleshoot.yaml` — spec C-8 / DF-1. Same stale framing, out of
  scope, and schema-validated by `tests/e2e/test_docs.py`.
- `src/`, `frontend/`, `tests/` — spec C-6. The three `openai/` call sites
  are **not** fixed here; FR-7 documents a corollary about them and stops.

## 3. Per-task change plan

### Task A — de-stale `docs/hosts/dgx-spark.md` (FR-1, FR-2, FR-3)

File style: hard-wrapped at ~70 columns. Match it. Do not reflow
neighbouring lines (C-4).

#### A.1 — §3 pull list and trailing prose (FR-1)

*Target:* `docs/hosts/dgx-spark.md` L87–92 (code fence) and L94–95 (prose).

*Change:* replace two of the three pulls; keep one tiny model for smoke
tests. Keep `llama3.2` as that tiny model rather than introducing a new
tiny tag — it is already in the doc, and every tag we add is an
unverifiable registry claim (spec T-d, risk R-1).

Draft:

````markdown
```bash
ollama pull qwen3.6:35b-a3b    # current-generation MoE
ollama pull qwen3.5:27b        # current-generation dense
ollama pull llama3.2           # tiny — smoke tests only
ollama list
```

On Spark the binding constraint is a model's **weight footprint in
GB**, not its parameter count: quantization changes bytes per
parameter by up to 4x, so a parameter count alone tells you nothing
about whether a model fits. See §7 for the GB-keyed bands and the
Ollama vs vLLM sizing rule of thumb.
````

*Must not:* claim throughput, VRAM figures, or "verified" status for
either new tag (C-2, T-b/T-c are third-party).

#### A.2 — §5 Ollama example models (FR-1 corollary, R-5)

*Target:* `docs/hosts/dgx-spark.md` L211 —
`| Model    | e.g. \`qwen2.5:14b\`, \`llama3.2\`   |`

*Change:* one line. Replace `qwen2.5:14b` with `qwen3.5:27b` so the RLM
Studio table does not name a model §3 no longer tells you to pull. Keep
`llama3.2`. Preserve the table's column padding.

*Rationale:* this is a self-contradiction **inside the file FR-1 edits**,
created by FR-1. Leaving it is worse than a one-line diff. (Contrast R-4,
which is in a file outside the allowlist and is therefore left alone.)

#### A.3 — §7 utilization default gets a BF16 qualifier (FR-2 / DF-2)

*Target:* `docs/hosts/dgx-spark.md` L256 — "Start at 0.4 for 7B-class
models, 0.7 for 32B-class."

*Change:* qualify in place, do not rewrite (DF-2). Draft:

```markdown
Start at 0.4 for 7B-class models, 0.7 for 32B-class (BF16 and
unquantized — for quantized weights, size from the GB bands below).
```

#### A.4 — §7 failure pattern #2 made measurable (FR-3)

*Target:* `docs/hosts/dgx-spark.md` L264–266.

*Change:* **additive only.** The existing three sentences stay verbatim
(C-3 binds this whole "Common errors" region). Append to the bullet:

```markdown
2. **Model loads, then no KV cache room left.** Seen with
   `Qwen/Qwen3-32B` at 0.4. Weights fit, cache does not. Raise
   utilization *and* reduce `--max-model-len`. This is measurable
   rather than trial-and-error: vLLM prints the resulting KV-cache
   token budget during boot (the exact wording varies by version —
   look for the startup line naming the GPU KV cache size and a
   token count). If that number is below your client's worst-case
   `prompt + max_tokens`, requests will fail no matter how cleanly
   the weights loaded. Large-weight MoE models make this worse than
   the `Qwen3-32B` example above suggests: weights are only the
   first claim on a shared pool.
```

*Hedging is mandatory.* Spec T-g: the literal string `GPU KV cache size:`
appears nowhere in this repo (re-grepped — the only match for "KV cache
size" is a prose comment in `serve-vllm-dgx.sh`). Describe the line; do
not quote it as verbatim-verified.

#### A.5 — §7 "Practical ranking" rewritten in GB (FR-2)

*Target:* `docs/hosts/dgx-spark.md` L272–277 (the four lines of the
ranking block, heading included). Replace wholesale.

Draft:

```markdown
**Practical ranking for this Spark setup — keyed on weights in GB:**

- **<= 45 GB of weights — comfortable.** Fits alongside a large KV
  cache. Verified example: `RedHatAI/Qwen3-Coder-Next-NVFP4`
  (~44 GB NVFP4) at `--max-model-len 131072` with
  `--gpu-memory-utilization 0.72` — see
  [`dgx-spark-vllm.md`](dgx-spark-vllm.md) §3.
- **45–90 GB of weights — possible with tuning and a reduced
  context.** Raise utilization, lower `--max-model-len`, and expect
  to re-derive both per model. Between 90 and 100 GB there is no
  configuration reported either way — treat that range as untested.
- **> 100 GB of weights — needs multi-node.** No single-Spark
  configuration fits the weights plus a usable KV cache.

Parameter count is the wrong unit here, and MoE sparsity is the
second trap: `A10B` means 10 B *active* parameters, which reduces
**compute** per token but **not KV cache**. KV is sized by total
layers x KV heads x context, so a 122B-A10B model reserves KV as if
every layer were dense. Over-provisioning `--max-model-len` on a
large MoE model is the usual way this goes wrong.
```

Numbers are taken from the pitch and spec FR-2 verbatim; ~44 GB, 131072
and 0.72 come from `dgx-spark-vllm.md:74`, `:82`, `:41`. Nothing is
invented. The 90–100 GB clause is the only addition — see R-7.

### Task B — quarantined §9 in `dgx-spark-vllm.md` (FR-4)

File style: long unwrapped lines. Match it — do **not** hard-wrap.

*Insertion point:* after the FR-7 corollary at the end of §8 (i.e. after
current L200 plus the Task E paragraph), **before** `## References`
(current L202). Never appended to EOF (spec D-2).

*Heading (exact):* `## 9. Candidate models (UNVERIFIED — 2026-07-27)`

Required contents, in order:

**9.0 Lede** — one paragraph, must state: nothing in §9 has been booted on
this hardware; §3 is verified for `RedHatAI/Qwen3-Coder-Next-NVFP4` and for
no other model; every number below is a hypothesis; a `<TBD-measure>` must
never be pasted into a production command — measure it with the command
given beside it, record the result, then promote per §9.5.

**9.1 Candidates** — table reproducing the pitch's three rows with an
explicit provenance column. No throughput number is restated as a repo
claim (T-b). Columns: `Model | Reported weights | Source (third-party,
unreproduced here)`. Rows: `RedHatAI/Qwen3.5-122B-A10B-NVFP4` / ~75 GB /
NVIDIA developer forum thread; `Qwen3.6-35B-A3B (FP8)` / ~35 GB /
community blog; `Qwen3.5-27B-FP8` / ~31 GB / community blog. One trailing
sentence naming the verified reference point (~44 GB) so the deltas are
readable.

**9.2 Carries over unchanged** — bullets, each with its reason category:

| Item | Reason it is model-independent |
|---|---|
| Blocker #1 (`MAX_JOBS=2 NINJA_JOBS=2`) | Toolchain-level — flashinfer's JIT concurrency, not the checkpoint. **Gets worse with larger weights**: the nvcc processes and the resident weights draw on the same unified pool, so for a ~75 GB first boot start at `MAX_JOBS=1 NINJA_JOBS=1` |
| Blocker #3 (`--max-model-len` vs envelope) | Client-side — sized by the agent client's request envelope |
| Blocker #4 (`hosted_vllm/` vs `openai/`) | Transport-level — a LiteLLM route-selection property; the model is not involved |
| `--enforce-eager` | Platform-level — CUDA-graph capture stability on Spark |
| `pkill` / `drop_caches` preamble | Host-state hygiene before the memory probe |
| `--trust-remote-code` | Loader-level — required whenever the checkpoint ships custom `modeling_*.py`; check per checkpoint, harmless to keep |

**9.3 Must be re-verified per model** — table, one row per item, columns
`Flag | Why it may change | Starting point | §6 step that tests it`:

- `--tool-call-parser` — `qwen3_coder` targets Qwen3-Coder-Next's
  `<tool_call>` XML emission; other families emit other shapes (T-h) —
  `<TBD-parser>` — §6 Step 2.
- `--gpu-memory-utilization` — scales with weight footprint —
  `<TBD-measure>` — §6 Step 1, then Step 3 under load.
- `--max-model-len` — the KV budget shrinks as weights grow (see the §5
  note) — `<TBD-measure>` — §6 Step 1.
- `--reasoning-parser` — **Blocker #2's "omit it" guidance likely
  inverts.** Qwen3-Coder-Next has no separate thinking output; Qwen3.5/3.6
  are reported to have thinking modes whose tool-calling reliability
  depends on thinking being *enabled*. Untested here (T-f) — start
  omitted, then try `<TBD-parser>` if Step 2 fails with the XML landing in
  `reasoning` — §6 Step 2.
- `--model` / `--served-model-name` — per checkpoint — n/a — §6 Step 1.

**9.4 Starting-point commands** — every block prefixed with a comment
reading `# GUESS — not verified on this hardware`. Two forms, and the
reason for two is load-bearing (see R-6):

1. *First boot of a large candidate* — the raw
   `python -m vllm.entrypoints.openai.api_server` form derived from §3,
   because `MAX_JOBS` / `NINJA_JOBS` / `VLLM_USE_FLASHINFER_MOE_FP4` are
   **unconditional `export`s inside the wrapper script** (L57–62), so
   `MAX_JOBS=1 bash start-qwen3-coder-next.sh` silently has no effect.
   One sentence in §9 must say this.
2. *Subsequent boots / parser experiments* — the wrapper one-liner:

   ```bash
   # GUESS — not verified on this hardware.
   VLLM_MODEL_PATH=~/dgx-spark-vllm/models/<candidate> \
   VLLM_SERVED_MODEL_NAME=<org>/<candidate> \
   VLLM_MAX_MODEL_LEN=<TBD-measure> \
   VLLM_GPU_UTIL=<TBD-measure> \
   VLLM_TOOL_CALL_PARSER=<TBD-parser> \
   bash scripts/dgx-spark/vllm/start-qwen3-coder-next.sh
   ```

   Each delta from §3 justified in prose: why the parser is a placeholder
   (T-h), why the two memory values are placeholders (the §5 derivation),
   why `--enforce-eager` and `--trust-remote-code` are not deltas at all.

**9.5 Measurement commands** — one per placeholder, so no `<TBD-measure>`
is ever resolved by guessing:

| Placeholder | How to measure |
|---|---|
| weight footprint | `du -sh ~/dgx-spark-vllm/models/<candidate>` — on-disk size is the usable proxy for resident weights |
| `--max-model-len` actually served | `curl -s http://localhost:8000/v1/models \| jq '.data[0].max_model_len'` (§6 Step 1) |
| KV token budget | boot once at a deliberately low `--max-model-len`, read the KV-cache line from the boot log (wording varies by version), then scale |
| available parsers | `python -m vllm.entrypoints.openai.api_server --help` and read the `--tool-call-parser` choices for the local build |
| `--enable-anthropic-api` availability | `python -m vllm.entrypoints.openai.api_server --help \| grep anthropic` (see §8 corollary) |

**9.6 Promotion criteria** — §6 Steps 1, 2 and 3 all PASS against the
candidate; every `<TBD-measure>` replaced by a measured value; and an
`aider-polyglot python/bowling` n=3 result recorded **alongside the
existing anchor-benchmark bullet in `## References`**. Spec D-3: there is
no anchor *table* — do not create one, and do not edit the References
bullet in this change (there is no result yet — C-2). Only when all of
that holds may a configuration move from §9 into §3/§5.

### Task C — §5 memory-table caveat (FR-5)

*Target:* `docs/hosts/dgx-spark-vllm.md`, appended after L85 (the
flashinfer-recompilation paragraph), i.e. at the end of §5. **No row
value, and no existing sentence, is edited** (C-1).

Draft (single unwrapped paragraph, matching file style):

```markdown
**These rows assume ~44 GB of weights and are not portable as-is.** Weights and KV cache come out of the same fixed unified pool, so each additional GB of weights costs roughly 1 GB of KV budget. That is a derivation from a fixed pool, not a measured result — but it is the right first approximation. A ~75 GB model therefore subtracts ~31 GB at every row, which is enough to put the `131072` row out of reach entirely. Do not scale these rows by eye for another model; re-derive them per §9.
```

Spec T-j: presented as a derivation, never as a measurement. The 75 − 44 =
~31 GB arithmetic uses only numbers already in these docs.

### Task D — parameterize `start-qwen3-coder-next.sh` (FR-6)

Exactly three new variables (DF-6 — the script hardcodes more than three
knobs; the rest stay hardcoded).

#### D.1 — header comment block (L15–20)

Append three lines after L20. Do **not** realign the existing six lines
(C-4); `VLLM_MAX_NUM_BATCHED_TOKENS` is longer than the current longest
name, so it takes a single space before `default:` rather than forcing a
reflow:

```bash
#   VLLM_TOOL_CALL_PARSER   default: qwen3_coder
#   VLLM_MAX_NUM_BATCHED_TOKENS default: 8192
#   VLLM_REASONING_PARSER   default: (unset — §3 passes no --reasoning-parser; see Blocker #2)
```

#### D.2 — defaults (after L33)

```bash
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-qwen3_coder}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-}"
```

`:-` keeps all three safe under the file's `set -euo pipefail`.

#### D.3 — banner (L64–70), per OQ-3

Two added lines, placed to keep the existing five in order:

```bash
echo "    tool-call-parser:     $VLLM_TOOL_CALL_PARSER"
echo "    reasoning-parser:     ${VLLM_REASONING_PARSER:-(none)}"
```

`--max-num-batched-tokens` is deliberately not echoed — the banner already
omits `--max-num-seqs` and other fixed flags; it shows operator-facing
knobs only.

#### D.4 — argv assembly (L72–83)

`--reasoning-parser` must be **absent**, not empty, by default. A bare
`"${ARR[@]}"` on an empty array trips `set -u` on bash < 4.4, so build the
whole argument list as an array — which is also the existing repo pattern
in `scripts/dgx-spark/vllm/serve-vllm-dgx.sh:69-77`:

```bash
VLLM_ARGS=(
  --model "$VLLM_MODEL_PATH"
  --served-model-name "$VLLM_SERVED_MODEL_NAME"
  --enforce-eager
  --max-model-len "$VLLM_MAX_MODEL_LEN"
  --max-num-seqs 1
  --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$VLLM_GPU_UTIL"
  --enable-auto-tool-choice
  --tool-call-parser "$VLLM_TOOL_CALL_PARSER"
  --trust-remote-code
)

# Verified config passes no --reasoning-parser at all (Blocker #2).
# Only add the flag when explicitly requested.
if [[ -n "$VLLM_REASONING_PARSER" ]]; then
  VLLM_ARGS+=(--reasoning-parser "$VLLM_REASONING_PARSER")
fi

VLLM_ARGS+=(--host 0.0.0.0 --port "$VLLM_PORT")

exec python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}"
```

Flag order in the default case is unchanged from L73–83; the argv diff in
§6 is what proves it. Do not rename the script (FR-6.4).

#### D.5 — document the new knobs in §3 (FR-6, permitted by C-1)

*Target:* `docs/hosts/dgx-spark-vllm.md` L62, the parenthetical listing
six env vars. **Append names only**; do not reorder the existing six, do
not touch any value in §3. Draft tail:

```markdown
… (overridable via `VLLM_MODEL_PATH`, `VLLM_SERVED_MODEL_NAME`, `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_UTIL`, `VLLM_PORT`, `VLLM_VENV`, `VLLM_TOOL_CALL_PARSER`, `VLLM_MAX_NUM_BATCHED_TOKENS`, and `VLLM_REASONING_PARSER` — the last defaults to unset, because the verified config above passes no `--reasoning-parser`; see Blocker #2).
```

#### D.6 — `verify-tool-calls.sh` (FR-6.3) — no-op

Spec D-1: `VLLM_MODEL` is already an env var
(`verify-tool-calls.sh:26`, documented at `:19`). Re-confirmed at HEAD.
**Change nothing.** Task T1 records the confirmation; T-final asserts the
file is byte-identical.

### Task E — narrow the §8 known limitation (FR-7)

*Target:* `docs/hosts/dgx-spark-vllm.md`, inserted after L200 (the "**Fix
(out of scope for this docs change):**" paragraph), still inside §8, ahead
of the new §9.

Draft:

```markdown
**Corollary — this blocks RLM Studio, not every client.** The three call sites above are RLMKit's own prefix tables, so the limitation is scoped to requests that leave RLMKit through LiteLLM. A Claude Code client pointed at vLLM's native Anthropic-compatible endpoint (`--enable-anthropic-api`) takes LiteLLM out of the path entirely, and Blocker #4's cause is LiteLLM's route selection — so that path cannot hit it. Caveat: `--enable-anthropic-api` is version-dependent and is not used anywhere in this repo; verify it exists in your local build (`python -m vllm.entrypoints.openai.api_server --help | grep anthropic`) before relying on it. This narrows the tracked follow-up above and may lower its priority — it does not remove it, and nothing here changes the preceding sentence.
```

Spec D-5: the preceding sentence's "once LiteLLM is ≥ 1.50" qualifier is
left intact and is **not** restated in the pitch's shortened form.

## 4. Risks

**R-1 — unverifiable Ollama tags.** `qwen3.6:35b-a3b` and `qwen3.5:27b`
(T-d) cannot be resolved without network access to the Ollama registry.
*Mitigation:* keep the already-present `llama3.2` as the smoke-test pull so
at least one tag in §3 is known-good; add no throughput or size claim for
the two new tags; the §3 block is a `pull` example, not a verified config.

**R-2 — §9 read as endorsement.** A reader skims and copies a
starting-point command. *Mitigation:* UNVERIFIED in the heading, the 9.0
lede, `# GUESS` on every command block, `<TBD-*>` on every value, and the
AC-2 sweep in §6 below.

**R-3 — eroding the line-3 verification claim.** *Mitigation:* four
insertions total in that file, none inside the §3 code block or the §5
table rows; the only protected-section edit is an append of three names to
L62's parenthetical, explicitly permitted by C-1.

**R-4 — `scripts/dgx-spark/README.md` will name a model the host doc no
longer lists** (spec D-6). Confirmed at HEAD: `:19` says
`ollama pull gpt-oss:20b` and `:28` uses `model="gpt-oss:20b"` in the
Python example. Once FR-1 drops `gpt-oss:20b` from `dgx-spark.md` §3 the
two files disagree. *Mitigation:* that file is outside the §2 allowlist —
leave it. Record it as a named follow-up in the PR description rather than
widening the diff. Reversing this is a one-line change if the reviewer
prefers.

**R-5 — the same inconsistency *inside* an in-scope file.** NEW; not
recorded in spec.md. `docs/hosts/dgx-spark.md:211` (the RLM Studio Ollama
table) gives example models `qwen2.5:14b`, `llama3.2` — the exact list FR-1
removes from §3. *Mitigation:* fix it (task A.2), because unlike R-4 it is
a self-contradiction within the file this change is de-staling.

**R-6 — the wrapper cannot deliver FR-4's `MAX_JOBS=1` advice.** NEW; not
recorded in spec.md. `start-qwen3-coder-next.sh:57-62` sets
`VLLM_USE_FLASHINFER_MOE_FP4`, `MAX_JOBS` and `NINJA_JOBS` with
unconditional `export`s, so `MAX_JOBS=1 bash start-qwen3-coder-next.sh`
silently has no effect. *Mitigation:* §9.4's first-boot command uses the
raw §3-derived form and states why in one sentence. Do **not** parameterize
`MAX_JOBS` — FR-6/DF-6 authorize exactly three variables. Record as a
follow-up.

**R-7 — the band boundaries leave 90–100 GB undefined.** NEW. The pitch
and spec FR-2 both say `<= 45` / `45–90` / `> 100`. *Mitigation:* keep both
numbers exactly as specified (moving one would be inventing a threshold)
and add a single clause marking 90–100 GB as untested. Honest, invents
nothing.

**R-8 — `set -u` and the conditional flag.** A conditionally-appended
`--reasoning-parser` via a possibly-empty array errors on bash < 4.4.
*Mitigation:* single always-non-empty `VLLM_ARGS` array (D.4), matching
`serve-vllm-dgx.sh`. `bash -n` alone will not catch this — the dry run in
§6 will.

**R-9 — Learn-tab rendering.** DF-5: `dgx-spark.md` is served to RLM Studio
as cookbook slug `hosts-dgx-spark` (`src/rlmkit/server/routes/docs.py:56`)
and that renderer resolves no relative links, so A.5's cross-reference
renders unlinked. *Mitigation:* write the link so the *text* names the file
(`[\`dgx-spark-vllm.md\`](dgx-spark-vllm.md)`), readable either way. AC-4
is scoped to repo-relative resolution. Run `tests/e2e/test_docs.py`
regardless — it asserts the file exists and is non-empty.

**R-10 — hard-wrap vs long-line style mismatch.** The two docs use opposite
conventions (C-4). *Mitigation:* the drafts above are already written in
each file's own style; a reviewer should reject any hunk that reflows
untouched lines.

## 5. Open questions

Each has a documented default. **The build agent is never blocked by these
— apply the default and note it.**

### OQ-1 — Does this change warrant a `CHANGELOG.md` entry?

**Default: no entry.** Reasoning (DF-7): the closest precedent, `f336ffc`
("docs(hosts): add Qwen3-Coder-Next on DGX Spark + vLLM operator manual"),
*added* `dgx-spark-vllm.md` and both scripts — a strictly larger docs
change — and left no CHANGELOG entry. Grepping `CHANGELOG.md` for
`spark|dgx|qwen3-coder|operator manual` returns only L48–51, which is a
docs paragraph delivered as part of a feature phase, not a docs-only entry.
`CHANGELOG.md` is therefore **not** on the §2 allowlist.
*Reversal:* if the reviewer asks, add one bullet under `## [Unreleased]`;
one-line change, no re-verification needed.

### OQ-2 — Should `scripts/dgx-spark/vllm/README.md` document the two undocumented scripts?

**Default: no.** Reasoning (DF-4): that README documents five scripts and
mentions neither `start-qwen3-coder-next.sh` nor `verify-tool-calls.sh`.
Re-confirmed at HEAD. It is a pre-existing gap — it predates this change
and nothing here worsens it. Adding two sections is a separate docs change
with its own review surface, and the pitch's scope line covers
`scripts/dgx-spark/vllm/*.sh`, not the README.
*Reversal:* purely additive and independent; can be a follow-up PR. Record
it as a named follow-up in the PR description.

### OQ-3 — May the start script's banner gain two lines?

**Default: yes — exactly two** (`tool-call-parser:` and
`reasoning-parser:`). Reasoning: spec C-5 scopes "byte-identical" to *the
exported environment and the assembled argv*, and explicitly defers the
banner to this question. The banner is informational stdout; it is not
consumed by anything in the repo (grep confirms no caller parses it).
Consequence to state plainly in the PR description: **a default run's
stdout does change by two lines** — `tool-call-parser:     qwen3_coder`
and `reasoning-parser:     (none)`. The AC-3 dry run therefore diffs argv
and exported env, not raw stdout.
*Reversal:* drop both `echo` lines; the knobs remain documented in the
header block (D.1) and nothing else changes.

## 6. Verification plan

Run from the repo root, `/Users/gosha/dev/repo/rlmkit`.

### AC-1 — no parameter-count ceilings left unqualified

```bash
rg -n '[0-9]+B[–-][0-9]+B|[0-9]+B-class|[0-9]+B instruct' \
  docs/hosts/dgx-spark.md docs/hosts/dgx-spark-vllm.md

rg -n 'Fits comfortably|Not recommended here|Possible with tuning|fit comfortably up to' \
  docs/hosts/dgx-spark.md docs/hosts/dgx-spark-vllm.md
```

The second grep must return **nothing** — those are the old ranking bullets
(L274–277) and the old §3 prose (L94).

The first grep returns hits by design. Triage table — every surviving hit
must be on this list, and any hit not on it fails AC-1:

| Line (pre-change) | Text | Why it is acceptable |
|---|---|---|
| L155, L161 | `7B instruct model` / `--model Qwen/Qwen2.5-7B-Instruct` | §4 example command, not a ceiling |
| L256 | `0.4 for 7B-class … 0.7 for 32B-class` | BF16-qualified in place by A.3 |
| L265 | `Qwen/Qwen3-32B at 0.4` | An observed data point in a failure pattern, not a ceiling |
| L268–269 | `70B-class is too large for Spark` | Already BF16-qualified — DF-3, do not touch |
| L320 | `Qwen/Qwen2.5-7B-Instruct` | Endpoint-shape rule, unrelated to sizing |
| L211 | `qwen3.5:27b` after A.2 | A model *example*, not a ceiling |

Baseline for comparison (captured at HEAD, before any edit): the first grep
currently returns 16 lines in `dgx-spark.md` and 0 in `dgx-spark-vllm.md`.

### AC-2 — every new configuration value is in §9 or a placeholder

```bash
# Added lines in the verified doc that carry a concrete flag value:
git diff -U0 -- docs/hosts/dgx-spark-vllm.md \
  | grep '^+' | grep -vE '^\+\+\+' \
  | grep -E -- '--(gpu-memory-utilization|max-model-len|tool-call-parser|reasoning-parser|max-num-batched-tokens|max-num-seqs)[= ][^<]' \
  | grep -v '<TBD-'
```

Must return **nothing**, or every returned line must sit below the `## 9.`
heading. Cross-check the placeholder count is non-zero:

```bash
rg -n '<TBD-(measure|parser)>' docs/hosts/dgx-spark-vllm.md
rg -n 'UNVERIFIED|# GUESS' docs/hosts/dgx-spark-vllm.md
```

Also assert §3/§5/§7 values are untouched — the only permitted hunk in that
range is the L62 name append:

```bash
git diff -U0 -- docs/hosts/dgx-spark-vllm.md | grep -E '^@@'
# Every hunk header must map to: L62 (env-var names), end of §5 (~L85),
# end of §8 (~L200), and the §9 insert before L202. Nothing else.
```

### AC-3 — script runs identically with no env vars set

Syntax and lint:

```bash
bash -n scripts/dgx-spark/vllm/start-qwen3-coder-next.sh
command -v shellcheck >/dev/null && shellcheck scripts/dgx-spark/vllm/start-qwen3-coder-next.sh
```

Argv + exported-env diff, before vs after. The recipe stubs `python`,
`sudo` and `pkill` so the script is safe to run on a dev machine:

```bash
TMP=$(mktemp -d)
mkdir -p "$TMP/bin" "$TMP/venv/bin"
: > "$TMP/venv/bin/activate"

cat > "$TMP/bin/python" <<'EOF'
#!/usr/bin/env bash
printf 'ARGV %s\n' "$@"
env | grep -E '^(VLLM_USE_FLASHINFER_MOE_FP4|MAX_JOBS|NINJA_JOBS)=' | sort | sed 's/^/ENV /'
EOF
printf '#!/usr/bin/env bash\nexit 1\n' > "$TMP/bin/sudo"      # drop_caches path is tolerated
printf '#!/usr/bin/env bash\nexit 0\n' > "$TMP/bin/pkill"     # never kill real processes
chmod +x "$TMP/bin/"*

git show HEAD:scripts/dgx-spark/vllm/start-qwen3-coder-next.sh > "$TMP/before.sh"

run() {  # $1 = script path, $2 = output file, rest = env assignments
  local script="$1" out="$2"; shift 2
  env "$@" PATH="$TMP/bin:$PATH" VLLM_VENV="$TMP/venv" \
    bash "$script" 2>/dev/null | grep -E '^(ARGV|ENV) ' > "$out"
}

run "$TMP/before.sh" "$TMP/before.txt"
run scripts/dgx-spark/vllm/start-qwen3-coder-next.sh "$TMP/after.txt"

diff "$TMP/before.txt" "$TMP/after.txt" && echo "AC-3 default path: IDENTICAL"
```

`diff` must be empty. Then the positive half of AC-3 — a different parser
actually reaches the server:

```bash
run scripts/dgx-spark/vllm/start-qwen3-coder-next.sh "$TMP/override.txt" \
  VLLM_TOOL_CALL_PARSER=hermes VLLM_REASONING_PARSER=qwen3 VLLM_MAX_NUM_BATCHED_TOKENS=4096

grep -q 'ARGV hermes'  "$TMP/override.txt" && \
grep -q 'ARGV --reasoning-parser' "$TMP/override.txt" && \
grep -q 'ARGV 4096'    "$TMP/override.txt" && echo "AC-3 override path: OK"

# and the default run must NOT contain the flag at all:
grep -q 'reasoning-parser' "$TMP/after.txt" && echo "FAIL: flag leaked into default argv"
```

### AC-4 — relative links resolve

```bash
cd docs/hosts
grep -oE '\]\([^)]+\)' dgx-spark.md dgx-spark-vllm.md \
  | sed -E 's/^([^:]+):\]\((.+)\)$/\1 \2/' \
  | grep -vE ' (https?|mailto):' \
  | while read -r src link; do
      path="${link%%#*}"
      [ -z "$path" ] && continue
      [ -e "$path" ] || echo "BROKEN: $src -> $link"
    done
cd - >/dev/null
```

Must print nothing. Known links that must survive: `dgx-spark.md`,
`vllm.md`, `README.md#…` (from `dgx-spark-vllm.md`), and both
`../../scripts/dgx-spark/vllm/*.sh` paths at `dgx-spark-vllm.md:62` and
`:89`. Plus:

```bash
uv run pytest tests/e2e/test_docs.py -q      # DF-5: dgx-spark.md is served to the Learn tab
```

### AC-5 — reviewable split

```bash
git log --oneline master..HEAD
git show --stat HEAD~2   # docs(hosts) de-stale — dgx-spark.md only
git show --stat HEAD~1   # script parameterization — the .sh + dgx-spark-vllm.md:62
git show --stat HEAD     # unverified guidance — §9, §5 note, §8 corollary
test -f specs/dgx-spark-docs-refresh/pr-description.md
```

The PR description must carry two top-level headed sections — "Staleness
fixes (safe to merge)" and "Unverified guidance (needs a hardware run)" —
name the commits under each, and list the follow-ups: R-4, R-6, OQ-2, DF-1,
and the §8 `openai/` fix.

### Global scope guard

```bash
git diff --name-only master..HEAD | sort
```

Must list only the five paths in §2 and nothing else. Any other path is a
scope violation, not a judgement call.

```bash
git diff --exit-code -- scripts/dgx-spark/vllm/verify-tool-calls.sh   # D-1: must be a no-op
```

## 7. Delegation

CLAUDE.md's role table has no owner for `docs/` or `scripts/`, so
ownership is assigned by nearest domain. One owner per file, no overlaps.

| Owner | Tasks | Files owned | Acceptance |
|---|---|---|---|
| **Team Lead** | A (FR-1/2/3), B (FR-4), C (FR-5), E (FR-7), D.5, FR-8 | `docs/hosts/dgx-spark.md`, `docs/hosts/dgx-spark-vllm.md`, `specs/dgx-spark-docs-refresh/pr-description.md` | AC-1, AC-2, AC-5; C-1/C-2/C-3/C-4 hold |
| **Runtime Engineer** | D.1–D.4 (FR-6) | `scripts/dgx-spark/vllm/start-qwen3-coder-next.sh` | AC-3 both halves; `bash -n` clean; argv+env diff empty on the default path |
| **QA Engineer** | T1 no-op confirmation, the §6 verification sweep | none (read-only) + the throwaway dry-run harness under the scratchpad | AC-1..AC-5 all demonstrated with captured output; scope guard clean |

This is a small change; a single agent doing all three in order is also
acceptable. What is **not** acceptable is two agents editing
`dgx-spark-vllm.md` concurrently — four insertion points in one file with
line-number-sensitive anchors.

### Commit strategy

Stay on the current branch `docs/dgx-spark-vllm-qwen3-coder` (already the
DGX Spark docs branch, working tree clean). Three commits, mapped to the
PR halves so the reviewer can merge the first two without endorsing the
third:

1. `docs(hosts): size DGX Spark model guidance by weight footprint` —
   Task A only.
2. `feat(scripts): parameterize tool-call/reasoning parser in the Spark vLLM starter` —
   Task D (script + the L62 env-var list).
3. `docs(hosts): add quarantined UNVERIFIED candidate-model section` —
   Tasks B, C, E.

**No `git push`** (C-7). Show `git status` and `git diff` before each
commit and wait for explicit user approval — approval on commit 1 is not
approval for 2 and 3.
