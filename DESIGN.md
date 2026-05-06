# LLM Wiki Backbone — Design

Companion to `specs/llm-wiki-backbone/spec.md` and `plan.md`. This
document records the modes implemented, the directory layout, the
schema decisions, and — most importantly — the explicit
**borrow / divergence ledger** versus the reference design in
`gosha70/code-copilot-team`.

## Modes A–D from issue gosha70/rlmkit#37

| Mode | Issue label | Implementation |
|---|---|---|
| **A** | `direct` ingest of one source | `LLMBackend` in `rlmkit.wiki.backends`. CLI: `wiki-ingest --backend test <src>` (test) or programmatic `Ingestor(backend=LLMBackend(client=…))`. |
| **B** | `rag` wiki query | `query_wiki(question, wiki_dir)` in `rlmkit.wiki.query`. Keyword-overlap scoring; top-k page bodies returned as the answer when `rlm` is `None`. |
| **C** | `rlm_wiki` / `wiki + rlm` | **`RLMBackend`** in `rlmkit.wiki.backends` — the differentiator. Wraps an `RLM` controller, loads the corpus into the REPL as `P`, asks the model to apply the gate and draft the page using `peek` / `grep` / `subcall` to navigate. |
| **D** | wiki query with RLM fallback | `query_wiki(question, wiki_dir, raw_dir, rlm)`. The wiki-first LLM call returns `INSUFFICIENT_INFORMATION` when coverage is weak; the path then escalates to `rlm.run(prompt=corpus, query=question)` and surfaces the RLM trace. |

Modes A and B exist in cct already (in slightly different forms);
modes C and D are RLMKit's contribution and are what positions
RLMKit as "the scalable reasoning engine behind a persistent LLM
Wiki knowledge layer," in the words of the issue.

## Directory layout

```
src/rlmkit/wiki/
  __init__.py
  errors.py        # exit-code-mapped exception hierarchy
  schema.py        # PAGE_TYPE_DIRS, frontmatter parser (stdlib-only)
  proposal.py      # IngestRequest / IngestProposal / proposal-file render
  backends.py      # IngestBackend Protocol + TestBackend / LLMBackend / RLMBackend
  ingest.py        # Ingestor orchestrator + two-layer validate_proposal
  linter.py        # Python port of cct's lint-wiki.sh (six checks)
  query.py         # wiki-first query path with RLM fallback
  cli.py           # python -m rlmkit.wiki.cli {ingest,query,lint}

knowledge/
  raw/                          # ingestion input (gitignored content)
  wiki/
    index.md  overview.md  log.md
    concepts/  workflows/  incidents/  decisions/
    playbooks/ glossary/   open-questions/
    schema/                     # cct-borrowed schema (verbatim, with attribution)
    scripts/                    # reserved for future use

scripts/
  wiki-ingest                   # exec → python -m rlmkit.wiki.cli ingest
  wiki-query                    # exec → python -m rlmkit.wiki.cli query
  wiki-lint                     # exec → python -m rlmkit.wiki.cli lint

specs/llm-wiki-backbone/
  spec.md  plan.md

tests/wiki/
  fixtures/sample-incident.md
  test_schema_and_linter.py
  test_ingest_test_backend.py
  test_rlm_backend_e2e.py       # the differentiator E2E
  test_query.py                 # Mode B + Mode D
```

## Borrow / divergence ledger

Every borrow from `code-copilot-team` is labelled, and every
divergence is labelled. **Borrow** = adopted as-is; **diverge** =
intentionally different.

### Borrowed verbatim
- **Page-type taxonomy** (10 types: `concept | workflow | incident |
  decision | playbook | glossary | open-question | index | log |
  overview`) — `knowledge/wiki/schema/page-types.md`. Identical schema
  keeps cct and RLMKit wikis interchangeable for any future cross-
  curation work.
- **Universal frontmatter** (`page_type`, `slug`, `title`, `status`,
  `last_reviewed`, `sources:`) — same file. Already validated end-to-
  end in cct.
- **Four-question gate** (reusable / citable / non-duplicative /
  new-contributor-relevant) — `knowledge/wiki/schema/ingest-rules.md`.
- **Citation rules** (`path` + `sha`, `issue`/`pr`, `url` + `retrieved`)
  — `knowledge/wiki/schema/citation-rules.md`.
- **Lint rule set** (six checks) — `knowledge/wiki/schema/lint-rules.md`.
- **`WIKI_MAINTAINER` curator persona** — `knowledge/wiki/schema/WIKI_MAINTAINER.md`.
- **Two-layer validation** (JSON shape + semantic cross-consistency
  between structured fields and embedded YAML frontmatter) —
  `code-copilot-team/specs/wiki-ingest-pipeline/spec.md` §Interface.
  Reproduced as `rlmkit.wiki.ingest.validate_proposal`.
- **JSON-extraction strategy** (fenced ` ```json ` first, balanced-
  brace fallback) — same source. Reproduced as
  `rlmkit.wiki.backends._extract_json_object`.
- **Proposal file shape** (frontmatter keys: `proposal_kind`,
  `proposal_date`, `source_path`, `backend`, `ingestor_version`,
  `gate_disposition`, `gate_reason`, `target_slug`,
  `target_page_type`) — same source.
- **Exit-code semantics** (0 success, 2 backend-not-found,
  3 backend-failure, 4 contract-violation, 5 source-missing,
  6 output-write) — same source.
- **Schema-files-read-from-disk-at-runtime rule** — never embed
  schema text in source so the prompt stays in sync with whatever
  the schema currently says.

### Diverged (with rationale)

- **Linter is a Python module, not a Bash script.** `rlmkit.wiki.linter`
  vs `cct/knowledge/wiki/scripts/lint-wiki.sh`. Behavior matches the
  six rules in cct's `lint-rules.md`, but RLMKit's tests, CLI, and CI
  are all Python-native, so the port reduces context-switching cost
  for RLMKit contributors. Equivalence is intent; if the two diverge
  on an edge case, that is a bug.

- **Code under `src/rlmkit/wiki/`, not `scripts/wiki_ingest/`.** cct's
  pipeline lives under `scripts/` because cct is a methodology repo
  whose deliverable is shared skill files. RLMKit is a Python
  library; the wiki backbone is a first-class subpackage that other
  RLMKit code (including the future API layer in `src/rlmkit/api.py`)
  can import. Putting it under `src/` is the right home.

- **Three backend implementations vs one.** cct ships exactly one
  default backend (a copilot-CLI subprocess wrapper). RLMKit ships
  three: `TestBackend`, `LLMBackend` (single completion against any
  RLMKit `LLMClient`), and `RLMBackend` (recursive controller). The
  copilot-CLI subprocess backend is **not implemented** in RLMKit's
  v1 because RLMKit users already have an `LLMClient` — reusing it
  is simpler than spawning a sibling copilot. The cct backend is
  easy to add later; the protocol is the same.

- **`RLMBackend` has no parallel in cct.** This is the value-add the
  experiment is supposed to test: cct cannot drive the gate +
  draft loop with recursive code execution because cct does not
  ship a recursive controller. RLMKit does. `RLMBackend` exists to
  prove that wiring is real — see `tests/wiki/test_rlm_backend_e2e.py`.

- **Mode D query path has no parallel in cct.** cct's wiki is
  curator-driven (humans choose what to query); a programmatic
  `query_wiki(question, …, rlm=…)` with RLM fallback is RLMKit-
  specific. Same value-add: RLMKit's controller is the synthesis
  engine when wiki coverage is weak.

- **Output directory under `doc_internal/proposals/` is consistent
  with cct, but RLMKit's `/specs` is gitignored.** That is an
  RLMKit-side convention I did not change; the spec/plan are still
  written there per the global SDD rule and force-added during the
  session commit.

- **CLI ships only `--backend test` in v1.** The `LLMBackend` /
  `RLMBackend` paths require a configured `LLMClient` that the CLI
  cannot guess at safely. Programmatic users wire them via
  `Ingestor(backend=…)`. cct's CLI auto-detects copilot binaries on
  `PATH`; that is appropriate for a methodology repo and not for
  a library.

- **Stdlib-only on the wiki hot path.** Deliberately the same
  decision cct made (no `pyyaml`, no `jsonschema`) but the
  motivation is slightly different: in RLMKit, the wiki backbone
  must not pull a transitive dependency that complicates the
  `pyproject.toml` pin matrix.

## How RLMKit's recursive controller is wired into `wiki + rlm` mode

`RLMBackend.ingest()` does three things:

1. **Read the corpus** (file *or* directory). Directories are
   walked deterministically and concatenated with
   `=== <relpath> ===` headers so the model can address each
   source via `grep`/`peek`.
2. **Build a structured query** that includes:
   - the wiki curator persona instructions,
   - the four-question gate prose verbatim from
     `knowledge/wiki/schema/ingest-rules.md`,
   - the page-type templates verbatim,
   - the citation rules verbatim,
   - explicit instructions to return one JSON object as the FINAL
     answer.
3. **Hand off to `RLM.run(prompt=corpus, query=query)`.** The
   controller loads the corpus as `P` in the REPL environment and
   drives the loop. The model navigates with `peek`/`grep`/`subcall`,
   and when ready returns `FINAL: ```json …```` with the proposal.
4. **Parse and validate.** `_parse_proposal_json` extracts the
   JSON via the cct-borrowed two-strategy extractor; the Ingestor
   runs the two-layer validation; any contract violation raises
   `ContractViolation` (exit 4) with a precise message.

The end-to-end test `tests/wiki/test_rlm_backend_e2e.py` runs the
whole flow with a `MockLLMClient` so the assertion that
"RLMKit's recursive controller drives wiki ingest" is mechanically
verified without a live LLM.

## What this build does **not** do

- No real cloud-LLM ingest. The LLMBackend works against any
  `LLMClient`, but the test suite uses `MockLLMClient`; running it
  against real Claude / OpenAI is left to the operator.
- No copilot-CLI subprocess backend (cct's `claude → codex →
  cursor` autodetect). The protocol is open; a contributor can add
  it without changing the pipeline.
- No auto-merge of approved proposals into `knowledge/wiki/`.
  Identical to cct: human curator gates promotion.
- No reranker, no embeddings retrieval in Mode B. Keyword overlap
  is the v1 algorithm. Replacing it with a proper hybrid retriever
  is straightforward and out of scope for the 90-minute
  experiment.

## Acceptance status

- [x] `python -m rlmkit.wiki.cli ingest --backend test <fixture>`
      writes a valid proposal and exits 0.
- [x] `python -m rlmkit.wiki.cli lint` exits 0 against the seeded
      `knowledge/wiki/`.
- [x] Mode C E2E test passes with `MockLLMClient` driving an `RLM`
      controller — proves the recursive-controller wiring.
- [x] Two-layer validation rejects a draft whose embedded
      frontmatter disagrees with the structured fields.
- [x] No regression in existing rlmkit core tests
      (`tests/test_domain.py`, `tests/test_use_cases.py`,
      `tests/test_port_compliance.py` — 250/250 pass).
