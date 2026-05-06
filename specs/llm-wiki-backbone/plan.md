---
feature_id: llm-wiki-backbone
spec_mode: lightweight
status: draft
date: 2026-05-05
issue: gosha70/rlmkit#37
---

# Implementation Plan — LLM Wiki Backbone

## Context

This plan implements the spec at `specs/llm-wiki-backbone/spec.md`.
It is a single-session build (target ≤90 min); phases below are
sequential checkpoints, not parallel work streams.

## Phases

### Phase 1 — Schema seed + linter (≈15 min)

Files:
- `knowledge/wiki/schema/{page-types,ingest-rules,citation-rules,lint-rules,WIKI_MAINTAINER}.md`
  — copies of cct's schema with a leading attribution block.
- `knowledge/wiki/{index,overview,log}.md` — minimal seed pages so
  the wiki lints clean from day one.
- `src/rlmkit/wiki/__init__.py` (empty package marker).
- `src/rlmkit/wiki/schema.py` — frontmatter parser (awk-style port),
  `PAGE_TYPE_DIRS` map, `parse_frontmatter()` helper.
- `src/rlmkit/wiki/linter.py` — Python port of `lint-wiki.sh`. Six
  checks. Exposes `lint_wiki(wiki_dir: Path) -> LintReport`.

Verification: `python -m rlmkit.wiki.cli lint` (or programmatic
`lint_wiki(...)`) exits 0 against the seeded tree.

### Phase 2 — Ingest pipeline + TestBackend + LLMBackend (≈25 min)

Files:
- `src/rlmkit/wiki/proposal.py` — `IngestRequest`,
  `IngestProposal`, `render_proposal_file()` helper.
- `src/rlmkit/wiki/backends.py` — `IngestBackend` Protocol;
  `TestBackend` (deterministic, derives slug/page_type from H1);
  `LLMBackend` (single completion against any `LLMClient`).
- `src/rlmkit/wiki/ingest.py` — `Ingestor` orchestrator: composes
  prompt from schema excerpts, dispatches to backend, runs the
  two-layer validation, writes proposal file.
- `src/rlmkit/wiki/errors.py` — exception hierarchy mapped to exit
  codes (`BackendNotFound`, `BackendFailure`, `ContractViolation`,
  `SourceMissing`, `OutputWriteFailure`).
- `tests/wiki/test_proposal.py`, `tests/wiki/test_linter.py`,
  `tests/wiki/fixtures/sample-incident.md`.

Verification: `pytest tests/wiki/test_proposal.py
tests/wiki/test_linter.py` passes.

### Phase 3 — RLM backend (Mode C, the differentiator) (≈25 min)

Files:
- `src/rlmkit/wiki/backends.py` — extend with `RLMBackend(rlm: RLM)`.
  Loads the source corpus as the RLM `P` variable, system prompt
  instructs the model to "apply the four-question gate, draft a
  typed wiki page, return JSON". The model uses `peek`/`grep` to
  navigate large content. Final answer parsed as the
  `IngestProposal` JSON.
- `src/rlmkit/wiki/cli.py` — `python -m rlmkit.wiki.cli` with
  subcommands `ingest`, `query`, `lint`.
- `scripts/wiki-ingest`, `scripts/wiki-query`, `scripts/wiki-lint`
  — Bash entrypoints that exec into the Python module.
- `tests/wiki/test_rlm_backend.py` — end-to-end Mode C test using
  `MockLLMClient` to script the recursive controller's responses.

Verification: `pytest tests/wiki/test_rlm_backend.py` passes; the
test asserts the `IngestProposal` round-trips through the
controller and the validator.

### Phase 4 — Query path with RLM fallback (Mode D) (≈10 min)

Files:
- `src/rlmkit/wiki/query.py` — `query_wiki(question, wiki_dir,
  raw_dir, rlm)` per spec. Wiki-first scoring (simple keyword/slug
  overlap; reranking is out of scope for v1). RLM fallback when
  the LLM signals "insufficient information".
- `tests/wiki/test_query.py` — covers the wiki-first hit path and
  the RLM-fallback path.

Verification: `pytest tests/wiki/` passes; wiki linter still clean.

### Phase 5 — Wrap (≈10 min)

- Run `pytest -q tests/wiki/`.
- Run rlmkit's existing test suite to verify no regression
  (`pytest -q tests/test_domain.py tests/test_use_cases.py` —
  fastest layer; full suite if budget allows).
- `DESIGN.md` and `RESULT.md` written at repo root.
- `.gitignore` entry for `doc_internal/proposals/`.
- Commit per phase.

## Out of scope (deferred to a v2)

- Auto-merge of approved proposals back into `knowledge/wiki/`.
- A real reranker for Mode B (currently keyword-overlap only).
- Multi-source synthesis driven by Mode C as a "build the whole
  wiki from a directory" command. v1 ingests one source per run;
  Mode C just brings the recursive controller to that one run.
- Concrete copilot-CLI subprocess backend (cct's
  `claude → codex → cursor` autodetect). The protocol is open;
  contributors can add it without changing the pipeline.
- A web UI / Streamlit POC for browsing the wiki.

## Risks

- **Two-layer validation parser drift.** Hand-rolled YAML
  frontmatter parsing must match cct's well enough that pages
  written for cct also pass RLMKit's linter. Mitigation: borrow
  the awk-style logic verbatim and add a parity test against one
  cct fixture.
- **MockLLMClient scripting fragility.** The Mode C test scripts
  RLM responses in order. If the controller's prompt template
  changes, the test brittle-fails. Mitigation: keep the test
  focused on the proposal output, not on the exact intermediate
  REPL exchanges.
- **Time budget.** 90-minute target is tight given the breadth of
  modes A–D. Phase 3 is the value-add; if budget squeezes, defer
  Phase 4 (Mode D) before Phase 3.
