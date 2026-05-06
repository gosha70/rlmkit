---
feature_id: llm-wiki-backbone
spec_mode: lightweight
status: draft
issue: gosha70/rlmkit#37
date: 2026-05-05
---

# LLM Wiki Backbone — Spec

## Problem

RLMKit today exposes three execution modes (`direct`, `rag`, `rlm`,
plus `compare`) that all operate against an *ad-hoc* prompt. None of
them maintain a **persistent, distilled knowledge layer** between
sessions. As corpora grow (large repos, postmortem archives,
multi-document research dumps), naive RAG and direct prompting both
degrade: RAG retrieves chunks without higher-order organization, and
direct prompting blows the context window.

[`gosha70/rlmkit#37`](https://github.com/gosha70/rlmkit/issues/37)
asks for a **first-class LLM-Wiki feature** — a curated markdown
knowledge layer compiled from raw sources, queried first, with
RLMKit's recursive controller used as the scalable reasoning engine
when corpora exceed what flat retrieval can handle.

The reference design is the LLM Wiki pipeline already shipped in the
sibling repo at `/Users/gosha/dev/repo/code-copilot-team` (PRs #12,
#26, #28). That pipeline operates on a single source per invocation,
emits proposals to `doc_internal/proposals/`, and leaves human
approval gating. **What it does not do — and what RLMKit can — is
recursive synthesis across many large sources.** That gap is this
spec's value-add.

## Non-goals

- Auto-merging proposals into `knowledge/wiki/`. Human approval
  remains gating, identical to cct's design.
- Replacing RLMKit's existing `rag` or `direct` modes. The wiki is a
  complement, not a substitute.
- A web UI or rich review surface in v1. Markdown on disk + git is
  the review surface.
- Concrete cloud-LLM SDK adapters specific to the wiki. Wiki ingest
  reuses RLMKit's existing `LLMClient` Protocol and any provider
  already wired in (LiteLLM, OpenAI, Anthropic, Ollama, …).

## User scenarios — modes A–D from issue #37

### Mode A — `direct` wiki ingest (small corpus, single source)
Curator passes one source file. Pipeline applies the four-question
gate via a single LLM call, drafts a typed page, writes proposal to
`doc_internal/proposals/`. Equivalent to cct's `wiki-ingest`.

### Mode B — `rag` wiki query (wiki-first retrieval)
Caller asks a question. Pipeline retrieves matching wiki pages by
slug/title/keyword first; only falls back to raw-source RAG when
wiki coverage is weak.

### Mode C — `rlm_wiki` / `wiki + rlm` mode (the differentiator)
Curator passes a **directory** of raw sources (or one very large
source). Instead of a single LLM call, RLMKit's recursive controller
loads the corpus as the `P` variable in a REPL, lets the model
explore it via `peek` / `grep` / `subcall`, and emits one or more
typed wiki proposals. This is the path that requires RLMKit's
recursive controller and that cct's pipeline cannot do.

### Mode D — `wiki` query with RLM fallback
Caller asks a question; wiki-first lookup misses. Pipeline runs the
recursive controller against the raw source corpus to synthesize an
answer, then offers to **promote** the synthesized answer back into
the wiki as a new proposal (closing the loop from #37 §C).

## Interface

### Domain types (`rlmkit.wiki.proposal`)

```python
@dataclass(frozen=True)
class IngestRequest:
    source_path: Path                   # file or directory
    mode: Literal["direct", "rlm"]      # wiki ingest mode
    backend_name: str                   # "test" | "llm" | "rlm"

@dataclass(frozen=True)
class IngestProposal:
    disposition: Literal["accept", "reject"]
    reason: str
    page_type: str | None
    slug: str | None
    title: str | None
    draft_markdown: str | None
    sources: list[dict]
```

### Backend protocol (`rlmkit.wiki.backends`)

```python
class IngestBackend(Protocol):
    name: str
    def ingest(self, request: IngestRequest, schema_excerpts: dict[str, str]) -> IngestProposal: ...
```

Three concrete backends in v1:

1. **`TestBackend`** — deterministic, derives the proposal from the
   source's first H1. Used by tests and `--backend test`. No LLM
   call.
2. **`LLMBackend`** — single LLM call against any `LLMClient`
   (RLMKit's existing Protocol). Equivalent to cct's CLI-subprocess
   backend, but uses RLMKit's own provider stack.
3. **`RLMBackend`** — wraps an `RLM` controller. Loads the source
   corpus into the REPL as `P`, asks the model to apply the gate
   and draft the page using `peek`/`grep`/`subcall` to navigate
   large content. Returns the same `IngestProposal` type.

### Query protocol (`rlmkit.wiki.query`)

```python
@dataclass(frozen=True)
class WikiQueryResult:
    answer: str
    pages_consulted: list[str]          # wiki slugs hit
    fell_back_to_rlm: bool              # True if Mode D was used
    rlm_trace: list[dict] | None
```

```python
def query_wiki(
    question: str,
    wiki_dir: Path,
    *,
    raw_dir: Path | None = None,
    rlm: RLM | None = None,
) -> WikiQueryResult
```

Algorithm:
1. **Wiki-first.** Walk `wiki_dir`, score pages against the
   question (slug/title/H1/keyword overlap), assemble top-k page
   bodies, ask the LLM (via `rlm.client`) to answer from them.
2. **Coverage check.** If the LLM's answer says "insufficient
   information" (a sentinel phrase the prompt requests when
   appropriate) **and** `raw_dir` + `rlm` are available, escalate
   to Mode D: run `rlm.run(prompt=raw_corpus, query=question)`.
3. Return both the answer and the trail (pages consulted; whether
   Mode D fired; the RLM trace if it did).

### CLI surface (`scripts/wiki-ingest`, `scripts/wiki-query`,
`scripts/wiki-lint`)

```
scripts/wiki-ingest <source>                    # default: LLMBackend
scripts/wiki-ingest --backend test <source>     # deterministic test
scripts/wiki-ingest --backend rlm <dir>         # Mode C: recursive
scripts/wiki-ingest --output-dir <dir>          # override
scripts/wiki-query "<question>"                 # Mode B / D
scripts/wiki-lint                               # run linter
```

### Output

Identical to cct: one proposal file per invocation under
`doc_internal/proposals/<YYYY-MM-DD>-<slug>.md`. Frontmatter:
`proposal_kind`, `proposal_date`, `source_path`, `backend`,
`ingestor_version`, `gate_disposition`, `gate_reason`,
`target_slug`, `target_page_type`. Body for accept = full wiki page
markdown (frontmatter + body). Body for reject = gate reasoning.

### Error semantics

Reuse cct's exit codes for CLI parity:
- 0 = success (accept or reject; both are pipeline successes)
- 2 = backend not found
- 3 = backend invocation failure
- 4 = contract violation (proposal fails two-layer validation)
- 5 = source missing
- 6 = output write failure

## Validation — two-layer (borrowed from cct)

1. **Shape.** `IngestProposal` fields match expected types.
2. **Semantic cross-consistency.** When `disposition == "accept"`,
   the embedded YAML frontmatter inside `draft_markdown` must agree
   with the structured `page_type` / `slug` / `title` / `sources`,
   `slug` must be kebab-case, and `(page_type, slug)` must satisfy
   the directory-placement rule the linter would enforce.

## Schema — page types, citations, lint

The wiki schema is **borrowed verbatim** from cct's
`knowledge/wiki/schema/` directory (`page-types.md`,
`citation-rules.md`, `ingest-rules.md`, `lint-rules.md`,
`WIKI_MAINTAINER.md`) with attribution. RLMKit adopts the identical
page-type taxonomy, the identical four-question gate, the identical
universal frontmatter, and the identical `sources:` rules. See
`DESIGN.md` for the borrow/divergence ledger.

Divergence: the linter is a Python port (`rlmkit.wiki.linter`)
rather than a Bash script, because RLMKit's tests, CLI, and CI are
all Python-native. Behavior is intended to match cct's
`lint-wiki.sh` rule-for-rule; equivalence is tested.

## Requirements

1. `rlmkit.wiki` Python package under
   `src/rlmkit/wiki/{__init__,proposal,schema,backends,linter,query,cli}.py`,
   stdlib-only for the hot path; no new third-party deps.
2. `IngestBackend` Protocol with three implementations
   (`TestBackend`, `LLMBackend`, `RLMBackend`). The RLM backend
   wires `rlmkit.core.rlm.RLM` to drive the gate + draft loop.
3. Two-layer validation as described above; failures raise a
   typed `ContractViolation` exception.
4. `rlmkit.wiki.linter` reproducing the six checks from cct's
   `lint-rules.md` (frontmatter, page_type, slug, directory, link
   integrity, orphans). `bash`-script parity is non-blocking but
   intent.
5. `rlmkit.wiki.query.query_wiki` implementing wiki-first retrieval
   with optional RLM fallback (Mode D).
6. CLI scripts at `scripts/wiki-ingest`, `scripts/wiki-query`,
   `scripts/wiki-lint`. Exit codes documented in `--help`.
7. `knowledge/` directory at repo root with the cct-borrowed schema
   files (with attribution), an empty wiki tree
   (`index.md`, `overview.md`, `log.md`, plus type subdirs), and
   `raw/` for ingestion input.
8. **One end-to-end test** using the `TestBackend`: ingest a
   committed fixture source, assert the proposal lints clean
   against `rlmkit.wiki.linter` if dropped into the wiki tree.
   **Plus** one test that exercises Mode C end-to-end with a
   `MockLLMClient` driving an `RLM` controller — proves the
   recursive-controller wiring works without a live LLM.
9. `doc_internal/proposals/` gitignored.
10. `DESIGN.md` and `RESULT.md` at repo root (this session's
    deliverable).

## Constraints

- **No automatic merge into `knowledge/wiki/`.** Proposals only.
- **Stdlib-only on the wiki hot path.** No `pyyaml`, no
  `jsonschema`. Use cct's awk-style frontmatter parser ported to
  Python.
- **Reuse RLMKit primitives.** The wiki module imports `RLM`,
  `RLMConfig`, `LLMClient` from `rlmkit.core` / `rlmkit.llm`. No
  parallel LLM stack.
- **Backwards compatibility.** Existing `interact()` /
  `compare_matrix()` / `RLM.run()` paths must not regress.
- **No edits inside `code-copilot-team/`.** Read-only substrate.

## Acceptance criteria

- [ ] `python -m rlmkit.wiki.cli ingest --backend test <fixture>` exits 0,
      writes a proposal file with the documented frontmatter.
- [ ] The end-to-end Mode C test (`MockLLMClient` + `RLM`) emits a
      proposal whose embedded frontmatter passes the two-layer
      validation.
- [ ] `python -m rlmkit.wiki.cli lint` exits 0 against the seeded
      empty-but-valid `knowledge/wiki/` tree.
- [ ] `pytest tests/wiki/` exits 0.
- [ ] `DESIGN.md` records every cct borrow and every divergence
      with a one-line rationale.

## Sources

- `issue: gosha70/rlmkit#37` — feature request driving this spec.
- `path: code-copilot-team/specs/wiki-ingest-pipeline/spec.md` —
  v1 ingest pipeline reference design (single-source).
- `path: code-copilot-team/knowledge/wiki/schema/page-types.md` —
  borrowed page-type taxonomy.
- `path: code-copilot-team/knowledge/wiki/schema/ingest-rules.md` —
  borrowed four-question gate.
- `path: code-copilot-team/knowledge/wiki/schema/citation-rules.md` —
  borrowed citation rules.
- `path: code-copilot-team/knowledge/wiki/schema/lint-rules.md` —
  borrowed lint rules.
- `path: code-copilot-team/knowledge/wiki/scripts/lint-wiki.sh` —
  reference implementation for the Python linter port.
- `url: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f`
  retrieved: 2026-05-05 — the original LLM Wiki framing referenced
  by issue #37.
