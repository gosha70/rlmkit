---
id: llm-wiki-backbone
status: draft
spec_mode: feature
authors: autonomous-build-agent
created: 2026-05-05
issue: gosha70/rlmkit#37
---

# LLM Wiki Backbone for RLMKit

## 1. Problem statement

When RLMKit is pointed at large or heterogeneous corpora, naive retrieval and
single-shot prompt synthesis degrade as the model is forced to rediscover
context on every query. The "LLM Wiki" pattern (Karpathy gist 442a6bf555914893e9891c11519de94f)
addresses this by maintaining a distilled, persistent markdown knowledge layer
built incrementally from raw sources. Today, RLMKit has `direct`, `rag`, `rlm`,
`compare`, and `auto` modes — but no first-class workflow for **compiling and
maintaining** a durable, queryable wiki layer derived from raw inputs.

This spec adds an **LLM Wiki backbone** that:

1. ingests raw sources into a persistent on-disk wiki layer,
2. answers queries against that wiki first, falling back to raw analysis only
   when wiki coverage is insufficient,
3. uses RLMKit's recursive controller as a synthesis backend when corpora
   exceed what a single prompt window can absorb.

## 2. Goals / non-goals

### Goals
- Ship a runnable, deterministic wiki workflow within rlmkit's existing
  Clean-Architecture conventions (domain → application → infrastructure).
- Add two new execution modes — `wiki` and `wiki_rlm` — alongside the existing
  `direct`, `rag`, `rlm` modes. They are first-class strategies, callable from
  the same evaluator harness.
- Provide bounded, idempotent operations: **ingest**, **query**, **lint**,
  **promote**, **update**.
- Persist the wiki as plain markdown on disk (`knowledge/wiki/`) so it is
  human-readable, diffable in git, and survives without the running service.
- Stay LLM-provider-agnostic: every operation accepts an `LLMClient` Protocol
  instance, so a deterministic stub backend can drive end-to-end tests.

### Non-goals (this iteration)
- A bespoke vector index for the wiki — wiki retrieval reuses rlmkit's existing
  `OpenAIEmbedder` / chunker via the same plumbing as `RAGStrategy`.
- A UI surface (Streamlit / Next.js). The spec defines APIs and CLI hooks; UI
  is follow-up.
- Distributed wiki state, multi-writer concurrency, or transactional updates —
  the wiki is a single-writer markdown tree; concurrency is out of scope.
- Auto-promotion of every answer back into the wiki. v1 supports a
  user-invoked `promote` operation; auto-promotion is a follow-up policy
  layer.

## 3. Implementation directions (issue #37 mapping)

### A. Modes / backends
Two new strategy classes register under `rlmkit/strategies/`:

| Mode name (constant) | Behavior |
|---|---|
| `wiki` | Query-only: answer from the persisted wiki layer using the existing RAG retriever scoped to `knowledge/wiki/`. No raw-source lookup. |
| `wiki_rlm` | Wiki-first then RLM fallback: try `wiki`; if confidence is below a configurable threshold or coverage is reported missing, hand off to the RLM controller scoped to the linked `knowledge/raw/` sources. |

`wiki` and `wiki_rlm` join `MODE_DIRECT`, `MODE_RAG`, `MODE_RLM` in
`sandbox_vars.py`. They are constants — never magic strings — per the project's
"no magic strings" rule.

`auto` mode is **not** modified in this iteration. It can prefer wiki later
once the wiki layer has been observed in the wild; that is a follow-up policy
decision.

### B. Raw → wiki → query pipeline
On-disk layout (single-source-of-truth: this directory):

```
knowledge/
  raw/                          # mirrors of original sources
    <source-id>.md              # one markdown file per ingested source
  wiki/
    index.md                    # auto-generated table of contents
    log.md                      # append-only history of ingest / promote events
    overview.md                 # corpus-level summary (regenerated on bulk ingest)
    concepts/<slug>.md          # concept pages
    workflows/<slug>.md         # how-to / runbook style
    incidents/<slug>.md         # post-incident write-ups
    decisions/<slug>.md         # ADR-style decision records
    playbooks/<slug>.md         # operator-facing playbooks
    glossary/<slug>.md          # term definitions
  schema/
    page_schema.yaml            # required frontmatter fields for wiki pages
    ingest_rules.md             # what gets ingested, what gets dropped
    citation_rules.md           # how wiki pages cite raw sources
    lint_rules.md               # consistency / link rules
```

Page frontmatter (required, validated by lint):

```yaml
---
title: <human-readable title>
type: concept | workflow | incident | decision | playbook | glossary | overview
slug: <kebab-case>
sources: [<raw-source-id>, ...]   # back-pointers to knowledge/raw/
status: draft | active | stale
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

### C. Bounded operations

Each operation is a pure function over the wiki tree plus an `LLMClient`. They
are exposed both as Python entry points (under `rlmkit.wiki`) and as a CLI
group (`rlmkit wiki <op>` — wired in plan, not all in this iteration).

- **`ingest(source_path, *, client, root)`** — read raw file → mirror under
  `knowledge/raw/` → ask LLM to draft / update wiki pages → write pages →
  append to `log.md`.
- **`query(question, *, client, root, mode='wiki')`** — embed question →
  retrieve top-k wiki chunks → call LLM with `wiki` system prompt → return a
  `StrategyResult` with citations to wiki page slugs **and** raw source IDs.
- **`lint(root)`** — pure-Python checks: required frontmatter fields,
  resolvable links, no orphan pages, no stale `sources` references. Returns a
  `LintReport` (pass/fail + list of `LintIssue`).
- **`promote(answer, sources, *, client, root, page_type)`** — turn an answer
  produced by another mode (typically `wiki_rlm` after RLM fallback) into a
  new or updated wiki page.
- **`update(source_path, *, client, root)`** — re-run ingest for one source,
  diff resulting pages, log changes.

### D. RLMKit-for-scale

`wiki_rlm` mode invokes the existing `RLMStrategy` when:

- the wiki returned no chunks above the relevance threshold, **or**
- the LLM's wiki-mode answer reports `coverage: insufficient` in its structured
  reply, **or**
- the request explicitly requests a refresh.

The RLM controller is given the **raw** sources linked from the candidate
wiki pages (or the full `knowledge/raw/` directory if there is no candidate),
not the wiki itself. The RLM answer can then be promoted back into the wiki
via `promote()`, closing the loop.

## 4. User scenarios

1. **Bulk ingest.** A user drops 200 markdown files into `knowledge/raw/`,
   runs `rlmkit wiki ingest --all`, and ends up with a populated
   `knowledge/wiki/` plus an auto-generated `index.md`.
2. **Wiki-first query.** User asks "What is our chunking strategy?" via the
   Python API in `wiki` mode; the wiki has a `concepts/chunking.md`, the
   answer cites it.
3. **Wiki-fallback query.** User asks a question the wiki does not cover.
   `wiki_rlm` mode falls back to RLM over the raw sources, returns an answer,
   and the user invokes `promote()` to durably add a new concept page.
4. **Lint after change.** User edits a page by hand; `lint()` flags missing
   frontmatter, broken links, or orphaned pages before the next commit.

## 5. Architectural fit

The wiki layer follows rlmkit's existing layering:

- **Domain (`rlmkit/domain/wiki.py`):** dataclasses for `WikiPage`,
  `WikiCitation`, `LintIssue`, `LintReport`, `IngestResult`. Pure stdlib. No
  filesystem, no LLM.
- **Application (`rlmkit/application/use_cases/wiki/`):** orchestration —
  `IngestSourceUseCase`, `QueryWikiUseCase`, `LintWikiUseCase`,
  `PromoteAnswerUseCase`. Depends only on Protocols (LLMClient, Embedder,
  filesystem port).
- **Infrastructure (`rlmkit/infrastructure/wiki/`):**
  `MarkdownWikiRepository` (the on-disk implementation),
  `FrontmatterParser`, `WikiLinter`. Concrete I/O lives here.
- **Strategies (`rlmkit/strategies/wiki.py`, `rlmkit/strategies/wiki_rlm.py`):**
  thin adapters that satisfy the existing `LLMStrategy` Protocol so the
  evaluator harness picks them up automatically.
- **Prompts (`prompts/wiki/*.yaml`):** versioned YAML — `wiki_query.yaml`,
  `wiki_ingest.yaml`, `wiki_promote.yaml`. **No inline prompt strings.**

## 6. Schema & taxonomy decisions (curator-deferred → decided)

- **Directory:** `knowledge/` at repo root (sibling of `src/`). Picked over
  `data/` because the issue explicitly names `knowledge/`, and it stays human-
  curated rather than treated as runtime data.
- **Mode names:** `wiki` and `wiki_rlm`. Underscore form to match
  `MODE_RAG`, `MODE_RLM`. The issue's `wiki + rlm` notation maps to
  `wiki_rlm`.
- **Page taxonomy:** the six issue types — concepts, workflows, incidents,
  decisions, playbooks, glossary — plus a top-level `overview.md` and an
  auto-generated `index.md` and `log.md`. No additional types in v1.
- **Validator strictness:** lint is **strict on frontmatter** (missing
  required field is an error) and **warn-only on link rot** (broken link →
  warning). Rationale: frontmatter is machine-consumed; links can lag a
  rename for a moment.
- **RLM invocation:** `wiki_rlm` constructs an `RLMStrategy` configured with
  the default `RLMConfig`, and feeds it the concatenated raw sources linked
  from the candidate wiki pages (or all raws if no candidate). The RLM's
  answer is wrapped in a `StrategyResult` with `metadata.fallback = "rlm"`.

## 7. Test strategy

- **Unit tests** for the linter, frontmatter parser, and on-disk repository
  (using `tmp_path` fixtures).
- **One end-to-end test** that runs ingest → query → promote with a
  deterministic stub LLM client (`StubLLMClient`) and a deterministic stub
  embedder (`StubEmbedder`) so the test does not depend on any provider key.
- All tests use rlmkit's existing `pytest` config; `uv run pytest tests/wiki/`
  is the entry point.

## 8. Risks

- **Prompt drift.** The wiki ingest prompt determines page quality. We accept
  this and pin the prompt under `prompts/wiki/` with a `version:` field.
- **Schema sprawl.** Adding more page types later may invalidate existing
  pages. Mitigation: lint enforces only the v1 types; new types require an
  ADR.
- **Coverage detection.** The wiki-fallback heuristic is naive (top-k score
  threshold + structured `coverage` field). Real-world calibration is a
  follow-up.

## 9. Out-of-scope (explicit follow-ups)

- CLI `rlmkit wiki` group (only the Python API ships in v1).
- Automatic wiki refresh on raw-source change (file watcher).
- Web UI for browsing the wiki.
- Multi-language / non-markdown wiki pages.
- Auto-promotion from `wiki_rlm` runs.
