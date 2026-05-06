# LLM Wiki backbone — design notes

This document captures the decisions made while implementing rlmkit#37
("LLM Wiki / Knowledge Base compilation feature"). The full spec lives at
`specs/llm-wiki-backbone/spec.md`; this file is the human-friendly summary.

## Modes

Two new modes are added alongside `direct`, `rag`, `rlm`, `compare`, `auto`:

| Mode constant | Behavior |
|---|---|
| `MODE_WIKI = "wiki"` | Query-only against the persistent wiki layer. `content` arg ignored — the wiki on disk is the source of truth. |
| `MODE_WIKI_RLM = "wiki_rlm"` | Wiki-first; if coverage is `missing` (or `partial` and `fallback_on_partial=True`), the existing `RLMStrategy` runs over the linked raw sources and the result is wrapped with fallback metadata. |

`auto` mode is intentionally **not** modified yet — calibrating when `auto`
should prefer `wiki` is a follow-up policy decision once real corpora exist.

## Directory layout

```
knowledge/
  raw/                          mirror of every ingested source
    <source-id>.md
  wiki/
    index.md                    auto-generated table of contents
    log.md                      append-only ingest / promote log
    overview.md                 (optional) corpus-level summary page
    concepts/<slug>.md
    workflows/<slug>.md
    incidents/<slug>.md
    decisions/<slug>.md
    playbooks/<slug>.md
    glossary/<slug>.md
  schema/
    page_schema.yaml            required frontmatter
    ingest_rules.md             what gets ingested (and what doesn't)
    citation_rules.md           how pages cite raw sources
    lint_rules.md               lint codes + severities
```

I picked `knowledge/` at repo root over `data/` because issue #37 explicitly
names that path, and because the wiki is human-curated, not runtime data.

## Schema

Every wiki page carries this frontmatter (lint rejects pages without it):

```yaml
title: <human-readable>
slug:  <kebab-case>
type:  concept | workflow | incident | decision | playbook | glossary | overview
sources: [<raw-source-id>, ...]
status: draft | active | stale
created: YYYY-MM-DD
updated: YYYY-MM-DD
```

`type` controls which subdirectory the page lives in. `overview` is the only
type that lives at the wiki root (`wiki/overview.md`).

## Operations

All four operations are bounded, idempotent on re-run, and Protocol-driven:

| Use case | What it does |
|---|---|
| `IngestSourceUseCase.execute(source_id, content)` | Mirrors raw → asks LLM for YAML page list → writes pages → rebuilds index → appends log entry. |
| `QueryWikiUseCase.execute(question)` | Embeds query + page corpus, ranks by cosine, builds context, calls LLM. Reply must start with `COVERAGE: full|partial|missing`; `WikiQueryResult` exposes the parsed coverage and citations. |
| `LintWikiUseCase.execute()` | Pure validation — frontmatter is **strict** (errors block), link rot and stale source pointers are **warnings**. Six lint codes: `FM001`–`FM003`, `LK001`, `SR001`, `OP001`. |
| `PromoteAnswerUseCase.execute(...)` | Wraps a free-form answer (typically from a `wiki_rlm` fallback) as a structured `WikiPage` and writes it. |

## How rlmkit features integrate

- **LLMClient Protocol.** All wiki use cases accept any `LLMClient`. The
  existing `LiteLLMAdapter` works; tests use a `StubLLMClient`.
- **EmbeddingProvider Protocol.** Query reuses `OpenAIEmbedder` (or any
  `EmbeddingProvider`); tests use `StubEmbedder`.
- **RLM controller.** `WikiRLMStrategy` constructs an `RLMStrategy` with the
  default `RLMConfig` and feeds it the concatenated raw sources linked from
  the candidate wiki pages. The RLM result is wrapped with
  `metadata["fallback_backend"] = "rlm"` so callers can distinguish wiki
  answers from RLM-derived answers.
- **Strategy registry.** `WikiStrategy` and `WikiRLMStrategy` register in
  `rlmkit.strategies.__init__`, so the existing `MultiStrategyEvaluator`
  picks them up by name.
- **Prompts.** Every wiki prompt lives in versioned YAML under
  `src/rlmkit/prompts/wiki/`. `prompts/wiki/__init__.py` exposes
  `get_wiki_prompt(name)` with an `lru_cache`. No inline prompt strings —
  this matches CLAUDE.md's hard rule.
- **Constants.** `MODE_WIKI`, `MODE_WIKI_RLM`, lint codes, coverage values,
  page-type-to-dir mapping all live as module-level constants. No magic
  strings cross module boundaries.
- **Layering.** Files map cleanly onto rlmkit's existing Clean Architecture:
  `domain/wiki.py` (stdlib only) → `application/ports/wiki_port.py` and
  `application/use_cases/wiki/` → `infrastructure/wiki/`. Strategies are
  thin adapters over the use cases.

## Coverage detection

The wiki query system prompt forces the LLM to begin its answer with one of
`COVERAGE: full | partial | missing`. `_parse_coverage_header` strips that
header before returning the body. `WikiRLMStrategy` reads the parsed
coverage to decide whether to fall back. This is a deliberately simple
heuristic — production calibration is a follow-up.

## Decisions made (curator-deferred)

- **Directory:** `knowledge/` (not `data/`).
- **Mode names:** underscored — `wiki`, `wiki_rlm` (matches `rag`, `rlm`).
- **Page taxonomy:** the six issue types + `overview`. No new types in v1.
- **Validator strictness:** strict frontmatter, warn-only on link rot and
  stale sources.
- **RLM invocation:** default `RLMConfig`, raw inputs are concatenated from
  the candidate pages' `sources` field (or all raws if no candidate).
- **Prompts location:** `src/rlmkit/prompts/wiki/*.yaml` (consistent with
  the existing `prompts/` package data).
- **Citation form:** inline `[<slug>]` for wiki pages, `[<source-id>]` for
  raw inputs.

## Punted to follow-up

- CLI (`rlmkit wiki ingest|query|lint|promote`). The Python API ships in v1.
- Streamlit / Next.js UI for browsing the wiki.
- Auto-promotion of `wiki_rlm` fallback answers (today the user calls
  `PromoteAnswerUseCase` explicitly).
- Filesystem-watcher-driven incremental re-ingest on raw source change.
- Coverage-threshold calibration on real corpora.
- `auto` mode preference for `wiki` once the wiki has been observed in use.
- Migration scripts for adding new page types (would require an ADR per
  CLAUDE.md's "graph schema changes via migration scripts" rule).

## File-level summary

```
src/rlmkit/
  domain/wiki.py                                          [NEW] stdlib-only entities
  application/ports/wiki_port.py                          [NEW] WikiRepositoryPort
  application/use_cases/wiki/__init__.py                  [NEW]
  application/use_cases/wiki/ingest.py                    [NEW]
  application/use_cases/wiki/query.py                     [NEW] WikiQueryResult + COVERAGE
  application/use_cases/wiki/lint.py                      [NEW] LintWikiUseCase + codes
  application/use_cases/wiki/promote.py                   [NEW]
  application/sandbox_vars.py                             [EDIT] +MODE_WIKI / MODE_WIKI_RLM
  infrastructure/wiki/__init__.py                         [NEW]
  infrastructure/wiki/markdown_repository.py              [NEW] tmp_path-friendly
  infrastructure/wiki/frontmatter.py                      [NEW] YAML parse/serialize
  infrastructure/wiki/index_writer.py                     [NEW]
  strategies/wiki.py                                      [NEW] WikiStrategy
  strategies/wiki_rlm.py                                  [NEW] WikiRLMStrategy
  strategies/__init__.py                                  [EDIT] register
  prompts/wiki/__init__.py                                [NEW] get_wiki_prompt
  prompts/wiki/wiki_query.yaml                            [NEW]
  prompts/wiki/wiki_ingest.yaml                           [NEW]
  prompts/wiki/wiki_promote.yaml                          [NEW]
pyproject.toml                                            [EDIT] package-data for wiki YAML
knowledge/wiki/{index,log}.md                             [NEW]
knowledge/wiki/{concepts,workflows,...}/                  [NEW] empty dirs
knowledge/schema/{page_schema.yaml,ingest_rules.md,
                  citation_rules.md,lint_rules.md}        [NEW]
specs/llm-wiki-backbone/spec.md                           [NEW]
specs/llm-wiki-backbone/plan.md                           [NEW]
tests/wiki/                                               [NEW] 19 tests, all stubbed
DESIGN.md                                                 [NEW] this file
RESULT.md                                                 [NEW]
```

Total: 19 wiki tests pass; pre-existing 250 sample tests still pass.
