# LLM Wiki Backbone — Design

Implementation of `gosha70/rlmkit#37`: Karpathy-pattern LLM Wiki
maintainer wired into RLMKit as **first-class strategies** in the
existing strategy registry.

## Issue #37 modes A–D — how they map

### Mode A — wiki / wiki + rlm modes

Two new strategies registered in `src/rlmkit/strategies/__init__.py`
alongside `DirectStrategy`, `RAGStrategy`, `RLMStrategy`,
`IndexedRAGStrategy`:

| Mode constant   | Strategy class    | Behaviour |
|-----------------|-------------------|-----------|
| `MODE_WIKI`     | `WikiStrategy`    | Reads `index.md`, follows links to top-N relevant pages, asks the configured `WikiBackend` for an answer + citations. Pages-loaded list is recorded in `doc_internal/wiki-query-log.jsonl`. |
| `MODE_WIKI_RLM` | `WikiRLMStrategy` | Runs `WikiStrategy.query` first; if the answer is empty / citations are below `min_citations` / `force_rlm=True`, falls back to `rlmkit.core.rlm.RLM` with the loaded wiki pages joined as the document substrate. The recursive controller's full budget / sandbox / step-limit machinery applies. |

Both classes satisfy the existing `LLMStrategy` protocol so they
slot directly into `MultiStrategyEvaluator` for comparison runs.

### Mode B — raw → wiki → query directory layout

```
knowledge/
  wiki/
    index.md
    log.md
    overview.md
    concepts/
    workflows/
    incidents/
    decisions/
    playbooks/
    glossary/
    open-questions/
    schema/                         # bundled at src/rlmkit/strategies/wiki/schema/
```

Schema files (`page-types.md`, `ingest-rules.md`,
`citation-rules.md`, `lint-rules.md`) are **borrowed verbatim**
from `code-copilot-team/knowledge/wiki/schema/` so pages produced
by `rlmkit wiki ingest` are wire-compatible with cct's
`./scripts/wiki ingest`.

### Mode C — bounded operations

All four operations are reachable via the new
`rlmkit wiki <verb>` CLI (and `python -m rlmkit.cli.wiki <verb>`)
and via the strategy classes:

| Verb     | CLI                         | Module                                                       |
|----------|-----------------------------|--------------------------------------------------------------|
| ingest   | `rlmkit wiki ingest <src>`  | `src/rlmkit/strategies/wiki/ingestor.py::ingest`             |
| promote  | `rlmkit wiki promote <dir>` | `src/rlmkit/strategies/wiki/promoter.py::promote`            |
| query    | `rlmkit wiki query "..."`   | `src/rlmkit/strategies/wiki/querier.py::query`               |
| lint     | `rlmkit wiki lint [--health]` | `src/rlmkit/strategies/wiki/structural_lint.py::lint` + `health_lint.py::lint_health` |

`promote` is the **only** module that writes to `knowledge/wiki/`.
It stages the patch in a temp tree, runs the structural linter
against the staged tree, and only commits on lint exit 0.

### Mode D — RLMKit where it is strongest

`WikiRLMStrategy` is the issue-#37-mode-D differentiator. When
the wiki layer's distilled answer is thin, the strategy hands
the loaded wiki pages to the recursive controller as the
document substrate and lets the controller's recursive
exploration synthesise across them. This is "use RLMKit as a
smarter fallback analyzer when wiki coverage is incomplete"
verbatim from the issue.

Fallback triggers (any of):
- wiki query returned an empty answer
- citation count below `min_citations` (default 1)
- `force_rlm=True` was passed at construction

## Directory layout

```
src/rlmkit/
  strategies/
    __init__.py                 # registers WikiStrategy, WikiRLMStrategy
    wiki_strategy.py            # the two strategy classes
    wiki/                       # supporting package
      __init__.py
      entities.py               # PageEdit, WikiPatchSet, WikiState, ...
      errors.py                 # WikiError taxonomy
      yaml_lite.py              # frontmatter parser (stdlib only)
      wiki_state.py             # index/log + relevance-ranked candidates
      prompts.py                # composer functions for the 3 LLM tasks
      backends.py               # WikiBackend protocol + 2 adapters
      ingestor.py               # multi-page patch-set generator
      promoter.py               # atomic apply + structural lint gate
      structural_lint.py        # slug/dir/links/orphan checks
      health_lint.py            # weak-orphan + stale-claim + cross-link + LLM contradictions
      querier.py                # index-first navigation + answer
      schema/                   # borrowed verbatim from cct
        ATTRIBUTION.md
        page-types.md
        ingest-rules.md
        citation-rules.md
        lint-rules.md
  cli/
    __init__.py
    wiki.py                     # rlmkit wiki dispatcher

scripts/
  rlmkit-wiki                   # bash wrapper for the CLI

tests/
  test_wiki_strategy.py         # 10 tests; 2151 pre-existing tests still pass
```

## Schema / entity decisions — borrowed vs. diverged

**Borrowed verbatim from code-copilot-team:**

- `page-types.md`, `ingest-rules.md`, `citation-rules.md`,
  `lint-rules.md` — copied unmodified to keep page wire-format
  compatible across the two projects' workflows. Diverging here
  would mean a curator's wiki output from `rlmkit wiki ingest`
  could not promote into the cct wiki and vice versa, with no
  upside.
- `WikiState` relevance heuristic (token-overlap, slug + path
  + first-400 chars) — battle-tested in cct, no need to invent
  a different one.
- `_extract_index_links` regex (`\]\(([^)]+\.md)(?:#…)?\)`) —
  same shape, same tests pass.
- Promoter's "stage to temp tree, lint the staged tree, commit
  on green" pattern — the cct discovery that linting against
  the staged (post-apply) tree, not the live tree, is what
  makes cross-edit promotions valid (a `update` to a freshly
  `create`-d page).
- Weak-orphan algorithm (single inbound edge from `index.md`)
  and the `incidents/concepts/...` directory taxonomy.
- Exit-code taxonomy: 1=generic / 3=backend / 4=contract / 5=
  source-missing / 6=output-dir.

**Diverged from cct because:**

- **Backend protocol.** cct's `Backend` subprocesses out to
  `claude -p` / `cursor-agent -p` / `codex exec` because cct
  is copilot-CLI-shaped. RLMKit already has a rich `LLMClient`
  protocol with budget tracking, retries, async streaming, and
  100+ providers via LiteLLM. We diverged: `LLMClientWikiBackend`
  wraps any `LLMClient`, and `DeterministicTestBackend`
  provides the CI-safe path. No subprocess boundary.
- **Registration as Strategies.** cct exposes the operations
  through a standalone `scripts/wiki_ingest/` package. RLMKit
  registers them in the strategy registry to satisfy the
  architectural constraint and to unlock comparison via
  `MultiStrategyEvaluator` (so `wiki` and `wiki_rlm` can be
  benchmarked head-to-head against `direct`/`rag`/`rlm` on the
  same prompt).
- **`WikiRLMStrategy` is new.** Not present in cct because
  cct has no recursive controller. This is the issue-#37
  mode-D differentiator.
- **CLI lives in Python `argparse`, not Bash + Python.** cct's
  bash wrapper is intentional (`Bash 3.2 + awk` is a
  documented constraint in the cct repo). RLMKit has no such
  constraint, and the Python CLI is easier to test.
- **Dropped subprocess gate-only mode and `--legacy-single-source`
  alias.** RLMKit has no v1-substrate to maintain backward
  compatibility against.

## How the recursive controller is wired into wiki + rlm

`WikiRLMStrategy.run` is the integration point:

```
1. Run WikiStrategy.run(query)   → wiki_result
2. If wiki coverage is weak:
     substrate = "# index.md\n\n" + index_md
                 + for each loaded page: "# <path>\n\n" + content
     rlm = RLM(client=self.client, config=self.rlm_config)
     rlm_result = rlm.run(prompt=substrate, query=query)
     # The recursive controller's exploration runs across the
     # loaded pages with the rlmkit sandbox + budget machinery.
   else:
     rlm_result = None
3. Return StrategyResult merged from (wiki_result, rlm_result)
   with metadata.fallback ∈ {"none","rlm","rlm-error"}.
```

Critically, the recursive controller is given the loaded wiki
pages as a **document substrate**, not an empty prompt — the
same way `RLMStrategy.run` hands `content` to `RLM.run`. This
means the recursive exploration takes place *inside* the wiki
state the index-first navigation already loaded; the controller
recurses across that distilled-and-curated material rather than
across raw sources.

The trace and metadata from both phases are merged so a
downstream `MultiStrategyEvaluator` row reports total tokens,
total steps, and the fallback path taken.

## Testing

`tests/test_wiki_strategy.py` — 10 tests, all pass:

- `test_strategy_protocol_compliance` — structural compliance
- `test_wiki_strategy_runs_query_against_fixture`
- `test_ingest_promote_query_round_trip` — full e2e using the
  deterministic backend
- `test_promote_rejects_invalid_patch` — atomicity proof
- `test_structural_lint_clean_on_seed`
- `test_health_lint_runs_without_backend` — flags weak-orphan
- `test_wiki_rlm_strategy_falls_back_when_forced` — the
  mode-D path
- `test_wiki_rlm_strategy_skips_fallback_when_coverage_strong`
- `test_cli_lint_reports_clean`
- `test_cli_ingest_writes_proposal`

Pre-existing suite: `2151 passed, 3 skipped` (unchanged).
