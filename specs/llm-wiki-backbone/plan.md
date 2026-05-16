---
feature_id: llm-wiki-backbone
status: in_progress
---

# Plan — LLM Wiki Backbone

## Phase 1 — Skeleton

- New module tree: `src/rlmkit/strategies/wiki/`
  - `__init__.py`
  - `entities.py` — PageEdit, WikiPatchSet, WikiState, Citation,
    QueryAnswer, HealthFinding (frozen dataclasses)
  - `errors.py` — domain-specific errors mirroring cct exit-code taxonomy
  - `yaml_lite.py` — frontmatter parsing (borrowed from cct)
  - `wiki_state.py` — load index/log + relevance-ranked candidates
    (borrowed from cct, simplified)
  - `prompts.py` — compose_multi_prompt, compose_query_prompt,
    compose_health_prompt (borrowed shape; rlmkit-flavored)
  - `backends.py` — `WikiBackend` protocol + `LLMClientWikiBackend`
    + `DeterministicTestBackend`
  - `ingestor.py` — orchestrates ingest_multi → patch-set + preview
    files
  - `promoter.py` — staged tree + structural lint gate + commit
  - `structural_lint.py` — minimal structural linter (slug uniqueness,
    required H2s for promotable types, intra-wiki link integrity,
    orphan-from-index)
  - `querier.py` — index-first navigation + answer
  - `health_lint.py` — weak-orphan + stale-claim + missing-cross-link
    checks; LLM contradictions pass via backend
  - `schema/` — copied from cct (page-types, ingest-rules,
    citation-rules, lint-rules) with attribution

- New file: `src/rlmkit/strategies/wiki_strategy.py`
  - `WikiStrategy` — implements `LLMStrategy.run` by routing to the
    querier; exposes `ingest`, `promote`, `query`, `lint_health`
    helpers
  - `WikiRLMStrategy` — first runs `WikiStrategy.query`; if the
    answer is empty / citations sparse / `force_rlm=True`, invokes
    `rlmkit.core.rlm.RLM` with the loaded wiki pages joined as the
    document substrate, returning the recursive answer
  - Registered in `strategies/__init__.py` alongside
    DirectStrategy/RAGStrategy/RLMStrategy

- Mode constants: add `MODE_WIKI = "wiki"`, `MODE_WIKI_RLM = "wiki_rlm"`
  to `application/sandbox_vars.py`.

## Phase 2 — CLI

- `src/rlmkit/cli/__init__.py`, `src/rlmkit/cli/wiki.py`:
  argparse-based dispatcher for `rlmkit wiki ingest|promote|query|lint`.
- `scripts/rlmkit-wiki` (executable bash wrapper) that calls
  `python -m rlmkit.cli.wiki "$@"`.
- Optional console-script entry would touch `pyproject.toml`; we
  will leave that to a follow-up to keep the change minimal.

## Phase 3 — Tests

- `tests/test_wiki_strategy.py`:
  - `WikiStrategy.run` dispatches to query and returns
    `StrategyResult` with citation count in metadata.
  - End-to-end ingest → promote → query against a tmp wiki
    using `DeterministicTestBackend`.
  - `WikiRLMStrategy` wires through to `RLM` when `force_rlm=True`.
  - Promoter atomicity: rejected-by-lint patch-sets leave the
    wiki untouched.
  - Health lint detects weak-orphans on a fixture wiki.

- Run pre-existing test suite to confirm no regressions.

## Phase 4 — Docs + commits

- `DESIGN.md` at repo root.
- `RESULT.md` at repo root.
- One commit per phase.

## Borrow vs. diverge — quick log

- **Borrow** schema files verbatim from cct (page-types,
  ingest-rules, citation-rules, lint-rules) — single source of
  truth, no point forking.
- **Borrow** WikiState relevance heuristic, frontmatter parser,
  index-link-extraction regex, weak-orphan algorithm — they're
  battle-tested in cct.
- **Diverge** on backend protocol: cct subprocesses out to
  `claude -p` / `cursor-agent -p`; rlmkit already has a
  rich `LLMClient` with budget tracking, retries, streaming.
  Wrap that via `LLMClientWikiBackend` instead of forking the
  subprocess pattern.
- **Diverge** on registration: cct exposes the operations
  through a standalone `scripts/wiki` package; rlmkit registers
  them as Strategies in the existing registry to satisfy the
  architectural constraint and unlock `MultiStrategyEvaluator`
  comparisons.
- **Add** `WikiRLMStrategy` — not present in cct because cct
  has no recursive controller. This is the issue #37 mode-D
  differentiator.
