---
id: llm-wiki-backbone
status: planned
spec: ./spec.md
created: 2026-05-05
---

# Implementation plan — LLM Wiki backbone

## Phasing

### Phase 1 — Domain layer (no I/O)
- `src/rlmkit/domain/wiki.py`
  - `PageType` (StrEnum: concept | workflow | incident | decision | playbook | glossary | overview)
  - `WikiCitation` (slug, page_type, raw_sources)
  - `WikiPage` (frontmatter dataclass + body str)
  - `LintSeverity` (StrEnum: error | warning)
  - `LintIssue` (path, severity, code, message)
  - `LintReport` (issues list; `passed` property; `errors` / `warnings` lists)
  - `IngestResult` (source_id, pages_created, pages_updated, log_entry)

### Phase 2 — Application ports & use cases
- `src/rlmkit/application/ports/wiki.py`
  - `WikiRepositoryPort` (Protocol):
    `read_page`, `write_page`, `list_pages`, `read_raw`, `write_raw`,
    `list_raws`, `append_log`, `read_index`, `write_index`.
- `src/rlmkit/application/use_cases/wiki/ingest.py`
  - `IngestSourceUseCase.execute(source_id, content) -> IngestResult`
  - Calls LLMClient with `wiki_ingest` prompt → parses YAML page list → writes via repo.
- `src/rlmkit/application/use_cases/wiki/query.py`
  - `QueryWikiUseCase.execute(question) -> StrategyResult`
  - Uses chunker + embedder to score wiki pages, builds context, calls LLM.
- `src/rlmkit/application/use_cases/wiki/lint.py`
  - `LintWikiUseCase.execute() -> LintReport`
  - Pure validation; no LLM.
- `src/rlmkit/application/use_cases/wiki/promote.py`
  - `PromoteAnswerUseCase.execute(answer, sources, page_type) -> WikiPage`

### Phase 3 — Infrastructure
- `src/rlmkit/infrastructure/wiki/markdown_repository.py`
  - `MarkdownWikiRepository(root: Path)` implements `WikiRepositoryPort`.
- `src/rlmkit/infrastructure/wiki/frontmatter.py`
  - `parse_frontmatter(text) -> (frontmatter_dict, body)`
  - `serialize_page(page: WikiPage) -> str`
- `src/rlmkit/infrastructure/wiki/index_writer.py`
  - Builds `index.md` from page listing.

### Phase 4 — Strategies (rlmkit-style)
- `src/rlmkit/strategies/wiki.py` — `WikiStrategy` (LLMStrategy)
- `src/rlmkit/strategies/wiki_rlm.py` — `WikiRLMStrategy` (LLMStrategy)
- Register in `strategies/__init__.py`.

### Phase 5 — Prompts & constants
- `prompts/wiki/wiki_query.yaml`
- `prompts/wiki/wiki_ingest.yaml`
- `prompts/wiki/wiki_promote.yaml`
- Add `MODE_WIKI = "wiki"`, `MODE_WIKI_RLM = "wiki_rlm"` to `sandbox_vars.py`.

### Phase 6 — Bootstrap content
- `knowledge/wiki/index.md`, `log.md`, `overview.md` (empty/seed forms).
- `knowledge/schema/page_schema.yaml`, `ingest_rules.md`, `citation_rules.md`,
  `lint_rules.md`.

### Phase 7 — Tests
- `tests/wiki/test_domain.py` (linter dataclasses)
- `tests/wiki/test_frontmatter.py` (parse/serialize round-trip)
- `tests/wiki/test_repository.py` (tmp_path-based)
- `tests/wiki/test_lint.py` (frontmatter validation, link-rot warning)
- `tests/wiki/test_e2e_ingest_query_promote.py` — **the e2e deliverable**:
  - StubLLMClient returns canned YAML for ingest, canned answer for query.
  - StubEmbedder gives deterministic embeddings (hash-based).
  - Asserts: pages written, index updated, query returns expected text,
    promote creates a new page.

### Phase 8 — Wrap
- `DESIGN.md` and `RESULT.md` at repo root.
- Final commit.

## File summary

```
src/rlmkit/
  domain/wiki.py                                                     [NEW]
  application/ports/wiki.py                                          [NEW]
  application/use_cases/wiki/__init__.py                             [NEW]
  application/use_cases/wiki/ingest.py                               [NEW]
  application/use_cases/wiki/query.py                                [NEW]
  application/use_cases/wiki/lint.py                                 [NEW]
  application/use_cases/wiki/promote.py                              [NEW]
  application/sandbox_vars.py                                        [EDIT: add MODE_WIKI*]
  infrastructure/wiki/__init__.py                                    [NEW]
  infrastructure/wiki/markdown_repository.py                         [NEW]
  infrastructure/wiki/frontmatter.py                                 [NEW]
  infrastructure/wiki/index_writer.py                                [NEW]
  strategies/wiki.py                                                 [NEW]
  strategies/wiki_rlm.py                                             [NEW]
  strategies/__init__.py                                             [EDIT: register]
prompts/wiki/wiki_query.yaml                                         [NEW]
prompts/wiki/wiki_ingest.yaml                                        [NEW]
prompts/wiki/wiki_promote.yaml                                       [NEW]
knowledge/wiki/{index,log,overview}.md                               [NEW]
knowledge/schema/{page_schema.yaml,ingest_rules.md,citation_rules.md,lint_rules.md} [NEW]
tests/wiki/*.py                                                      [NEW]
DESIGN.md                                                            [NEW]
RESULT.md                                                            [NEW]
```

## Risks / mitigations
- **Pydantic in domain layer** — rlmkit's domain layer uses stdlib only. Stick
  to `dataclasses` + manual validation.
- **Prompt YAML loader** — reuse `rlmkit.prompts.templates` if it has a YAML
  loader; otherwise read directly with `yaml.safe_load`.
- **Test isolation** — every test uses `tmp_path` for `root`; no global state.

## Done criteria
- `uv run pytest tests/wiki/ -q` exits 0.
- `WikiStrategy` and `WikiRLMStrategy` import cleanly and pass
  `isinstance(..., LLMStrategy)` (structural check).
- `lint()` over the seed `knowledge/` tree returns `LintReport(passed=True)`.
