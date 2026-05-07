---
feature_id: llm-wiki-backbone
spec_mode: full
status: draft
issue: 37
origin:
  issue: gosha70/rlmkit#37
  urls:
    - https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
  origin_claim: |
    Add a first-class LLM Wiki / knowledge-base feature to RLMKit.
    The wiki is a persistent, distilled markdown layer between raw
    sources and the model. RLMKit becomes the scalable reasoning
    engine behind it: ingest raw sources into multi-page wiki
    patches, promote (apply) patches atomically, query the wiki
    index-first with citations, and lint knowledge health. When
    coverage is weak the recursive RLM controller is wired in as
    the synthesis fallback (mode `wiki + rlm`).
---

# LLM Wiki Backbone for RLMKit

## Problem

RLMKit today exposes four execution modes — `direct`, `rag`, `rlm`,
`compare`, `auto` — all of which re-feed raw source material into
the model on every query. As corpora grow this loses recall and
context-window quality. Karpathy's LLM Wiki gist proposes a
persistent, distilled markdown layer maintained by an LLM via
three operations (ingest, query, knowledge-health lint) plus a
write gate (promote). RLMKit's recursive controller is a natural
fit for the synthesis step when wiki coverage is weak.

This spec adds the wiki layer as **first-class strategies** in
the existing strategy registry (`src/rlmkit/strategies/`) so the
four wiki operations are reachable through the same
`MultiStrategyEvaluator` and routing layer that handles direct,
rag, and rlm. The architecture choice is non-negotiable: the
wiki feature lives inside the strategy registry, not in a
sidecar `src/rlmkit/wiki/` subpackage.

## User Scenarios

1. **Ingest a source.** A curator runs
   `rlmkit wiki ingest specs/some-spec.md`. The pipeline reads
   `knowledge/wiki/index.md`, `log.md`, and a relevance-ranked
   set of candidate pages, asks the configured LLM client to
   produce a `WikiPatchSet` (1..N page edits + index/log
   updates), and writes the proposal to
   `doc_internal/proposals/<date>-<slug>/`.

2. **Promote a proposal.** The curator runs
   `rlmkit wiki promote doc_internal/proposals/<dir>`. The
   promoter is the **only writer** to `knowledge/wiki/`. It
   stages the patch in a temp tree, runs the structural linter
   against the staged tree, and only on lint exit 0 commits the
   files into `knowledge/wiki/` and archives the proposal under
   `.applied/`.

3. **Query the wiki.** The curator runs
   `rlmkit wiki query "what does our wiki say about X?"`. The
   querier reads `index.md` first, follows links to a bounded
   relevance-ranked set of pages (default ≤5), and prints an
   answer + `(page, fragment)` citations. Pages-loaded list is
   appended to `doc_internal/wiki-query-log.jsonl` for audit.

4. **Wiki + RLM (the differentiator).** When wiki coverage is
   weak, mode `wiki + rlm` falls back to RLMKit's recursive
   controller. `WikiRLMStrategy` first runs a `WikiStrategy`
   query against the index. If the answer is empty / the
   citations are sparse / a `--force-rlm` flag is set, the
   strategy invokes the existing `RLM` controller from
   `core/rlm.py` with the loaded wiki pages as the document
   substrate. The recursive synthesis answer round-trips back
   into the wiki via `--file-back`, producing a `WikiPatchSet`
   that the curator can promote.

5. **Knowledge-health lint.** The curator (or CI) runs
   `rlmkit wiki lint --health`. The structural linter (existing
   substrate ported from cct) runs first; the health pass adds
   weak-orphan + stale-claim + missing-cross-link checks. The
   contradictions pass is the only LLM-dependent one and is
   skipped when no client is configured (advisory mode).

6. **Test backend.** A deterministic in-process backend (no
   network) handles `ingest` / `query` / `lint-health` so the
   end-to-end test can round-trip without API keys.

## Interface

### Python — strategy registration

```python
# src/rlmkit/strategies/__init__.py — extended
from .wiki_strategy import WikiStrategy, WikiRLMStrategy
```

### Python — entities

```python
@dataclass(frozen=True)
class PageEdit:
    path: str                       # "concepts/foo.md"
    action: Literal["create", "update", "append-log", "append-index"]
    new_content: str                # full markdown for create/update; one line for append-*
    rationale: str = ""

@dataclass(frozen=True)
class WikiPatchSet:
    edits: list[PageEdit]
    source_path: str
    rationale: str

@dataclass(frozen=True)
class WikiState:
    index_md: str
    log_md: str
    candidate_pages: dict[str, str]

@dataclass(frozen=True)
class Citation:
    page: str
    fragment: str

@dataclass(frozen=True)
class QueryAnswer:
    answer: str
    citations: list[Citation]
    pages_loaded: list[str]

@dataclass(frozen=True)
class HealthFinding:
    kind: Literal["contradiction", "stale-claim", "weak-orphan", "missing-cross-link"]
    severity: Literal["warning", "error"]
    pages: list[str]
    description: str
```

### Python — port

```python
# src/rlmkit/application/ports/wiki_backend.py
class WikiBackend(Protocol):
    def ingest_multi(self, prompt: dict) -> dict: ...
    def query(self, prompt: dict) -> dict: ...
    def lint_health(self, prompt: dict) -> dict: ...
```

A `LLMClientWikiBackend` adapter wraps the existing `LLMClient`
protocol so any rlmkit-configured provider becomes a wiki
backend. A `DeterministicTestBackend` provides the CI-safe path.

### CLI surface

```
rlmkit wiki ingest <source>             multi-page write plan
rlmkit wiki promote <proposal-dir>      atomic apply (only writer)
rlmkit wiki query "<question>"          index-first synthesis
rlmkit wiki query --file-back "..."     synthesise + emit patch-set
rlmkit wiki lint                        structural pass
rlmkit wiki lint --health               structural + knowledge-health
rlmkit wiki lint --health --strict      non-zero exit on health flags
```

All verbs accept `--backend test` for the deterministic path and
`--wiki-root knowledge/wiki` to override the default.

### Modes (Issue #37, A–D)

| Issue mode  | Strategy class       | Mode constant       | Wiring |
|-------------|----------------------|---------------------|--------|
| A — direct  | (existing) `DirectStrategy` | `MODE_DIRECT` | unchanged |
| A — rag     | (existing) `RAGStrategy` | `MODE_RAG` | unchanged |
| A — `wiki`  | `WikiStrategy` | `MODE_WIKI` | new |
| A — `wiki + rlm` | `WikiRLMStrategy` | `MODE_WIKI_RLM` | new — wraps `RLM` |
| B — raw → wiki → query | persisted layout under `knowledge/wiki/` (mirrors cct) | n/a | new |
| C — bounded ops | ingest / promote / query / lint verbs | n/a | new |
| D — RLMKit where strongest | `WikiRLMStrategy` falls through to `RLM` for cross-doc synthesis when wiki coverage is weak or `force_rlm=True` | `MODE_WIKI_RLM` | new |

## Requirements

1. **Strategies, not sidecar.** The wiki entry points are
   `WikiStrategy` and `WikiRLMStrategy`, registered in
   `src/rlmkit/strategies/__init__.py`, both implementing the
   `LLMStrategy` protocol so they compose with
   `MultiStrategyEvaluator` exactly like `DirectStrategy` /
   `RAGStrategy` / `RLMStrategy`. New mode constants
   `MODE_WIKI` and `MODE_WIKI_RLM` are added to
   `application/sandbox_vars.py`.

2. **Four operations.** `ingest`, `promote`, `query`, `lint` are
   reachable from the new CLI `rlmkit wiki <verb>`. The
   strategies' `run(content, query)` method dispatches to the
   query operation for use through `MultiStrategyEvaluator`.

3. **Promote = only writer.** No code path other than the
   promoter writes to `knowledge/wiki/`. Atomic: stage to temp
   tree, lint, commit. On any failure the live wiki is
   untouched.

4. **Index-first navigation.** Query reads `index.md` first and
   only loads pages reachable from the index. The pages-loaded
   audit log is written to `doc_internal/wiki-query-log.jsonl`.

5. **Wiki + RLM wiring.** `WikiRLMStrategy` must invoke
   `rlmkit.core.rlm.RLM` for synthesis, not call the LLMClient
   directly. The recursive controller's full budget /
   step-limit / sandbox machinery applies. This is the "use
   RLMKit where it is strongest" deliverable from issue #37
   mode D.

6. **Schema borrowed verbatim.** Page-types,
   ingest-rules, citation-rules, lint-rules are copied from the
   cct wiki schema unchanged (with attribution). Diverging the
   schema would mean two incompatible wiki dialects in the same
   user's workflow.

7. **Deterministic test backend.** A `DeterministicTestBackend`
   answers each operation with a fixed shape; the e2e test
   round-trips ingest → promote → query without network.

8. **No new top-level dirs.** Code lives under
   `src/rlmkit/strategies/wiki_strategy.py` plus
   `src/rlmkit/strategies/wiki/` for the supporting modules
   (state loader, prompt composer, promoter, querier, health
   linter, schema). The CLI entry point lives at
   `src/rlmkit/cli/wiki.py`. A thin `scripts/rlmkit-wiki`
   wrapper runs the CLI for shell ergonomics.

9. **No regressions.** All pre-existing tests continue to pass.

## Constraints / What NOT to Build

- **No vector store.** Karpathy's gist is explicit: index-first
  navigation works at moderate scale without embeddings. Token
  overlap + index links is the retrieval primitive. (This
  diverges from rlmkit's existing `rag` mode by design — the
  wiki is not RAG.)
- **No file watcher / cron.** All four operations are manual
  CLI only.
- **No third-party deps.** Pure stdlib for the wiki package.
- **No automatic merge into `knowledge/wiki/`.** Promote is
  always curator-triggered.
- **No global page cache.** WikiState is loaded fresh per
  operation; the file system is the source of truth.

## Out of scope (deferred)

- Wiki adapter generation from `knowledge/wiki/` for
  external agent contexts.
- LLM-driven contradictions check (the structural + heuristic
  passes ship; the LLM-call contradictions pass is wired
  through the backend protocol but only invoked when an LLM
  backend is provided — advisory mode is the default).
- WebSocket / streaming wiki edits.
