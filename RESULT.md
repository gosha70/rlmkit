# Result — LLM Wiki backbone session

## Wall-clock

- Session start: 2026-05-05 23:34 EDT (timestamp captured in /tmp).
- Last commit before this writeup: ~2026-05-06 00:24 EDT.
- Active wall-clock: roughly **50 minutes** of agent time.
- The raw start-vs-now delta is ~480 minutes because the harness paused the
  session and resumed it on a new calendar day; the agent itself was not
  working continuously across that gap. I'm flagging this honestly rather
  than claiming the bigger number — the comparison with the wiki-equipped
  session should be apples-to-apples.

## Self-rating against issue #37 (1–5)

### A — Modes / backends: **4**
Two new modes wired in (`MODE_WIKI`, `MODE_WIKI_RLM`) as first-class
strategies, registered in `rlmkit.strategies.__init__` next to
DirectStrategy / RAGStrategy / RLMStrategy. Did not modify `auto` to
prefer wiki — that's a deliberate follow-up, not an oversight, but it
costs a point.

### B — Raw → wiki → query pipeline: **4**
Full `knowledge/{raw,wiki,schema}/` tree implemented; ingest mirrors raw,
generates pages from a YAML LLM contract, and rebuilds `index.md`. Auto-
generated `index.md` and append-only `log.md` work. `overview.md` exists
as a page type but isn't auto-regenerated on bulk ingest yet.

### C — Bounded operations: **4**
Ingest, query, lint, promote all implemented as separate use cases with
clean Protocol boundaries. `update` is a thin re-call of `ingest` — it
works but isn't a distinct entry point. Lint has six codes split between
errors (frontmatter) and warnings (link rot, stale sources, orphans).
The fallback path from query → raw exploration is the `wiki_rlm`
strategy, which works.

### D — RLMKit-for-scale: **3**
`WikiRLMStrategy` does fall back to `RLMStrategy` over linked raws when
coverage is `missing`, and it tags the result with
`metadata.fallback_backend = "rlm"`. But it doesn't yet handle very large
multi-source corpora well: the raw concatenation step is naive (string
join with separators) and would blow past context for a real corpus.
RLMKit's recursive controller absorbs that, but the sizing logic that
decides *how many* raws to feed it is missing. Counts as a 3 because the
integration is wired, the scaling decisions are not.

## What I wish I had been told at session start

- **`specs/` is gitignored in this repo.** The global rule says specs/
  must be committed; the project's `.gitignore` line `/specs` says
  otherwise. I had to `git add -f` to land the SDD artifacts. Whichever
  is the curator's intent, knowing up front would have saved a round trip.
- **Whether the `prompts/` package referenced in CLAUDE.md is
  `src/rlmkit/prompts/` (the Python package) or a top-level `prompts/`
  directory.** I went with the Python package because that's where rlmkit
  already keeps its YAML prompts; the curator may have wanted the top-
  level layout shown in CLAUDE.md.
- **Whether `auto` mode should be modified.** I chose not to so the
  diff stays scoped, but a one-line "yes, prefer wiki when populated" or
  "no, leave it alone" would have removed the deliberation.
- **The exact stub-LLM contract you'd find natural for the e2e test.**
  I picked YAML-pages-out-of-the-LLM as the ingest contract; some teams
  prefer JSON or a structured tool-call. Not load-bearing, but a single
  example of the desired stub style would tighten the deliverable.

## "Would have copied X from Y" notes (memory rule)

None. I had no recall of an existing wiki implementation in another
project that I declined to import. The closest reference would have been
the LangChain DocumentLoader pattern, which I did not copy or import. The
ingest contract (LLM-returns-YAML-pages) was reached independently from
the issue's `ingest_rules.md` description.

## Tests

- `uv run --with pytest --with pytest-xdist pytest tests/wiki/ -q` →
  **19 passed in 4.5s**.
- Sanity check on existing suite (`tests/test_domain.py`,
  `tests/test_port_compliance.py`, `tests/test_use_cases.py`) →
  **250 passed**, no regressions.
- Tests use `StubLLMClient` and `StubEmbedder` — no network, no provider
  key required.

## Known gaps left for follow-up

(see DESIGN.md "Punted" section for the full list)

- CLI surface.
- Auto-promotion of `wiki_rlm` answers.
- Filesystem-watcher-driven incremental re-ingest.
- Coverage-threshold calibration once a real wiki exists.
- Smart raw-source budgeting for `WikiRLMStrategy` on very large corpora.
