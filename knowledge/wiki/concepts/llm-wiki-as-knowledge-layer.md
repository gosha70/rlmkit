---
page_type: concept
slug: llm-wiki-as-knowledge-layer
title: LLM Wiki as Knowledge Layer
status: stable
last_reviewed: 2026-05-05
sources:
  - issue: gosha70/rlmkit#37
  - url: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
    retrieved: 2026-05-05
---

# LLM Wiki as Knowledge Layer

## Summary

An LLM Wiki is a persistent, distilled markdown layer compiled
from raw sources. Instead of letting the model rediscover relevant
context on every query, the system maintains a curated knowledge
base and queries it first. Raw-source analysis (RAG, recursive
exploration) is the fallback, not the default.

## Key ideas

- **Curated, not auto-generated.** Pages are gated by a four-
  question check (reusable, citable, non-duplicative, new-
  contributor-relevant). Most session content is not wiki-worthy.
- **Typed pages.** Every page is one of `concept`, `workflow`,
  `incident`, `decision`, `playbook`, `glossary`, `open-question`,
  `index`, `log`, or `overview`, with type-specific structure.
- **Source-cited.** Every page (except `index` and `log`) carries
  a `sources:` block in its frontmatter. No source → no page.
- **Wiki-first query order.** Retrieval consults wiki pages
  before raw sources. The wiki is the canonical project memory.

## Where this shows up

- `knowledge/wiki/schema/` — the curator-facing schema.
- `src/rlmkit/wiki/ingest.py` — the ingest pipeline that drafts
  proposals.
- `src/rlmkit/wiki/query.py` — the wiki-first query path with
  RLM fallback.

## Related

- [Wiki Overview](../overview.md)
