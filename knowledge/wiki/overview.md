---
page_type: overview
slug: overview
title: RLMKit Wiki — Overview
status: stable
last_reviewed: 2026-05-05
sources:
  - issue: gosha70/rlmkit#37
  - path: specs/llm-wiki-backbone/spec.md
    sha: HEAD
---

# RLMKit Wiki — Overview

## Summary

The RLMKit wiki is a curated, persistent markdown knowledge layer
distilled from raw sources (specs, issues, PRs, session notes,
incident write-ups). Queries consult the wiki **first**; RLMKit's
recursive controller is the fallback synthesis engine when wiki
coverage is weak or when the corpus is too large for flat
retrieval.

## Key ideas

- **Wiki-first.** A page in the wiki is a contract that someone
  curated the answer. Re-deriving it from raw sources should be
  the exception.
- **Curator-gated, agent-assisted.** The ingest pipeline drafts
  proposals; a human curator decides whether to land them.
- **RLM-backed recursion.** When a corpus is too large for a
  single LLM call, RLMKit's recursive controller drives the
  ingest as a code-execution loop (mode `wiki + rlm`).

## Where this shows up

- `src/rlmkit/wiki/` — the implementation.
- `scripts/wiki-{ingest,query,lint}` — CLI entrypoints.
- `knowledge/wiki/schema/` — the curator-facing schema (borrowed
  verbatim from code-copilot-team).
- `specs/llm-wiki-backbone/` — spec + plan.

## Related

- [LLM Wiki as Knowledge Layer](concepts/llm-wiki-as-knowledge-layer.md)
