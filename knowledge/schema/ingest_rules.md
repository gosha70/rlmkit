# Ingest rules

A raw source becomes a wiki entry through `IngestSourceUseCase`. Every
ingested source is mirrored verbatim under `knowledge/raw/<source-id>.md`
before any LLM call. The LLM is asked to produce a YAML page list (see
`prompts/wiki/wiki_ingest.yaml`); each yielded page is written to its
type-specific directory (`concepts/`, `workflows/`, etc.).

## Always
- Mirror raw input first, in full, with no transformation.
- Set `sources: [<source-id>]` on every produced page.
- Use kebab-case slugs.
- Stamp `created` and `updated` with the ingest date.

## Never
- Inline raw content in a wiki page body — link by source id instead.
- Promote secrets, tokens, or PII from raw into wiki pages.
- Mix multiple types in one page; split it instead.
