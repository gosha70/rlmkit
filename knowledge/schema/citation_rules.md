# Citation rules

Every wiki page must trace each substantive claim back to a raw source.

- The `sources` frontmatter field lists the raw source ids the page derives
  from. The list must be non-empty for any page that summarizes external
  material (lint emits a warning otherwise).
- Inline citations in the body use the form `[<source-id>]` to reference an
  entry in `knowledge/raw/`, or `[<page-slug>]` to reference another wiki
  page.
- A wiki page must never cite a source that is not present in
  `knowledge/raw/`. The linter flags missing raw sources as warnings; CI
  treats them as informational, not blocking.
