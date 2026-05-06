# Wiki Backbone Bootstrap

Single-source fixture for the wiki ingest end-to-end test.

This file represents a hypothetical incident write-up that a
curator might want to promote into the wiki. The deterministic
test backend should derive a slug from the H1 above and emit a
proposal whose embedded frontmatter passes the two-layer validator.

## Background

A retrieval-only RAG pipeline stopped surfacing the right pages
after a re-chunking. The fix was unrelated; the lesson worth
capturing is the diagnostic ladder we walked.

## What we changed

We added a `chunk_size` invariant test and a sanity check that
counts surviving citations after a re-chunk.

## Sources

- internal incident report
- the failing CI run
