# Lint rules

`LintWikiUseCase` runs strictly on frontmatter and warn-only on link rot.

| Code  | Severity | Meaning                                                  |
|-------|----------|----------------------------------------------------------|
| FM001 | error    | Required frontmatter field missing.                      |
| FM002 | error    | Frontmatter present but unparseable / invalid.           |
| FM003 | error    | `type` not in the allowed set.                           |
| LK001 | warning  | Intra-wiki link target not found.                        |
| SR001 | warning  | `sources` references a raw id not in `knowledge/raw/`.   |
| OP001 | warning  | Page has empty `sources` list (orphan).                  |

A `LintReport` passes iff there are no errors. Warnings are surfaced but do
not fail the report.
