---
type: pitch
status: proposed
priority: P2
appetite: 1 week
---

# Pitch: Legacy Code Lint & Type Debt

## Problem
The CI pipeline (Phase 8) scopes ruff and mypy to the actively maintained modules
(`server/`, `application/`, `domain/`, `infrastructure/`, `prompts/`). The legacy
modules (`core/`, `ui/`, `__init__.py`) have ~2200 ruff errors and ~180 mypy errors
that are excluded from CI but block whole-project linting.

Key issues:
- `src/rlmkit/__init__.py` re-exports `LMStudioClient` and `vLLMClient` with
  conditional imports that confuse mypy (assigns `None` to a type alias)
- `src/rlmkit/ui/` (old Streamlit UI) has extensive lint violations
- `src/rlmkit/core/` (pre-Clean-Arch code) duplicates patterns now in `application/`

## Appetite
1 week. Most errors are auto-fixable (`ruff --fix`). The mypy errors require
manual review of legacy type annotations.

## Solution
1. Run `ruff check --fix` on all of `src/`
2. Fix remaining manual ruff errors
3. Fix `__init__.py` conditional imports to satisfy mypy
4. Either type-annotate or add `# type: ignore` to legacy modules
5. Remove `--follow-imports=silent` from CI mypy invocation
6. Remove the module-scoping from CI ruff (run on all of `src/`)

## Rabbit Holes
- Don't refactor legacy code — just fix lint/type issues
- If a legacy module is truly dead code, delete it instead of fixing it
- The Streamlit UI (`src/rlmkit/ui/`) may be removable entirely if the
  Next.js frontend has fully replaced it

## No-Gos
- No behavioral changes to legacy code
- No new features mixed with lint cleanup
