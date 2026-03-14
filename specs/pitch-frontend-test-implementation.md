---
type: pitch
status: proposed
priority: P1
appetite: 1 week
---

# Pitch: Frontend Test Implementation

## Problem
The frontend has 5 test files with real test implementations (chat, settings,
dashboard, traces, accessibility) but they require Node >= 18 to run (Vite 6+
uses `node:fs/promises` named exports). The CI workflow (Phase 8) includes
`npm run test` with Node 20, so these tests will run in CI but not locally
on machines with older Node.

Additionally, the new Phase 7 components (ResponseRating, PickWinner,
JudgeScores) have no test coverage yet.

## Appetite
1 week.

## Solution
1. Add tests for Phase 7 components:
   - `response-rating.test.tsx` — toggle on/off, optimistic state
   - `pick-winner.test.tsx` — selection, highlight, min 2 responses guard
   - `judge-scores.test.tsx` — dimension bars render, score display
2. Add tests for evaluation API functions in `api.ts`
3. Document Node >= 18 requirement in README or `.nvmrc`
4. Verify all 5 existing test files pass in CI (Node 20)

## Rabbit Holes
- Don't mock the full chat page — test components in isolation
- Don't add Playwright E2E tests yet (separate pitch)

## No-Gos
- No changes to component behavior
- No new dependencies beyond what's already in devDependencies
