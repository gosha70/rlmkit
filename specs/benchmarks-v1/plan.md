---
feature_id: benchmarks-v1
spec_mode: full
spec: ./spec.md
status: draft
date: 2026-08-15
origin:
  urls:
    - https://arxiv.org/abs/2512.24601
  transcripts:
    - "Owner, 2026-08-15: accepted 'a reproducible BENCHMARKS page' as a pre-launch differentiator."
  origin_claim: |
    Inherited from spec.md — public, reproducible numbers produced by the tool
    itself across ≥3 providers × all engines, published as BENCHMARKS.md.
---

# Plan — Reproducible benchmarks v1

Depends on `specs/rebrand-rlm-studio` (paths use `rlmstudio`); uses `rlm_official` from `specs/interop-official-rlm` if it has landed, otherwise runs three engines and says so. Size: **M** (≈5–7 working days including one full benchmark run + doc). Circuit breaker: if numbers are not reproducible from a clean clone by end of week 4, ship the runner + dataset and hold `BENCHMARKS.md` for a v1.0.1 docs release.

## 1. Reuse map
| Need | Existing |
|---|---|
| Dataset schema/loader | `src/rlmstudio/benchmark/dataset.py` (`BenchmarkCase`, `load_dataset`) — extend fields |
| Run × engines × providers | `application/use_cases/run_matrix_comparison.py` (`RunMatrixComparisonUseCase`, `MatrixSlotDTO`) — one slot per provider×engine |
| Telemetry (TTFT, tokens, cache, cost) | telemetry store + trace fields added by the prefill/decode work (`CHANGELOG` "Prefill / decode telemetry"); server route helpers compute aggregates — the runner must compute from raw trace keys in the application layer (dependency rule; see the `_rank` deviation note in CHANGELOG) |
| Judge | `prompts/judge_pointwise.yaml` via `LLMPort`; server `POST /api/evaluations/judge` shows the call shape |
| Report | `src/rlmstudio/benchmark/report.py` (`BenchmarkReport.summary/per_case_table/save_json/save_csv`) — add Markdown |
| CLI | `src/rlmstudio/cli/main.py` argparse dispatcher — add `bench` |
| Fakes for CI | in-memory fake LLM used by `tests/test_use_cases.py` |

## 2. Allowlist
`src/rlmstudio/benchmark/**`, `src/rlmstudio/cli/main.py`, `benchmarks/**`, `BENCHMARKS.md`, `README.md` (one section), `docs/rlm-studio-guide.md` (one section), `.github/workflows/ci.yml` (one step), `tests/test_benchmark*.py`, `CHANGELOG.md`.

## 3. Steps
1. Dataset: extend `BenchmarkCase` (`min_tokens`, `task_type`, `rubric_hint`, `budget`); author `benchmarks/longdoc-v1.yaml` (≥12 cases; sources + licenses listed in a `sources:` block); loader tests.
2. Matrix driver: `benchmark/matrix_runner.py` — builds `MatrixSlotDTO`s from `providers × engines` using the same provider/sandbox/embedder wiring the server uses (factor a small builder out of `server/dependencies.py` into `application`/`infrastructure` if it is not already reusable — keep the dependency rule), runs per case, collects `RunResultDTO` + trace metrics.
3. Scoring: `benchmark/scoring.py` — exact/contains + judge via `LLMPort`; judge model recorded.
4. Report: Markdown table + aggregates; `BENCHMARKS.md` marker-based regeneration (`<!-- bench:start -->` … `<!-- bench:end -->`).
5. CLI `rlm-studio bench` + `benchmarks/run_benchmark.py` wrapper; `--dry-run` with fakes for CI (`bench-smoke`).
6. Full run: cloud ×2 (cheap tier), local ×1 (Spark vLLM or Ollama), engines ×3–4, three repetitions for cloud judge scores; commit `results/<date>/*.json` under `benchmarks/results/` and the regenerated `BENCHMARKS.md`.
7. Docs: README section cites the table; studio-guide "Reproducing the benchmarks"; CHANGELOG.

## 4. Test strategy
- Unit: dataset loader (new fields, validation), scoring (exact/contains; judge parsing with a fake), report Markdown rendering, marker regeneration idempotency.
- Integration (fakes): `bench --dry-run` on 2 cases × 2 engines produces JSON + Markdown; CI `bench-smoke`.
- Real run: AC-1 executed once by the owner; cost logged.

## 5. Risks
| Risk | Mitigation |
|---|---|
| Numbers unflattering to RLM on some tasks | Publish anyway with analysis — credibility is the product; "Not a fit" section already says small docs favour Direct |
| Provider drift after publication | Table carries date + exact model IDs; rerun per minor release |
| Cost overrun | cheap tiers, per-case budgets, `--limit` flag; documented total |
| Local hardware not available for a rerun | Ollama laptop row is the reproducible baseline; Spark row labelled with hardware |
