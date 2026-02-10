# RLMKit Test Strategy

**Version:** 1.0
**Date:** February 8, 2026
**Owner:** QA Lead

---

## Test Pyramid

```
         +-----------+
         |   E2E     |  ~20 tests (Playwright, future Cycle 2)
         |  Manual   |
         +-----------+
        /             \
       / Integration   \  ~100 tests (pytest, mock LLM)
      +----------------+
     /                  \
    /    Unit Tests      \  ~700 tests (pytest, fast, isolated)
   +---------------------+
```

### Layer Breakdown

| Layer | Target Count | Run Time | Dependencies | CI Gate |
|-------|-------------|----------|-------------|---------|
| Unit | ~700 | < 30s | None | Yes -- every push |
| Integration | ~100 | < 2 min | MockLLMClient, temp dirs, SQLite | Yes -- every push |
| E2E / Manual | ~20 | 5-10 min | Browser, running server | Nightly / pre-release |
| Live Provider | ~15 | 1-5 min | API keys (secrets) | On-demand / weekly |

---

## Coverage Targets

| Component | Target | Current | Notes |
|-----------|--------|---------|-------|
| `core/` (rlm, parsing, budget, actions) | 95% | ~96% | Critical path, must stay high |
| `strategies/` (direct, rlm, rag, evaluator) | 90% | ~90% | Strategy logic |
| `llm/` (clients, config, auto) | 85% | ~85% | Provider adapters |
| `storage/` (database, conversation, vector) | 85% | ~85% | Persistence layer |
| `envs/` (pyrepl, sandbox) | 90% | ~90% | Security-critical |
| `config.py` | 90% | ~90% | Configuration parsing |
| `api.py` | 90% | ~90% | Public API surface |
| `ui/` | 70% | ~70% | UI code, harder to unit test |
| **Overall** | **90%** | **~96%** | CI gate: fail if < 90% |

---

## Test Categories and Markers

```python
# pytest markers (registered in pyproject.toml)
@pytest.mark.integration   # Integration tests requiring multi-component setup
@pytest.mark.e2e           # End-to-end tests requiring full stack
@pytest.mark.live          # Tests requiring real API keys (external services)
@pytest.mark.slow          # Tests taking > 5 seconds
```

### Running by Category

```bash
# Unit tests only (default, fast)
pytest tests/ -m "not integration and not e2e and not live"

# Integration tests
pytest tests/integration/ -m integration

# All tests except live
pytest tests/ -m "not live"

# Live provider tests (requires API keys)
pytest tests/ -m live

# Full suite
pytest tests/
```

---

## Test Data Management

### Fixtures (conftest.py)

Shared fixtures are defined in `tests/integration/conftest.py`:

- **`mock_llm_simple`** -- MockLLMClient returning a single FINAL answer
- **`mock_llm_multistep`** -- MockLLMClient returning code steps then FINAL
- **`mock_llm_subcall`** -- MockLLMClient producing subcall actions
- **`mock_llm_failing`** -- Client that raises RuntimeError
- **`sample_content_short`** -- Short text (< 100 words)
- **`sample_content_medium`** -- Medium text (~2000 words)
- **`sample_content_large`** -- Large text (~50K chars)
- **`rlm_config_strict`** -- RLMConfig with low step/token limits
- **`tmp_db_path`** -- Temporary database file (auto-cleaned)

### Test Data Principles

1. **No external files required** -- All test data generated in fixtures or inline.
2. **Deterministic** -- MockLLMClient returns pre-programmed responses.
3. **Isolated** -- Each test gets its own temp directory and fresh state.
4. **No API keys needed** -- Integration tests use MockLLMClient exclusively.
5. **Live tests are opt-in** -- Gated behind `@pytest.mark.live` marker.

---

## CI Integration Plan

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
jobs:
  lint:
    steps:
      - ruff check src/ tests/
      - ruff format --check src/ tests/

  test-unit:
    steps:
      - pytest tests/ -m "not integration and not e2e and not live"
        --cov=rlmkit --cov-fail-under=90

  test-integration:
    steps:
      - pytest tests/integration/ -m integration
        --cov=rlmkit --cov-append

  test-live:
    # Only on main branch, with secrets
    if: github.ref == 'refs/heads/main'
    steps:
      - pytest tests/ -m live
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### Quality Gates

| Gate | Threshold | Blocks Merge? |
|------|-----------|---------------|
| Ruff lint (errors) | 0 errors | Yes |
| Unit tests | 100% pass | Yes |
| Integration tests | 100% pass | Yes |
| Code coverage | >= 90% | Yes |
| Live tests | Informational | No (advisory) |

---

## Performance Benchmark Approach

### Benchmarks to Track

1. **RLM Loop Throughput** -- Steps per second with MockLLMClient
2. **Large Content Handling** -- Time to peek/grep on 1M character content
3. **Strategy Comparison Overhead** -- Time to run MultiStrategyEvaluator
4. **Budget Tracker Overhead** -- Time per `check_limits()` call
5. **Token Estimation Speed** -- `estimate_tokens()` calls per second

### Benchmark Execution

```bash
# Run benchmarks (future: pytest-benchmark integration)
pytest tests/benchmarks/ -m benchmark --benchmark-only
```

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| RLM step (mock LLM) | < 10ms | Excluding LLM latency |
| peek() on 1M chars | < 50ms | String slicing |
| grep() on 1M chars | < 200ms | Regex search |
| estimate_tokens() | < 1ms | Simple heuristic |
| BudgetTracker.check_limits() | < 0.1ms | No I/O |

---

## Test Naming Conventions

```python
# File naming
test_<module>.py                    # Unit tests
tests/integration/test_<feature>.py # Integration tests

# Class naming
class Test<Feature>:               # Group related tests
class Test<Feature>Integration:    # Integration test groups

# Method naming
def test_<action>_<scenario>(self): # Descriptive name
def test_<action>_<edge_case>(self): # Edge cases explicit
```

---

## Failure Triage Process

1. **Red CI** -- Investigate within 1 hour during work hours.
2. **Flaky test** -- Add `@pytest.mark.flaky` temporarily, file issue, fix within 1 week.
3. **Coverage drop** -- Identify uncovered lines, add tests before merging.
4. **Live test failure** -- Check provider status first, then investigate code changes.
