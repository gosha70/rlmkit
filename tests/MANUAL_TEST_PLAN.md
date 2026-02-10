# RLMKit Manual Integration Test Plan

**Version:** 1.0
**Date:** February 8, 2026
**Owner:** QA Lead
**Scope:** End-to-end manual validation of RLMKit features

---

## Overview

This document defines manual integration test cases for RLMKit. Each test case is
assigned a priority level:

- **P0 (Critical Path)** -- Must pass for any release. Blocks deployment.
- **P1 (Important)** -- Should pass for a quality release. Can be waived with risk acceptance.
- **P2 (Nice to Have)** -- Enhances confidence. Failure triggers a bug but not a release block.

---

## Test Cases

### TC-001: First-Time Setup and Provider Configuration
**Priority:** P0
**Preconditions:** Clean environment, no prior RLMKit configuration.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Clone the repository and run `pip install -e ".[all]"` | Installation succeeds without errors |
| 2 | Run `python -c "import rlmkit; print(rlmkit.__version__)"` | Prints version string (e.g. `0.1.0`) |
| 3 | Set `OPENAI_API_KEY` environment variable to a valid key | Variable is set |
| 4 | Run `python -c "from rlmkit.llm import auto_detect_provider; print(auto_detect_provider())"` | Prints `openai` |
| 5 | Run `pytest tests/ -x --timeout=60` | All existing tests pass |

**Pass Criteria:** All steps succeed. Provider auto-detection returns the correct provider.

---

### TC-002: Direct Chat Mode -- Simple Query
**Priority:** P0
**Preconditions:** Valid provider configured (API key set or local model running).

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Create a short content string (< 100 words) | Content prepared |
| 2 | Call `interact(content, "Summarize this", mode="direct")` | Returns `InteractResult` |
| 3 | Check `result.answer` is non-empty | Answer contains relevant text |
| 4 | Check `result.mode_used == "direct"` | Mode is `direct` |
| 5 | Check `result.metrics["total_tokens"] > 0` | Token usage tracked |
| 6 | Check `result.metrics["execution_time"] > 0` | Execution time recorded |

**Pass Criteria:** Non-empty answer returned, correct mode, metrics populated.

---

### TC-003: RLM Mode -- Multi-Step Code Execution
**Priority:** P0
**Preconditions:** Valid provider configured. Content of 500+ words available.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Prepare content > 500 words (e.g., a long article) | Content prepared |
| 2 | Call `interact(content, "What are the main themes?", mode="rlm")` | Returns `InteractResult` |
| 3 | Check `result.mode_used == "rlm"` | Mode is `rlm` |
| 4 | Check `result.raw_result.steps >= 1` | At least one execution step |
| 5 | Inspect `result.raw_result.trace` | Trace contains assistant and execution entries |
| 6 | Verify trace shows code blocks being executed | Code execution visible in trace |

**Pass Criteria:** RLM runs multi-step exploration, trace shows code execution, answer is coherent.

---

### TC-004: RAG Mode -- Retrieval Augmented Generation
**Priority:** P1
**Preconditions:** OpenAI API key set (for embeddings). Content of 1000+ words.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Prepare long content (1000+ words) | Content prepared |
| 2 | Call `interact(content, "Find specific detail X", mode="rag")` | Returns `InteractResult` |
| 3 | Check `result.mode_used == "rag"` | Mode is `rag` |
| 4 | Verify answer references specific parts of the content | Relevant retrieval |
| 5 | Check token count is lower than direct mode on same content | RAG uses fewer tokens |

**Pass Criteria:** RAG retrieves relevant chunks and answer is grounded in content.

---

### TC-005: Compare Mode -- Multi-Strategy Evaluation
**Priority:** P0
**Preconditions:** Valid provider configured. Content of 500+ words.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Create a `MockLLMClient` with appropriate responses | Client created |
| 2 | Create `DirectStrategy` and `RLMStrategy` with the client | Strategies created |
| 3 | Create `MultiStrategyEvaluator` with both strategies | Evaluator created |
| 4 | Call `evaluator.evaluate(content, query)` | Returns `EvaluationResult` |
| 5 | Check `result.results` contains both `"direct"` and `"rlm"` keys | Both strategies ran |
| 6 | Call `result.get_summary()` | Summary includes `fastest`, `cheapest` |
| 7 | Call `result.get_comparison("direct", "rlm")` | Comparison dict returned |

**Pass Criteria:** Both strategies execute, results are comparable, summary identifies winners.

---

### TC-006: Document Upload and Processing
**Priority:** P1
**Preconditions:** UI running (Streamlit). PDF and DOCX test files available.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Launch Streamlit UI | UI loads without errors |
| 2 | Upload a PDF file (10+ pages) via the file uploader | File accepted, content extracted |
| 3 | Submit a query about the document content | Response generated |
| 4 | Upload a DOCX file | File accepted, content extracted |
| 5 | Upload a plain text file | File accepted |
| 6 | Attempt to upload an unsupported file type (e.g., .exe) | Rejected with clear error message |

**Pass Criteria:** Supported file types are processed correctly. Unsupported types rejected.

---

### TC-007: Deep Recursion (Depth > 1)
**Priority:** P1
**Preconditions:** MockLLMClient configured to produce subcall actions.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Create a MockLLMClient that returns a subcall action, then FINAL | Client configured |
| 2 | Create RLM with `RLMConfig` and run on content | Subcall is executed |
| 3 | Inspect execution trace for subcall entries | Trace shows nested call |
| 4 | Verify answer incorporates subcall results | Answer references sub-results |
| 5 | Set `max_recursion_depth=1` and trigger deep recursion | BudgetExceeded raised at depth limit |

**Pass Criteria:** Subcalls execute and return results. Depth limits enforced.

---

### TC-008: Budget Enforcement -- Token Limit
**Priority:** P0
**Preconditions:** None.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Create `BudgetLimits(max_tokens=100)` | Limits configured |
| 2 | Create `BudgetTracker` with the limits | Tracker created |
| 3 | Add LLM calls totaling > 100 tokens | Tokens accumulated |
| 4 | Call `tracker.check_limits()` | `BudgetExceeded` raised |
| 5 | Verify error message includes current and max token counts | Clear error message |

**Pass Criteria:** BudgetExceeded raised when token limit exceeded. Message is informative.

---

### TC-009: Budget Enforcement -- Cost Limit
**Priority:** P0
**Preconditions:** None.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Create `BudgetLimits(max_cost=0.05)` with `CostTracker` pricing | Limits configured |
| 2 | Add LLM calls that accumulate cost > $0.05 | Cost accumulated |
| 3 | Call `tracker.check_limits()` | `BudgetExceeded` raised |
| 4 | Verify the utilization percentage via `get_utilization()` | Cost utilization > 1.0 |

**Pass Criteria:** Cost tracking accurate. Limit enforced at threshold.

---

### TC-010: Budget Enforcement -- Step Limit (via RLM)
**Priority:** P0
**Preconditions:** None.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Create MockLLMClient that never returns FINAL (always code) | Client configured |
| 2 | Set `RLMConfig.execution.max_steps = 3` | Config set |
| 3 | Run `RLM.run(prompt, query)` | `BudgetExceeded` raised |
| 4 | Verify error message mentions max_steps=3 | Clear error message |

**Pass Criteria:** RLM stops at step limit. Error message accurate.

---

### TC-011: Provider Switching Mid-Session
**Priority:** P1
**Preconditions:** At least two providers configured (or use MockLLMClient instances).

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Run a query with provider A (e.g., MockLLMClient A) | Result from provider A |
| 2 | Create a new strategy with provider B | Strategy created |
| 3 | Run the same query with provider B | Result from provider B |
| 4 | Verify both results are valid but may differ | Both results valid |
| 5 | Verify metrics are tracked independently per provider | Separate metrics |

**Pass Criteria:** Provider switching works cleanly. No state leakage between providers.

---

### TC-012: Error Recovery -- Invalid API Key
**Priority:** P1
**Preconditions:** None.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Create a strategy with an invalid API key or failing client | Strategy created |
| 2 | Run a query | StrategyResult returned with `success=False` |
| 3 | Check `result.error` contains meaningful message | Error is descriptive |
| 4 | Verify no unhandled exception propagates | Graceful failure |
| 5 | Fix the client and retry | Query succeeds |

**Pass Criteria:** Errors are caught gracefully. No crashes. Recovery works after fix.

---

### TC-013: Sandbox Security -- Blocked Imports
**Priority:** P0
**Preconditions:** Safe mode enabled.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Create `PyReplEnv` with `safe_mode=True` | Environment created |
| 2 | Execute code containing `import os` | `SecurityError` raised or import blocked |
| 3 | Execute code containing `import subprocess` | Import blocked |
| 4 | Execute code containing `open("/etc/passwd")` | Operation blocked |
| 5 | Execute code using allowed module `import json` | Import succeeds |
| 6 | Execute code using `import re` | Import succeeds |

**Pass Criteria:** Dangerous imports and operations blocked. Safe modules allowed.

---

### TC-014: Concurrent Sessions -- No Cross-Talk
**Priority:** P1
**Preconditions:** None.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Create two separate RLM instances with different MockLLMClients | Instances created |
| 2 | Run queries on both instances | Both return results |
| 3 | Verify instance A's result does not contain instance B's data | No cross-talk |
| 4 | Verify execution environments are isolated (variables do not leak) | Isolation confirmed |

**Pass Criteria:** Sessions are fully isolated. No shared mutable state.

---

### TC-015: Export and Persistence
**Priority:** P1
**Preconditions:** SQLite storage configured.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Create a `Database` and `ConversationStore` | Store created |
| 2 | Save a conversation with messages and metadata | Conversation saved |
| 3 | Retrieve the conversation by ID | Conversation matches saved data |
| 4 | Verify messages are ordered correctly | Correct ordering |
| 5 | Delete the conversation | Conversation removed |
| 6 | Attempt to retrieve deleted conversation | Returns None or empty |

**Pass Criteria:** Conversations persist across save/load cycles. Deletion works.

---

### TC-016: Auto Mode Selection
**Priority:** P1
**Preconditions:** None.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Call `interact(short_content, query, mode="auto")` where short_content < 8K tokens | `mode_used == "direct"` |
| 2 | Call `interact(medium_content, query, mode="auto")` where medium is 8K-100K tokens | `mode_used == "rag"` |
| 3 | Call `interact(large_content, query, mode="auto")` where large > 100K tokens | `mode_used == "rlm"` |

**Pass Criteria:** Auto mode selects appropriate strategy based on content size thresholds.

---

### TC-017: Multi-Model Cost Optimization
**Priority:** P1
**Preconditions:** None (uses ModelConfig with mock verification).

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Create `ModelConfig.cost_optimized("openai")` | Config has `gpt-4o` root, `gpt-4o-mini` sub |
| 2 | Create `ModelConfig.cost_optimized("anthropic")` | Config has sonnet root, haiku sub |
| 3 | Create `ModelConfig.from_single_model("gpt-4o")` | Same model for root and sub |
| 4 | Verify `ModelMetrics` tracks root vs. sub usage separately | Metrics split correctly |
| 5 | Call `metrics.savings_vs_single_model(cost_per_token)` | Returns positive savings value |

**Pass Criteria:** Multi-model config resolves correctly. Metrics track per-model usage.

---

### TC-018: Execution Trace Completeness
**Priority:** P1
**Preconditions:** None.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Run RLM with MockLLMClient returning 3 code steps then FINAL | RLM completes |
| 2 | Inspect `result.trace` | Contains alternating assistant/execution entries |
| 3 | Each trace entry has `step` number | Step numbers sequential |
| 4 | Execution entries have `raw_result` with `stdout` | Raw output captured |
| 5 | Final assistant entry is the FINAL response | Trace ends with FINAL |

**Pass Criteria:** Trace is complete, ordered, and contains all execution details.

---

### TC-019: Performance -- Large Document
**Priority:** P2
**Preconditions:** MockLLMClient (to isolate from API latency).

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Generate content of 1 million characters | Content created |
| 2 | Create RLM with MockLLMClient (peek + FINAL responses) | RLM configured |
| 3 | Run and measure wall-clock time | Completes in < 5 seconds |
| 4 | Verify `peek()` returns correct substring of large content | Content access correct |
| 5 | Monitor memory usage stays reasonable (< 500MB above baseline) | Memory bounded |

**Pass Criteria:** Large content handled without excessive memory or time.

---

### TC-020: Dark Mode / Theme (UI)
**Priority:** P2
**Preconditions:** Streamlit UI running.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Open UI in default theme | UI renders correctly |
| 2 | Toggle to dark mode (if available) or set Streamlit dark theme | Theme switches |
| 3 | Navigate all pages (Chat, Metrics, Configuration) | All pages readable |
| 4 | Verify charts and graphs adapt to theme | No invisible elements |
| 5 | Toggle back to light mode | Theme reverts cleanly |

**Pass Criteria:** All UI elements visible and readable in both themes.

---

## Summary

| Priority | Count | Test Case IDs |
|----------|-------|---------------|
| P0 | 7 | TC-001, TC-002, TC-003, TC-005, TC-008, TC-010, TC-013 |
| P1 | 11 | TC-004, TC-006, TC-007, TC-009, TC-011, TC-012, TC-014, TC-015, TC-016, TC-017, TC-018 |
| P2 | 2 | TC-019, TC-020 |
| **Total** | **20** | |

## Execution Notes

- Tests marked P0 should be run before every release candidate.
- P1 tests should be run during the full regression cycle.
- P2 tests are recommended during cool-down periods.
- For tests requiring live API keys, use the `@pytest.mark.live` marker in automated equivalents.
- Mock-based tests can run in CI without any external dependencies.
