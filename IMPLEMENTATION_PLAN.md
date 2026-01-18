# RLMKit Implementation Plan

## Executive Summary

**Goal:** Build an intelligent context middleware (RLM Toolkit) that acts as a middleman between users and LLMs, managing large content dynamically to avoid context window limitations while maintaining accuracy.

**Approach:** Code generation-based exploration where the LLM writes Python code to navigate and analyze large content.

**Timeline:** 6-week Shape UP cycle with incremental vertical slices

**Methodology:** Test-driven development with 80%+ coverage per cycle

---

## Design Decisions (Locked In)

### ✅ 1. No Artificial Context Limits
- RLM dynamically manages content, not restricts it
- The system works around context limitations, not enforces them
- Budget tracking is for step/recursion control, not content blocking

### ✅ 2. Code Generation Approach
- LLM writes Python code to explore content
- More flexible than declarative tool calls
- Enables complex multi-step exploration logic

### ✅ 3. Skip MCP Integration (Initially)
- Focus on core functionality first
- MCP can be added later as extension

### ✅ 4. Provider-Agnostic Design
- Works with any local LLM (custom fine-tuned models)
- Generic HTTP interface (OpenAI-compatible)
- Supports Ollama, LM Studio, vLLM, etc.

### ✅ 5. Shape UP SDLC
- 6-week cycles with clear milestones
- Vertical slices deliver working functionality
- Test coverage at each increment

---

## Core Innovation: RLM as Advanced RAG++

### Traditional RAG Limitations:
- Static retrieval: embed → search → inject top-k
- No dynamic exploration
- Can't refine search based on findings
- Limited context management

### RLM Advantages:
- **Dynamic inspection:** LLM decides what to explore
- **Recursive refinement:** Can drill down into relevant sections
- **Strategic selection:** Only accesses what's needed
- **State management:** Maintains exploration history
- **Never exceeds context:** Smart content windowing

---

## Architecture Overview

```
┌───────────────────────────────────────────────┐
│  User Query + Large Content P (50K+ tokens)   │
└─────────────────┬─────────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────────┐
│  RLM Controller                                │
│  • Manages message history                     │
│  • Enforces step/depth budgets                │
│  • Parses LLM responses (code vs final)       │
│  • Tracks exploration state                   │
└─────────────────┬─────────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────────┐
│  LLM Provider (Local Custom Model)             │
│  • Input: conversation history + tool docs     │
│  • Output: Python code OR final answer        │
└─────────────────┬─────────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────────┐
│  PyReplEnv (Sandboxed Execution)               │
│  • Executes Python code safely                 │
│  • Provides: P (content) + tools              │
│  • Restricted: imports, builtins, timeout     │
│  • Captures: stdout, stderr, exceptions       │
└─────────────────┬─────────────────────────────┘
                  │
           ┌──────┴──────┐
           ▼             ▼
    ┌──────────┐   ┌──────────┐
    │  Tools   │   │ subcall  │
    │  • peek  │   │ (recurse)│
    │  • grep  │   └──────────┘
    │  • chunk │
    │  • select│
    └──────────┘
```

---

## Repository Structure

```
rlmkit/
  pyproject.toml              # Package configuration
  README.md                   # User documentation
  IMPLEMENTATION_PLAN.md      # This file
  LICENSE                     # MIT
  .gitignore
  
  src/
    rlmkit/
      __init__.py             # Public API exports
      
      core/
        __init__.py
        rlm.py               # Main RLM class
        config.py            # RLMConfig
        trace.py             # Trace logging
        errors.py            # Custom exceptions
        parsing.py           # Code/FINAL extraction
      
      envs/
        __init__.py
        pyrepl_env.py        # Python REPL environment
        sandbox.py           # Safe execution
        tools.py             # Built-in tools (peek, grep, etc.)
      
      llm/
        __init__.py
        base.py              # LLMClient protocol
        http_client.py       # Generic HTTP client
        mock_client.py       # For testing
      
      controller/
        __init__.py
        loop.py              # Main control loop
        prompts.py           # System prompt templates
  
  examples/
    basic_qa.py              # Simple Q&A example
    recursive_summary.py     # Multi-level summarization
    large_doc_analysis.py    # Full document analysis
  
  tests/
    conftest.py              # Pytest configuration
    
    test_parsing.py          # Code extraction tests
    test_sandbox.py          # Security tests
    test_tools.py            # Tool function tests
    test_env.py              # Environment tests
    test_loop_mock.py        # Loop with mock LLM
    test_integration.py      # End-to-end tests
  
  docs/
    architecture.md          # System design
    safety.md               # Security model
    prompting.md            # LLM prompt engineering
```

---

## 6-Week Shape UP Cycle

### Hill Chart Metaphor
Each cycle represents a "hill" to climb:
- **Uphill (figuring out):** Design, exploration, unknowns
- **Downhill (executing):** Implementation, testing, delivery

---

## Week 1: Execution Foundation

### Hill 1: "LLM can write and execute Python code safely"

#### **Vertical Slice 1.1: Basic Execution (Day 1-2)**

**Goal:** Execute Python code and capture output

**Deliverables:**
- [ ] Repository structure created
- [ ] `pyproject.toml` with dependencies
- [ ] `PyReplEnv` class (basic version)
- [ ] Execute code, capture stdout/stderr
- [ ] Return structured result dict

**Tests:**
```python
def test_execute_simple_code():
    env = PyReplEnv()
    result = env.execute("x = 1 + 1\nprint(x)")
    assert result['stdout'] == "2\n"
    assert result['exception'] is None

def test_capture_exception():
    env = PyReplEnv()
    result = env.execute("1 / 0")
    assert result['exception'] is not None
    assert "ZeroDivisionError" in result['exception']
```

#### **Vertical Slice 1.2: Safety Layer (Day 3-4)**

**Goal:** Prevent dangerous operations

**Deliverables:**
- [ ] Restricted `__builtins__` (no `open`, `exec`, `eval`)
- [ ] Import whitelist mechanism
- [ ] Block `os`, `sys`, `subprocess`, `socket`
- [ ] `sandbox.py` module

**Tests:**
```python
def test_blocks_file_operations():
    env = PyReplEnv(safe_mode=True)
    result = env.execute("open('/etc/passwd', 'r')")
    assert result['exception'] is not None
    assert "not allowed" in result['exception'].lower()

def test_blocks_dangerous_imports():
    env = PyReplEnv(safe_mode=True)
    result = env.execute("import os")
    assert result['exception'] is not None

def test_allows_safe_imports():
    env = PyReplEnv(safe_mode=True, allowed_imports=['json'])
    result = env.execute("import json\nx = json.dumps({'a': 1})")
    assert result['exception'] is None
```

#### **Vertical Slice 1.3: Execution Limits (Day 5)**

**Goal:** Enforce timeouts and output limits

**Deliverables:**
- [ ] Timeout mechanism (signal or multiprocessing)
- [ ] Stdout/stderr truncation
- [ ] Execution time tracking

**Tests:**
```python
def test_timeout_infinite_loop():
    env = PyReplEnv(max_exec_time_s=1.0)
    result = env.execute("while True: pass")
    assert result['timeout'] is True

def test_truncates_huge_output():
    env = PyReplEnv(max_stdout_chars=100)
    result = env.execute("print('x' * 10000)")
    assert len(result['stdout']) <= 100
    assert result['truncated'] is True
```

**✅ Milestone 1: Can safely execute arbitrary Python code**

---

## Week 2: Content Navigation

### Hill 2: "Code can inspect large content P"

#### **Vertical Slice 2.1: Basic Access (Day 1-2)**

**Goal:** Access content via peek()

**Deliverables:**
- [ ] `P` injected into env globals
- [ ] `peek(start, end)` function
- [ ] Bounds checking
- [ ] Tools available in executed code

**Tests:**
```python
def test_peek_returns_substring():
    env = PyReplEnv()
    env.set_content("Hello World! This is a test.")
    result = env.execute("output = peek(0, 5)")
    assert env.get_var('output') == "Hello"

def test_peek_handles_out_of_bounds():
    env = PyReplEnv()
    env.set_content("Short")
    result = env.execute("output = peek(0, 1000)")
    assert len(env.get_var('output')) == 5  # Only returns what exists
```

#### **Vertical Slice 2.2: Search Tools (Day 3)**

**Goal:** Find patterns in content

**Deliverables:**
- [ ] `grep(pattern, max_matches, context)` function
- [ ] Returns match positions + snippets
- [ ] Output size limits enforced

**Tests:**
```python
def test_grep_finds_pattern():
    env = PyReplEnv()
    env.set_content("The cat and the dog. The bird and the cat.")
    result = env.execute("matches = grep(r'cat')")
    matches = env.get_var('matches')
    assert len(matches) == 2
    assert matches[0]['start'] == 4

def test_grep_respects_max_matches():
    env = PyReplEnv()
    env.set_content("test " * 100)
    result = env.execute("matches = grep(r'test', max_matches=5)")
    assert len(env.get_var('matches')) == 5
```

#### **Vertical Slice 2.3: Chunking & Selection (Day 4-5)**

**Goal:** Structure content and combine portions

**Deliverables:**
- [ ] `chunk(mode, size, overlap)` function
- [ ] `select(ranges)` function
- [ ] Performance optimized for large content

**Tests:**
```python
def test_chunk_by_lines():
    env = PyReplEnv()
    env.set_content("\n".join([f"Line {i}" for i in range(100)]))
    result = env.execute("ranges = chunk(mode='lines', size=10)")
    ranges = env.get_var('ranges')
    assert len(ranges) == 10  # 100 lines / 10 per chunk

def test_select_combines_ranges():
    env = PyReplEnv()
    env.set_content("0123456789" * 10)
    result = env.execute("text = select([(0, 5), (10, 15)])")
    assert env.get_var('text') == "0123401234"
```

**✅ Milestone 2: Can navigate 100K+ token documents**

---

## Week 3: Controller Loop

### Hill 3: "LLM decides what to explore, sees results, decides next step"

#### **Vertical Slice 3.1: Parsing (Day 1-2)**

**Goal:** Extract code blocks and final answers

**Deliverables:**
- [ ] `parsing.py` module
- [ ] Extract Python fenced code blocks
- [ ] Extract `FINAL: ...` answers
- [ ] Extract `FINAL_VAR: name` references

**Tests:**
```python
def test_extract_code_block():
    text = "Here's the code:\n```python\nx = 1\n```\nDone"
    code = extract_python_code(text)
    assert code == "x = 1"

def test_extract_final_answer():
    text = "After analysis...\nFINAL: The answer is 42"
    answer = extract_final(text)
    assert answer == "The answer is 42"

def test_extract_final_var():
    text = "FINAL_VAR: result"
    var_name = extract_final_var(text)
    assert var_name == "result"
```

#### **Vertical Slice 3.2: Basic Loop (Day 3-4)**

**Goal:** Controller orchestration

**Deliverables:**
- [ ] `RLM` class with `run(prompt, query)` method
- [ ] Message history management
- [ ] Execute code → capture result → continue loop
- [ ] Integration with PyReplEnv

**Tests:**
```python
def test_basic_loop_mock_llm():
    mock = MockLLMClient([
        "```python\nx = peek(0, 10)\n```",
        "```python\nFINAL('Found it')\n```"
    ])
    rlm = RLM(client=mock, config=RLMConfig())
    result = rlm.run(prompt="Large content...", query="Find X")
    assert result.answer == "Found it"
    assert result.stats['steps'] == 2
```

#### **Vertical Slice 3.3: Budgets & Error Handling (Day 5)**

**Goal:** Enforce limits and handle failures

**Deliverables:**
- [ ] `max_steps` enforcement
- [ ] `max_recursion_depth` tracking
- [ ] Graceful error handling
- [ ] Trace logging

**Tests:**
```python
def test_stops_at_max_steps():
    mock = MockLLMClient(["```python\npass\n```"] * 100)
    config = RLMConfig(max_steps=5)
    rlm = RLM(client=mock, config=config)
    result = rlm.run(prompt="test", query="test")
    assert result.stats['steps'] == 5

def test_handles_execution_errors():
    mock = MockLLMClient(["```python\n1/0\n```"])
    rlm = RLM(client=mock, config=RLMConfig())
    result = rlm.run(prompt="test", query="test")
    # Should not crash, should report error in trace
    assert "ZeroDivisionError" in str(result.trace)
```

**✅ Milestone 3: Complete controller loop working with mock LLM**

---

## Week 4: Recursion

### Hill 4: "Can recursively analyze content portions"

#### **Vertical Slice 4.1: subcall Design (Day 1-2)**

**Goal:** Nested RLM instances

**Deliverables:**
- [ ] `subcall(text, instruction)` function in env
- [ ] Creates new RLM instance with subcontext
- [ ] Returns result to parent
- [ ] Depth tracking

**Tests:**
```python
def test_subcall_basic():
    mock = MockLLMClient(["FINAL: Summarized"])
    rlm = RLM(client=mock, config=RLMConfig())
    env = PyReplEnv(rlm=rlm)
    result = env.execute("""
result = subcall("Long text...", instruction="Summarize")
""")
    assert "Summarized" in env.get_var('result')
```

#### **Vertical Slice 4.2: Depth Limits (Day 3)**

**Goal:** Prevent infinite recursion

**Deliverables:**
- [ ] Max depth enforcement
- [ ] Depth passed to child instances
- [ ] Clear error messages

**Tests:**
```python
def test_respects_max_recursion_depth():
    # Mock that always recurses
    mock = MockLLMClient(["```python\nsubcall('x', 'test')\n```"] * 10)
    config = RLMConfig(max_recursion_depth=2)
    rlm = RLM(client=mock, config=config)
    
    with pytest.raises(BudgetExceeded):
        rlm.run(prompt="test", query="test")
```

#### **Vertical Slice 4.3: Integration & Examples (Day 4-5)**

**Goal:** Real use cases

**Deliverables:**
- [ ] Map-reduce example
- [ ] Hierarchical summarization
- [ ] Trace shows recursion tree

**Tests:**
```python
def test_map_reduce_pattern():
    # Mock: chunk → subcall each → combine
    mock = MockLLMClient([
        "```python\nranges = chunk(mode='lines', size=10)\n```",
        "```python\nsummaries = [subcall(select([r]), 'summarize') for r in ranges]\n```",
        "FINAL: Combined summary"
    ])
    rlm = RLM(client=mock, config=RLMConfig(max_recursion_depth=5))
    result = rlm.run(prompt="Long doc...", query="Summarize")
    assert result.answer == "Combined summary"
```

**✅ Milestone 4: Recursive analysis working end-to-end**

---

## Week 5: Provider Integration

### Hill 5: "Works with real LLMs"

#### **Vertical Slice 5.1: Provider Interface (Day 1-2)**

**Goal:** Abstract LLM communication

**Deliverables:**
- [ ] `LLMClient` protocol
- [ ] `MockLLMClient` (already done)
- [ ] Message format specification
- [ ] Response parsing

**Tests:**
```python
def test_mock_client_interface():
    mock = MockLLMClient(["response 1", "response 2"])
    response = mock.complete([
        {"role": "user", "content": "test"}
    ])
    assert response == "response 1"
```

#### **Vertical Slice 5.2: HTTP Provider (Day 3-4)**

**Goal:** Connect to real LLMs

**Deliverables:**
- [ ] `HTTPLLMClient` for OpenAI-compatible APIs
- [ ] Works with Ollama, LM Studio, vLLM
- [ ] Configurable endpoint, model, temperature
- [ ] Error handling and retries

**Tests:**
```python
def test_http_client_ollama(live_server):
    # Integration test with real Ollama
    client = HTTPLLMClient(
        base_url="http://localhost:11434",
        model="llama3.1:8b"
    )
    response = client.complete([
        {"role": "user", "content": "Say 'test'"}
    ])
    assert len(response) > 0
```

#### **Vertical Slice 5.3: Prompt Engineering (Day 5)**

**Goal:** Effective system prompts

**Deliverables:**
- [ ] System prompt templates
- [ ] Tool documentation format
- [ ] Code generation examples
- [ ] FINAL instruction format

**Content:**
```python
SYSTEM_PROMPT = """
You are an RLM (Recursive Language Model) controller.

Your job: analyze large content P by writing Python code.

Available tools (already imported):
- peek(start, end) → str: Get substring of P
- grep(pattern, max_matches=20, context=0) → list: Search P
- chunk(mode='lines', size=200, overlap=20) → list: Get ranges
- select(ranges) → str: Combine multiple ranges
- subcall(text, instruction) → str: Recursively analyze portion

Response format:
1. To explore: Write a Python code block
   ```python
   matches = grep(r'climate change')
   relevant = select([m['range'] for m in matches[:5]])
   summary = subcall(relevant, "Summarize findings")
   ```

2. To answer: Use FINAL
   FINAL: Based on analysis, the answer is...

Rules:
- Don't print huge outputs (use limits)
- Store final answer in FINAL() or return via FINAL_VAR: varname
- You have {max_steps} steps and depth {max_recursion_depth}

Current query: {query}
"""
```

**✅ Milestone 5: Works with your custom local LLM**

---

## Week 6: Polish & Release

### Hill 6: "Production-ready v0.1.0"

#### **Vertical Slice 6.1: Examples (Day 1-2)**

**Deliverables:**
- [ ] `examples/basic_qa.py` - Simple Q&A
- [ ] `examples/recursive_summary.py` - Multi-level summarization
- [ ] `examples/large_doc_analysis.py` - Full analysis workflow
- [ ] Sample data files (large texts)

#### **Vertical Slice 6.2: Documentation (Day 3-4)**

**Deliverables:**
- [ ] `README.md` with quickstart
- [ ] `docs/architecture.md` - System design
- [ ] `docs/safety.md` - Security model
- [ ] `docs/prompting.md` - How to prompt your LLM
- [ ] API reference (docstrings)

#### **Vertical Slice 6.3: Packaging & Release (Day 5)**

**Deliverables:**
- [ ] `pyproject.toml` finalized
- [ ] `pip install -e .` works
- [ ] GitHub Actions CI (lint + test)
- [ ] Version 0.1.0 tagged
- [ ] PyPI release (optional)

**✅ Milestone 6: Production-ready package**

---

## Test Coverage Strategy

### Unit Tests (per module)
- `test_parsing.py` - Code extraction
- `test_tools.py` - peek, grep, chunk, select
- `test_sandbox.py` - Security restrictions
- `test_env.py` - PyReplEnv functionality

### Integration Tests (per cycle)
- `test_loop_mock.py` - Full loop with mock LLM
- `test_recursion.py` - Nested subcalls
- `test_budgets.py` - Step/depth limits

### End-to-End Tests (final)
- `test_integration.py` - Real scenarios
- Performance tests with large content (100K+ tokens)
- Error handling edge cases

### Coverage Target
- **Week 1-2:** 80% coverage
- **Week 3-4:** 85% coverage
- **Week 5-6:** 90% coverage

### CI/CD Pipeline
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=rlmkit --cov-report=term
      - run: ruff check .
```

---

## API Design (Public Interface)

### Core Classes

```python
from rlmkit import RLM, RLMConfig, RLMResult
from rlmkit.llm import HTTPLLMClient

# Configuration
config = RLMConfig(
    max_steps=16,
    max_recursion_depth=6,
    max_exec_time_s=5.0,
    max_stdout_chars=1000,
    allowed_imports=['json', 're', 'math'],
    sandbox_mode='safe'
)

# LLM Client
client = HTTPLLMClient(
    base_url="http://localhost:11434",
    model="custom-model:latest",
    temperature=0.0
)

# Create RLM
rlm = RLM(client=client, config=config)

# Run analysis
result: RLMResult = rlm.run(
    prompt=large_content,  # 50K tokens
    query="What are the main findings about X?"
)

# Access results
print(result.answer)           # Final answer
print(result.stats)            # {steps: 8, recursion_calls: 3, ...}
print(result.trace)            # Full execution trace
```

### Result Object

```python
@dataclass
class RLMResult:
    answer: str                    # Final answer
    trace: Trace                   # Full trajectory
    stats: dict                    # Execution statistics
    
    # Stats includes:
    # - steps: int
    # - recursion_calls: int
    # - total_exec_time: float
    # - tools_used: dict[str, int]
```

---

## Success Criteria

### Functional Requirements
- ✅ Handles content > 100K tokens without exceeding context window
- ✅ LLM can write Python to explore content
- ✅ Safe code execution (no file/network access)
- ✅ Recursive analysis via subcall
- ✅ Works with local custom LLMs
- ✅ Complete trace logging

### Quality Requirements
- ✅ 80%+ test coverage
- ✅ No critical security vulnerabilities
- ✅ Documentation for all public APIs
- ✅ Working examples
- ✅ `pip install` works

### Performance Requirements
- ✅ Handles 100K token content in < 60s (with local LLM)
- ✅ Code execution timeout < 5s per step
- ✅ Memory efficient (streaming where possible)

---

## Risk Mitigation

### Technical Risks

**Risk 1: Sandbox Escape**
- *Mitigation:* Use restricted builtins, import whitelist, timeout
- *Testing:* Comprehensive security test suite
- *Documentation:* Clear warnings in docs/safety.md

**Risk 2: LLM Generates Invalid Code**
- *Mitigation:* Robust parsing, clear error messages, retry logic
- *Testing:* Test with various LLM outputs
- *Documentation:* Prompt engineering guide

**Risk 3: Performance with Very Large Content**
- *Mitigation:* Lazy loading, chunking, efficient search
- *Testing:* Performance benchmarks with 100K+ tokens
- *Documentation:* Performance tips in docs

**Risk 4: Local LLM Compatibility**
- *Mitigation:* Generic HTTP interface, flexible parsing
- *Testing:* Test with multiple LLM backends
- *Documentation:* Provider setup guide

---

## Future Enhancements (Post v0.1.0)

### Phase 2 (Weeks 7-9)
- [ ] MCP integration for external tools
- [ ] Streaming responses
- [ ] Persistent state/cache
- [ ] Advanced chunking strategies
- [ ] Semantic search integration

### Phase 3 (Weeks 10-12)
- [ ] Multi-document analysis
- [ ] Incremental updates (add content without restarting)
- [ ] Parallel subcalls
- [ ] Cost tracking (token usage)
- [ ] Web UI for exploration visualization

---

## Getting Started (Week 1, Day 1)

When ready to begin implementation:

1. **Setup repository:**
   ```bash
   cd /Users/gosha/dev/repo/rlmkit
   git init
   ```

2. **Create pyproject.toml:**
   - Python >= 3.10
   - Dependencies: pytest, ruff, black
   - Package: src/rlmkit/

3. **First test:**
   - Create `tests/test_env.py`
   - Write `test_execute_simple_code()`
   - Make it pass

4. **First implementation:**
   - Create `src/rlmkit/envs/pyrepl_env.py`
   - Implement basic `PyReplEnv.execute(code)`
   - Capture stdout/stderr

**Let's build! 🚀**

---

## Notes & Decisions Log

### 2026-01-18: Initial Planning
- Confirmed code generation approach vs declarative tools
- Decided to skip MCP in v0.1.0
- Agreed on 6-week Shape UP cycle
- Prioritized test coverage at each slice

### Design Philosophy
- **Explicit over implicit:** Clear configuration, no magic
- **Safety by default:** Sandbox mode on by default
- **Provider agnostic:** Works with any LLM via simple interface
- **Incremental delivery:** Working software every week
- **Test-driven:** Tests before or alongside implementation

---

*This plan is a living document. Update as design evolves.*
