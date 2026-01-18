# RLMKit

**Recursive Language Model toolkit** for managing large context through code generation.

[![Tests](https://img.shields.io/badge/tests-15%20passed-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

## 🎯 What is RLMKit?

RLMKit is an intelligent context middleware that acts as a middleman between users and LLMs. Instead of hitting context window limits, RLMKit enables LLMs to **write Python code** to explore and analyze large content dynamically.

### The Problem
- Traditional RAG: Static retrieval, no refinement
- Context windows: Fixed limits (4K, 8K, 32K tokens)
- Large documents: Can't fit everything into a single prompt

### The RLMKit Solution
- **Dynamic exploration:** LLM writes code to inspect content
- **Recursive refinement:** Can drill down into relevant sections
- **Strategic access:** Only loads what's needed
- **Never exceeds context:** Smart content windowing

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from rlmkit.envs import PyReplEnv

# Create execution environment
env = PyReplEnv()

# Set large content (could be 100K+ tokens)
env.set_content("Your large document content here...")

# Execute code to explore content
result = env.execute("""
# P is the large content
print(f"Content length: {len(P)}")
print(f"First 100 chars: {P[:100]}")
""")

print(result['stdout'])
# Output:
# Content length: ...
# First 100 chars: ...
```

## 📦 Current Status: Week 1 - Execution Foundation

### ✅ Completed (Vertical Slice 1.1 & 1.2)
- **PyReplEnv**: Python REPL-like execution environment
- **Code Execution**: Run Python code with stdout/stderr capture
- **Exception Handling**: Safe exception capture and reporting
- **Persistent State**: Variables persist across executions
- **Content Access**: Large content available via `P` variable
- **Output Limits**: Automatic truncation of huge outputs
- **Test Coverage**: 96% with 15 passing tests

### 🚧 In Progress (Week 1)
- **Safety Layer**: Sandbox restrictions (Day 3-4)
- **Execution Limits**: Timeout mechanism (Day 5)

### 📅 Upcoming (Weeks 2-6)
- **Week 2**: Content navigation tools (peek, grep, chunk, select)
- **Week 3**: Controller loop (LLM-driven exploration)
- **Week 4**: Recursion via subcall
- **Week 5**: Provider integration (local LLMs)
- **Week 6**: Polish, examples, documentation

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=rlmkit --cov-report=term-missing

# Run specific test file
pytest tests/test_env.py -v
```

## 🏗️ Architecture

```
┌───────────────────────────────────┐
│  User Query + Large Content P     │
└────────────┬──────────────────────┘
             │
             ▼
┌───────────────────────────────────┐
│  RLM Controller (Coming Week 3)   │
└────────────┬──────────────────────┘
             │
             ▼
┌───────────────────────────────────┐
│  LLM Provider (Coming Week 5)     │
└────────────┬──────────────────────┘
             │
             ▼
┌───────────────────────────────────┐
│  PyReplEnv ✅ (Current)           │
│  • Executes Python code           │
│  • Captures output                │
│  • Handles exceptions             │
└───────────────────────────────────┘
```

## 📚 Documentation

- **[Implementation Plan](IMPLEMENTATION_PLAN.md)**: Complete 6-week development plan
- **[Architecture](docs/architecture.md)**: System design (Coming Week 6)
- **[Safety](docs/safety.md)**: Security model (Coming Week 6)

## 🛠️ Development

### Project Structure

```
rlmkit/
├── src/rlmkit/
│   ├── core/           # Core functionality
│   │   ├── errors.py   # Exception classes ✅
│   │   └── ...
│   ├── envs/           # Execution environments
│   │   └── pyrepl_env.py  # Python REPL ✅
│   ├── llm/            # LLM providers (Week 5)
│   └── controller/     # Control loop (Week 3)
├── tests/              # Test suite
│   └── test_env.py     # Environment tests ✅
├── examples/           # Usage examples (Week 6)
└── docs/               # Documentation (Week 6)
```

### Coding Standards

- **Python 3.10+** required
- **Type hints** everywhere
- **Docstrings** for public APIs
- **Test coverage** 80%+ target
- **Ruff** for linting
- **Black** for formatting

### Contributing

We follow **Shape UP** methodology:
- 6-week cycles
- Vertical slices (working features incrementally)
- Test-driven development

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

Built following Shape UP principles for incremental delivery.

---

**Status**: Week 1 - Day 2 ✅ | **Next**: Safety Layer Implementation
