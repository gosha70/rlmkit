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

# Basic installation
pip install -e ".[dev]"

# With UI support (Streamlit, charts, file processing)
pip install -e ".[ui]"

# Everything (recommended for development)
pip install -e ".[all]"
```

### Basic Usage

```python
from rlmkit import RLM, RLMConfig
from rlmkit.llm import MockLLMClient

# Create LLM client (use OpenAI, Anthropic, etc. in production)
client = MockLLMClient([
    "```python\nprint(len(P))\n```",
    "FINAL: The document is about..."
])

# Create RLM instance
rlm = RLM(client=client, config=RLMConfig())

# Run on large content
result = rlm.run(
    prompt="Your large document content here...",
    query="What is this document about?"
)

print(result.answer)
# Output: The document is about...
```

### 🆕 NEW: RLM vs Direct Mode Comparison

```python
from rlmkit import RLM, RLMConfig
from rlmkit.llm import get_llm_client

# Create LLM client
client = get_llm_client(provider="openai", model="gpt-4")

# Create RLM instance
rlm = RLM(client=client, config=RLMConfig())

# Compare RLM vs Direct mode
result = rlm.run_comparison(
    prompt="Large document...",
    query="Summarize this"
)

# View comparison metrics
summary = result.get_summary()
print(f"Token Savings: {summary['token_savings']['savings_percent']:.1f}%")
print(f"Recommendation: {summary['recommendation']}")
```

### 🎨 Interactive UI

Launch the Streamlit web interface for interactive testing:

```bash
# Install UI dependencies
pip install -e ".[ui]"

# Launch the app
streamlit run src/rlmkit/ui/app.py
```

Features:
- 📁 Upload PDF, DOCX, TXT, and other files
- 🔄 Toggle between RLM and Direct modes
- 📊 Compare performance metrics with interactive charts
- 💾 Export results as JSON or Markdown
- 📈 Visualize token usage, costs, and execution time

## 📦 Current Status: Week 6 - Polish & Release! ✨

### ✅ Completed Features

#### Core Functionality (Weeks 1-5)
- **PyReplEnv**: Python REPL-like execution environment
- **Code Execution**: Run Python code with stdout/stderr capture
- **Exception Handling**: Safe exception capture and reporting
- **Persistent State**: Variables persist across executions
- **Content Access**: Large content available via `P` variable
- **Safety Layer**: Sandbox restrictions for secure execution
- **Execution Limits**: Timeout mechanism
- **Content Tools**: peek, grep, chunk, select for navigation
- **Controller Loop**: LLM-driven exploration
- **Recursion**: subcall support for nested queries
- **Provider Integration**: OpenAI, Anthropic, Ollama, LM Studio, vLLM
- **Budget Tracking**: Token usage, costs, and limits

#### New in Week 6: Polish & Release Features
- **🔄 RLM Toggle**: Easily switch between RLM and Direct mode
- **📊 Comparison Mode**: Run both modes and compare metrics side-by-side
- **📈 Performance Metrics**: Track tokens, cost, time, and efficiency
- **🎨 Interactive UI**: Streamlit web interface for testing
- **📁 File Processing**: Support for PDF, DOCX, TXT, MD, JSON, and code files
- **📉 Visualizations**: Interactive charts comparing RLM vs Direct mode
- **💾 Export Results**: JSON and Markdown report generation
- **Test Coverage**: 96% with comprehensive test suite

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

Copyright (c) 2026 EGOGE - All Rights Reserved.

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This software may be used and distributed according to the terms of the MIT license.

## 🙏 Acknowledgments

Built following Shape UP principles for incremental delivery.

---

**Status**: Week 1 - Day 2 ✅ | **Next**: Safety Layer Implementation
