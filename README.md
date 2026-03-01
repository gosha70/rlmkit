# RLMKit

**Recursive Language Model toolkit** — a Python library that lets LLMs write code to explore content that exceeds their context window.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## What is RLMKit?

LLMs have fixed context windows. When your document is too large to fit in a single prompt, you lose information. RLMKit solves this by giving the LLM a Python sandbox where it can write code to navigate and analyze content dynamically — exploring only what's relevant.

RLMKit provides:

- **Three execution modes** — Direct (full context), RAG (retrieval), and RLM (recursive code generation)
- **Auto mode** — selects the best strategy based on content size
- **100+ LLM providers** — via LiteLLM (OpenAI, Anthropic, Ollama, LM Studio, and more)
- **Budget controls** — cap tokens, cost, steps, and time
- **Sandboxed execution** — RestrictedPython prevents file/network/system access
- **[RLM Studio](#rlm-studio)** — a web app for experimenting, tuning, and comparing execution modes

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync --extra all

# Using pip
pip install -e ".[all]"
```

### Usage

```python
from rlmkit import interact

result = interact(
    content="Your document text here...",
    query="What is this about?",
    provider="openai",
    model="gpt-4o"
)

print(result.answer)
print(f"Tokens: {result.metrics['total_tokens']:,}")
print(f"Cost: ${result.metrics['total_cost']:.4f}")
print(f"Mode used: {result.mode_used}")
```

### Execution Modes

```python
# Direct — send full content in one LLM call (small documents)
result = interact(content, query, mode="direct")

# RAG — retrieval-augmented generation (medium documents)
result = interact(content, query, mode="rag")

# RLM — LLM writes Python to explore content recursively (large documents)
result = interact(content, query, mode="rlm")

# Auto — let RLMKit choose based on content size (default)
result = interact(content, query, mode="auto")

# Compare — run both RLM and Direct, return metrics for each
result = interact(content, query, mode="compare")
```

| Mode | Best For | How It Works |
|------|----------|--------------|
| `direct` | < 8K tokens | Full context in single LLM call |
| `rag` | 8K-100K tokens | Chunk, embed, retrieve relevant pieces |
| `rlm` | > 100K tokens | LLM writes code to navigate content via `peek()`, `grep()`, `chunk()` |
| `auto` | Any size | Selects direct/rag/rlm based on token count |
| `compare` | Benchmarking | Runs both RLM and Direct, compares metrics |

### Configuration

```python
result = interact(
    content=large_document,
    query="Summarize the key findings",
    mode="auto",
    provider="anthropic",
    model="claude-sonnet-4-5",
    max_steps=16,          # RLM loop budget
    temperature=0.7,
    verbose=True           # Print progress
)
```

### Supported Providers

Set the API key as an environment variable, then pass the provider name:

| Provider | Env Variable | Example Model |
|----------|-------------|---------------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` |
| Ollama | (local, no key) | `llama3` |
| LM Studio | (local, no key) | Any served model |
| Google | `GOOGLE_API_KEY` | `gemini-pro` |
| 100+ more | via LiteLLM | See [LiteLLM docs](https://docs.litellm.ai/) |

## RLM Studio

RLM Studio is a web application for experimenting with RLMKit interactively. Use it to demo RLM capabilities, tune runtime parameters, compare execution modes side by side, and monitor cost/performance metrics.

### Starting RLM Studio

```bash
# Terminal 1: Backend API server
uv run uvicorn src.rlmkit.server.app:app --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

Open [http://localhost:3001](http://localhost:3001) in your browser.

### What You Can Do

- **Chat** — Upload documents and query them using one or more Chat Providers in parallel. Responses appear in a side-by-side column layout with per-response metrics (tokens, cost, latency).
- **Dashboard** — View aggregated metrics per session: total tokens, cost, average latency, token savings. Charts break down performance by provider and execution mode.
- **Traces** — Inspect every execution step-by-step. See the code the LLM generated, the output it received, token counts, and timing for each step. Visualize as timeline, tree, or raw code.
- **Settings** — Configure providers, create Chat Providers, set budgets, manage profiles, customize system prompts, and switch themes.

For a complete walkthrough of all RLM Studio features, see **[docs/rlm-studio-guide.md](docs/rlm-studio-guide.md)**.

## Architecture

```
src/rlmkit/
├── domain/            # Entities, ports (zero external deps)
├── application/       # Use cases (run_rlm, etc.)
├── infrastructure/    # Adapters (LiteLLM, sandbox, storage)
│   ├── llm/           # LiteLLMAdapter (100+ providers)
│   └── sandbox/       # RestrictedPython sandbox
├── core/              # Legacy RLM controller, PyReplEnv
├── server/            # FastAPI API + WebSocket server
│   └── routes/        # chat, providers, sessions, traces, metrics
└── api.py             # Public interact() / complete() API

frontend/              # Next.js 16 + React 19 + shadcn/ui
├── src/app/           # Pages: chat, dashboard, traces, settings
├── src/components/    # Shared UI components
└── src/lib/           # API client, hooks
```

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=rlmkit --cov-report=term-missing

# Frontend type check
cd frontend && npx tsc --noEmit
```

## License

Copyright (c) 2026 EGOGE. MIT License — see [LICENSE](LICENSE).
