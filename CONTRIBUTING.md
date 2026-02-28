# Contributing to RLMKit

Thank you for your interest in contributing to RLMKit! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) for dependency management
- Docker (for Neo4j and vector DB services)
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/gosha70/rlmkit.git
cd rlmkit

# Install dependencies
uv sync

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Start backing services
docker compose up -d

# Verify everything works
uv run pytest
```

## Project Conventions

Please review `CLAUDE.md` for the full set of project conventions. Key points:

### Prompts

- All prompts live as versioned YAML files in `prompts/`. Never use inline prompt strings.

### LLM Calls

- Every LLM call must be wrapped in a traced function with latency and token logging.
- The embeddings model is pinned in config — never hardcode model names.

### RAG Pipeline

- Default to hybrid search (vector + BM25 keyword) before graph traversal.
- Chunk overlap should be 10–15% of chunk size.
- Never embed raw HTML/Markdown; always clean to plain text first.
- A reranking step is required before final context assembly.
- All retrieval results must include source references for citation.

### Knowledge Graph

- Graph schema is defined in code. Changes require migration scripts.
- Graph traversal depth is configurable (default: 2 hops).

### Code Style

- **Type hints** on all functions.
- **Docstrings** on all public APIs.
- Use **Ruff** for linting and **Black** for formatting.
- Target **80%+** test coverage.

## How to Contribute

### Reporting Bugs

Use the [Bug Report](https://github.com/gosha70/rlmkit/issues/new?template=bug_report.md) issue template. Include:

- Steps to reproduce
- Expected vs. actual behavior
- Python version, OS, and relevant model/provider info

### Suggesting Features

Use the [Feature Request](https://github.com/gosha70/rlmkit/issues/new?template=feature_request.md) issue template.

### Submitting Pull Requests

1. **Fork** the repository and create a branch from `main`.
2. **Follow** the coding standards described above.
3. **Write tests** — no live API calls in unit tests. Use mock LLM response fixtures.
4. **Run the test suite** before submitting:
   ```bash
   uv run pytest
   ```
5. **Fill out the PR template** completely.
6. **Keep PRs focused** — one logical change per PR.

### Testing Guidelines

- Use `pytest` with fixtures for mock LLM responses.
- Retrieval evaluation: measure `recall@k` and `precision@k` on labeled datasets.
- Answer quality: use LLM-as-judge with rubrics in `eval/rubrics/`.
- Graph tests: validate schema constraints and Cypher query correctness.
- If your change affects retrieval or generation quality, include eval results in your PR description.

## Project Structure Overview

| Directory | Purpose |
|-----------|----------|
| `src/rlmkit/` | Core library code |
| `src/rlmkit/envs/` | Execution environments |
| `src/rlmkit/llm/` | LLM provider integrations |
| `tests/` | Test suite |
| `benchmarks/` | Performance benchmarks |
| `examples/` | Usage examples |
| `frontend/` | Production UI (React+Vite) |
| `docker/` | Docker configurations |

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
