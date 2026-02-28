## Summary

Brief description of what this PR does and why.

## Changes

-
-

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Refactoring (no functional changes)
- [ ] Documentation update
- [ ] Evaluation / benchmarking

## Checklist

- [ ] Tests pass (`uv run pytest`)
- [ ] No hardcoded API keys, model names, or secrets
- [ ] Prompts are in `prompts/` as versioned YAML (no inline strings)
- [ ] LLM calls are wrapped in traced functions with logging
- [ ] Type hints added for new functions
- [ ] Docstrings added for new public APIs

## Retrieval / Quality Impact

If this PR changes chunking, retrieval, reranking, or generation:

- [ ] Eval results included below
- [ ] No regression in recall@k / precision@k
- Metrics before:
- Metrics after:

## Related Issues

Closes #
